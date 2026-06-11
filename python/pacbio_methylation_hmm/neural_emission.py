"""
Neural Emission Network: Deep Bi-LSTM feature extractor for HMM emission modeling.

Replaces the fixed Gaussian emission assumption with a data-driven neural network
that captures rich nucleotide context and IPD signal features.

Architecture:
  Input: nucleotides (B, T) long + ipd (B, T) float
    ├── Nucleotide Embedding (5 → embed_dim)
    ├── IPD Linear Projection (1 → embed_dim)
    ├── Optional PulseWidth Linear (1 → embed_dim)
    └── Concat → (B, T, input_dim)
        └── Bi-LSTM × n_layers (input_dim → hidden_dim*2)
            └── Emission Head (hidden_dim*2 → n_states)
                └── LogSoftmax → log P(O_t | q_t)  [pseudo emission]

The output log emission probabilities are directly consumed by the
Scaled Forward-Backward and Normalized Log-Viterbi HMM decoders.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


NUCLEOTIDE_VOCAB = {
    65: 0, 67: 1, 71: 2, 84: 3,
    97: 0, 99: 1, 103: 2, 116: 3,
    78: 4, 110: 4,
}
BASE_A, BASE_C, BASE_G, BASE_T, BASE_N = 0, 1, 2, 3, 4
VOCAB_SIZE = 5


@dataclass
class NeuralEmissionConfig:
    embed_dim: int = 16
    hidden_dim: int = 128
    n_lstm_layers: int = 3
    n_states: int = 2
    dropout: float = 0.1
    use_pulse_width: bool = False
    emission_log_clip: float = -50.0


class NeuralEmissionNetwork(nn.Module):
    """
    Deep Bi-LSTM network that outputs per-frame pseudo log emission probabilities.

    Takes nucleotide sequence context and IPD signal features as input,
    processes them through a multi-layer bidirectional LSTM, and produces
    log emission probabilities for each HMM state at each position.

    The key insight: instead of assuming P(IPD | state) ~ Gaussian,
    we let the Bi-LSTM learn arbitrary emission distributions conditioned
    on the full sequence context — capturing dependencies that a simple
    Gaussian cannot model (e.g., CpG dinucleotide context effects,
    sequence-dependent polymerase kinetics, local sequence composition).
    """

    def __init__(self, config: Optional[NeuralEmissionConfig] = None):
        super().__init__()
        self.config = config if config is not None else NeuralEmissionConfig()

        self.nucleotide_embedding = nn.Embedding(
            VOCAB_SIZE, self.config.embed_dim, padding_idx=4
        )

        self.ipd_projection = nn.Linear(1, self.config.embed_dim)

        input_dim = self.config.embed_dim * 2

        if self.config.use_pulse_width:
            self.pw_projection = nn.Linear(1, self.config.embed_dim)
            input_dim += self.config.embed_dim
        else:
            self.pw_projection = None

        self.input_layer_norm = nn.LayerNorm(input_dim)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.config.hidden_dim,
            num_layers=self.config.n_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.config.dropout if self.config.n_lstm_layers > 1 else 0.0,
        )

        lstm_output_dim = self.config.hidden_dim * 2

        self.emission_head = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(lstm_output_dim // 2, self.config.n_states),
        )

        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.emission_log_clip = self.config.emission_log_clip

    def forward(
        self,
        nucleotides: torch.Tensor,
        ipd_values: torch.Tensor,
        pulse_width: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass: compute pseudo log emission probabilities.

        Args:
            nucleotides: (B, T) long tensor of base indices (0=A, 1=C, 2=G, 3=T, 4=N)
            ipd_values: (B, T) float tensor of normalized IPD values
            pulse_width: Optional (B, T) float tensor of normalized PulseWidth values
            lengths: Optional (B,) long tensor of sequence lengths for packing

        Returns:
            log_emission: (B, T, n_states) pseudo log emission probabilities
        """
        base_emb = self.nucleotide_embedding(nucleotides)

        ipd_input = ipd_values.unsqueeze(-1)
        ipd_emb = self.ipd_projection(ipd_input)

        features = torch.cat([base_emb, ipd_emb], dim=-1)

        if self.pw_projection is not None and pulse_width is not None:
            pw_input = pulse_width.unsqueeze(-1)
            pw_emb = self.pw_projection(pw_input)
            features = torch.cat([features, pw_emb], dim=-1)

        features = self.input_layer_norm(features)

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                features, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True, total_length=features.size(1)
            )
        else:
            lstm_out, _ = self.lstm(features)

        logits = self.emission_head(lstm_out)

        log_emission = self.log_softmax(logits)

        log_emission = torch.clamp(log_emission, min=self.emission_log_clip, max=0.0)

        return log_emission


def encode_nucleotides(bases: np.ndarray) -> torch.Tensor:
    """
    Convert raw base arrays to embedded nucleotide indices.

    Args:
        bases: numpy array of ASCII values or 0-15 BAM encoding

    Returns:
        LongTensor of nucleotide indices (0-4)
    """
    if bases.dtype == np.uint8 and bases.max() <= 15:
        mapping = torch.tensor([4, 0, 1, 4, 2, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4],
                               dtype=torch.long)
        return mapping[torch.from_numpy(bases.astype(np.int64))]
    else:
        result = np.full(bases.shape, 4, dtype=np.int64)
        for ascii_val, idx in NUCLEOTIDE_VOCAB.items():
            result[bases == ascii_val] = idx
        return torch.from_numpy(result)
