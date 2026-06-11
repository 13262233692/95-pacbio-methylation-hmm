"""
Neural-HMM Hybrid Decoder: Bi-LSTM pseudo emissions + Scaled HMM log decoding.

This module implements the complete hybrid architecture:

  1. NeuralEmissionNetwork (Bi-LSTM) computes per-frame pseudo log emission
     probabilities P(O_t | q_t) conditioned on nucleotide context + IPD features.

  2. The Scaled Forward-Backward and Normalized Log-Viterbi algorithms consume
     these neural emissions exactly as they would consume Gaussian log emissions,
     preserving full numerical stability for ultra-long HiFi reads.

This hybrid design gets the best of both worlds:
  - Neural network: captures rich, non-linear, context-dependent emission patterns
  - HMM decoder: provides structured temporal smoothing via transition constraints
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

from .neural_emission import (
    NeuralEmissionNetwork,
    NeuralEmissionConfig,
    encode_nucleotides,
)


@dataclass
class NeuralHMMConfig:
    n_states: int = 2
    device: str = "cpu"

    initial_probs: np.ndarray = None
    trans_probs: np.ndarray = None

    embed_dim: int = 16
    hidden_dim: int = 128
    n_lstm_layers: int = 3
    dropout: float = 0.1
    use_pulse_width: bool = False
    emission_log_clip: float = -50.0

    learn_trans: bool = True
    trans_init_scale: float = 0.1

    def __post_init__(self):
        if self.initial_probs is None:
            self.initial_probs = np.array([0.8, 0.2])
        if self.trans_probs is None:
            self.trans_probs = np.array([[0.95, 0.05], [0.10, 0.90]])


class NeuralHMM(nn.Module):
    """
    Hybrid Neural-HMM for methylation state decoding.

    Architecture:
      Bi-LSTM Encoder → Pseudo Log Emissions → Scaled HMM Decoder

    The Bi-LSTM replaces the fixed Gaussian emission model, learning
    arbitrary emission patterns from data. The HMM decoder then performs
    structured temporal inference using the numerically stable Scaled
    Forward-Backward and Normalized Log-Viterbi algorithms.
    """

    STATE_UNMETHYLATED = 0
    STATE_METHYLATED = 1

    def __init__(self, config: Optional[NeuralHMMConfig] = None):
        super().__init__()
        self.config = config if config is not None else NeuralHMMConfig()
        self.device = torch.device(self.config.device)

        emission_config = NeuralEmissionConfig(
            embed_dim=self.config.embed_dim,
            hidden_dim=self.config.hidden_dim,
            n_lstm_layers=self.config.n_lstm_layers,
            n_states=self.config.n_states,
            dropout=self.config.dropout,
            use_pulse_width=self.config.use_pulse_width,
            emission_log_clip=self.config.emission_log_clip,
        )
        self.emission_net = NeuralEmissionNetwork(emission_config)

        log_init = torch.log(
            torch.tensor(self.config.initial_probs, dtype=torch.float64) + 1e-300
        )
        if self.config.learn_trans:
            init_logits = torch.log(
                torch.tensor(self.config.trans_probs, dtype=torch.float64) + 1e-300
            ) / self.config.trans_init_scale
            self.log_trans_logits = nn.Parameter(init_logits)
            self.log_initial = nn.Parameter(log_init.clone())
        else:
            self.log_initial = nn.Parameter(log_init, requires_grad=False)
            self.log_trans_logits = nn.Parameter(
                torch.log(
                    torch.tensor(self.config.trans_probs, dtype=torch.float64) + 1e-300
                ),
                requires_grad=False,
            )

        self.trans_init_scale = self.config.trans_init_scale
        self.emission_log_clip = self.config.emission_log_clip

    @property
    def log_trans(self) -> torch.Tensor:
        return F.log_softmax(self.log_trans_logits, dim=-1)

    def _neural_log_emission(
        self,
        nucleotides: torch.Tensor,
        ipd_values: torch.Tensor,
        pulse_width: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute pseudo log emission probabilities via Bi-LSTM.

        Returns:
            log_emission: (B, T, n_states) in log space, clamped for stability
        """
        ipd_f32 = ipd_values.to(torch.float32)
        pw_f32 = pulse_width.to(torch.float32) if pulse_width is not None else None
        log_emit = self.emission_net(nucleotides, ipd_f32, pw_f32, lengths)
        return log_emit.to(torch.float64)

    @staticmethod
    def _logsumexp(x: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
        return torch.logsumexp(x, dim=dim, keepdim=keepdim)

    def forward_scaled(
        self,
        log_emission: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Scaled Forward algorithm with per-step log-normalization (batched).

        Uses neural pseudo emissions instead of Gaussian emissions.
        All numerical stability guarantees from the Scaled Forward algorithm
        are preserved for sequences of arbitrary length.

        Args:
            log_emission: (B, T, n_states) from Bi-LSTM
            lengths: (B,) sequence lengths

        Returns:
            log_alpha_hat: (B, T, n_states) normalized forward variables
            log_C: (B, T) per-step log normalization constants
            log_evidence: (B,) log P(O) per sequence
        """
        B, T_max, n_states = log_emission.shape
        log_A = self.log_trans

        log_alpha_hat = torch.zeros((B, T_max, n_states), dtype=torch.float64, device=self.device)
        log_C = torch.zeros((B, T_max), dtype=torch.float64, device=self.device)

        log_alpha_hat[:, 0, :] = self.log_initial.unsqueeze(0) + log_emission[:, 0, :]
        log_C[:, 0] = self._logsumexp(log_alpha_hat[:, 0, :], dim=1)
        log_alpha_hat[:, 0, :] = log_alpha_hat[:, 0, :] - log_C[:, 0].unsqueeze(-1)

        for t in range(1, T_max):
            prev = log_alpha_hat[:, t - 1, :].unsqueeze(-1)
            trans = log_A.unsqueeze(0)
            log_alpha_hat[:, t, :] = log_emission[:, t, :] + self._logsumexp(prev + trans, dim=1)

            log_C[:, t] = self._logsumexp(log_alpha_hat[:, t, :], dim=1)
            log_alpha_hat[:, t, :] = log_alpha_hat[:, t, :] - log_C[:, t].unsqueeze(-1)

            valid_mask = (t < lengths)
            log_alpha_hat[:, t, :] = torch.where(
                valid_mask.unsqueeze(-1),
                log_alpha_hat[:, t, :],
                torch.zeros_like(log_alpha_hat[:, t, :]),
            )
            log_C[:, t] = torch.where(valid_mask, log_C[:, t], torch.zeros_like(log_C[:, t]))

        log_evidence = torch.zeros(B, dtype=torch.float64, device=self.device)
        for b in range(B):
            log_evidence[b] = log_C[b, :lengths[b]].sum()

        return log_alpha_hat, log_C, log_evidence

    def backward_scaled(
        self,
        log_emission: torch.Tensor,
        lengths: torch.Tensor,
        log_C: torch.Tensor,
    ) -> torch.Tensor:
        """
        Scaled Backward algorithm using forward pass normalization constants.
        """
        B, T_max, n_states = log_emission.shape
        log_A = self.log_trans

        log_beta_hat = torch.zeros((B, T_max, n_states), dtype=torch.float64, device=self.device)
        for b in range(B):
            log_beta_hat[b, lengths[b] - 1, :] = 0.0

        for t in range(T_max - 2, -1, -1):
            next_emit = log_emission[:, t + 1, :].unsqueeze(1)
            next_beta = log_beta_hat[:, t + 1, :].unsqueeze(1)
            trans = log_A.unsqueeze(0)
            log_beta_hat[:, t, :] = self._logsumexp(trans + next_emit + next_beta, dim=2)

            log_beta_hat[:, t, :] = log_beta_hat[:, t, :] + log_C[:, t + 1].unsqueeze(-1)

            valid_mask = (t < lengths)
            log_beta_hat[:, t, :] = torch.where(
                valid_mask.unsqueeze(-1),
                log_beta_hat[:, t, :],
                torch.zeros_like(log_beta_hat[:, t, :]),
            )

        return log_beta_hat

    def compute_posterior(
        self,
        log_emission: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute posterior probabilities via Scaled Forward-Backward.
        """
        log_alpha_hat, log_C, log_evidence = self.forward_scaled(log_emission, lengths)
        log_beta_hat = self.backward_scaled(log_emission, lengths, log_C)

        log_gamma = log_alpha_hat + log_beta_hat
        log_gamma = log_gamma - self._logsumexp(log_gamma, dim=2, keepdim=True)
        posteriors = torch.exp(log_gamma)

        return posteriors, log_alpha_hat, log_evidence

    def viterbi_batch(
        self,
        log_emission: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Normalized Log-Viterbi algorithm (batched).
        """
        B, T_max, n_states = log_emission.shape
        log_A = self.log_trans

        log_delta_hat = torch.zeros((B, T_max, n_states), dtype=torch.float64, device=self.device)
        psi = torch.zeros((B, T_max, n_states), dtype=torch.long, device=self.device)

        log_delta_hat[:, 0, :] = self.log_initial.unsqueeze(0) + log_emission[:, 0, :]
        delta_max = log_delta_hat[:, 0, :].max(dim=1, keepdim=True)[0]
        log_delta_hat[:, 0, :] = log_delta_hat[:, 0, :] - delta_max

        for t in range(1, T_max):
            prev = log_delta_hat[:, t - 1, :].unsqueeze(-1)
            trans = log_A.unsqueeze(0)
            scores = prev + trans
            best_scores, psi[:, t, :] = torch.max(scores, dim=1)
            log_delta_hat[:, t, :] = best_scores + log_emission[:, t, :]

            delta_max = log_delta_hat[:, t, :].max(dim=1, keepdim=True)[0]
            log_delta_hat[:, t, :] = log_delta_hat[:, t, :] - delta_max

            valid_mask = (t < lengths)
            log_delta_hat[:, t, :] = torch.where(
                valid_mask.unsqueeze(-1),
                log_delta_hat[:, t, :],
                torch.zeros_like(log_delta_hat[:, t, :]),
            )

        states = torch.zeros((B, T_max), dtype=torch.long, device=self.device)
        batch_idx = torch.arange(B, device=self.device)
        final_delta = log_delta_hat[batch_idx, lengths - 1, :]
        states[batch_idx, lengths - 1] = torch.argmax(final_delta, dim=1)

        for t in range(T_max - 2, -1, -1):
            next_states = states[:, t + 1]
            states[:, t] = psi[batch_idx, t + 1, next_states]
            valid_mask = (t < lengths)
            states[:, t] = torch.where(valid_mask, states[:, t], torch.zeros_like(states[:, t]))

        return states, log_delta_hat

    def compute_loss(
        self,
        nucleotides: torch.Tensor,
        ipd_values: torch.Tensor,
        lengths: torch.Tensor,
        labels: torch.Tensor,
        pulse_width: Optional[torch.Tensor] = None,
        lambda_hmm: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute hybrid training loss:
          L = CE_loss + λ_hmm * (-log P(O))

        Args:
            nucleotides: (B, T) long tensor
            ipd_values: (B, T) float tensor
            lengths: (B,) long tensor
            labels: (B, T) long tensor (0=unmethylated, 1=methylated)
            pulse_width: Optional (B, T) float tensor
            lambda_hmm: Weight for HMM evidence term

        Returns:
            dict with loss, ce_loss, hmm_loss
        """
        log_emission = self._neural_log_emission(nucleotides, ipd_values, pulse_width, lengths)

        B, T_max, n_states = log_emission.shape

        mask = torch.zeros((B, T_max), dtype=torch.bool, device=self.device)
        for b in range(B):
            mask[b, :lengths[b]] = True

        log_emission_flat = log_emission[mask]
        labels_flat = labels[mask]
        ce_loss = F.nll_loss(log_emission_flat, labels_flat)

        with torch.no_grad():
            _, _, log_evidence = self.forward_scaled(log_emission, lengths)
        hmm_loss = (-log_evidence / lengths.float().to(self.device)).mean()

        total_loss = ce_loss + lambda_hmm * hmm_loss

        return {
            "loss": total_loss,
            "ce_loss": ce_loss.detach(),
            "hmm_loss": hmm_loss.detach(),
        }

    @torch.no_grad()
    def predict_batch(
        self,
        nucleotides: torch.Tensor,
        ipd_values: torch.Tensor,
        lengths: torch.Tensor,
        pulse_width: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
    ) -> List[Dict]:
        """
        Full prediction: Bi-LSTM emission → HMM decoding.

        Returns:
            list of dicts with states, viterbi_states, posteriors, methylation_prob, log_evidence
        """
        log_emission = self._neural_log_emission(nucleotides, ipd_values, pulse_width, lengths)

        posteriors, _, log_evidence = self.compute_posterior(log_emission, lengths)
        viterbi_states, _ = self.viterbi_batch(log_emission, lengths)

        posteriors_np = posteriors.cpu().numpy()
        viterbi_np = viterbi_states.cpu().numpy()
        log_evidence_np = log_evidence.cpu().numpy()

        results = []
        for i in range(nucleotides.shape[0]):
            L = int(lengths[i])
            post = posteriors_np[i, :L]
            meth_prob = post[:, self.STATE_METHYLATED]
            states = (meth_prob >= threshold).astype(np.int32)

            results.append({
                "states": states,
                "viterbi_states": viterbi_np[i, :L].astype(np.int32),
                "posteriors": post,
                "methylation_prob": meth_prob,
                "log_evidence": float(log_evidence_np[i]),
            })

        return results

    def to_device(self, device: str):
        self.device = torch.device(device)
        return self.to(self.device)
