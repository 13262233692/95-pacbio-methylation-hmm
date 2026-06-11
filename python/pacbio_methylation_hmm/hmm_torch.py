"""
PyTorch-accelerated HMM for methylation detection.

Supports batch processing on GPU/CPU for high-throughput analysis.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass


@dataclass
class TorchHMMConfig:
    n_states: int = 2
    device: str = "cpu"

    initial_probs: np.ndarray = None
    trans_probs: np.ndarray = None
    emission_means: np.ndarray = None
    emission_vars: np.ndarray = None

    def __post_init__(self):
        if self.initial_probs is None:
            self.initial_probs = np.array([0.8, 0.2])
        if self.trans_probs is None:
            self.trans_probs = np.array([[0.95, 0.05], [0.10, 0.90]])
        if self.emission_means is None:
            self.emission_means = np.array([0.0, 1.5])
        if self.emission_vars is None:
            self.emission_vars = np.array([0.5, 1.0])


class MethylationHMMPyTorch(nn.Module):
    """
    PyTorch-based HMM for batch methylation state decoding.

    Supports GPU acceleration and batch processing of multiple read sequences.
    """

    STATE_UNMETHYLATED = 0
    STATE_METHYLATED = 1

    def __init__(self, config: Optional[TorchHMMConfig] = None):
        super().__init__()
        self.config = config if config is not None else TorchHMMConfig()
        self.device = torch.device(self.config.device)

        self.log_initial = nn.Parameter(
            torch.log(torch.tensor(self.config.initial_probs, dtype=torch.float64, device=self.device) + 1e-300),
            requires_grad=False,
        )
        self.log_trans = nn.Parameter(
            torch.log(torch.tensor(self.config.trans_probs, dtype=torch.float64, device=self.device) + 1e-300),
            requires_grad=False,
        )
        self.emission_means = nn.Parameter(
            torch.tensor(self.config.emission_means, dtype=torch.float64, device=self.device),
            requires_grad=False,
        )
        self.emission_vars = nn.Parameter(
            torch.tensor(self.config.emission_vars, dtype=torch.float64, device=self.device),
            requires_grad=False,
        )

    def _gaussian_log_pdf(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Compute log emission probabilities.

        Args:
            observations: (B, T) tensor

        Returns:
            log_emission: (B, T, n_states) tensor
        """
        B, T = observations.shape
        n_states = self.emission_means.shape[0]

        means = self.emission_means.view(1, 1, n_states)
        vars = self.emission_vars.view(1, 1, n_states)
        obs_expanded = observations.unsqueeze(-1).expand(B, T, n_states)

        log_probs = -0.5 * (
            torch.log(2 * np.pi * vars)
            + (obs_expanded - means) ** 2 / vars
        )
        return log_probs

    @staticmethod
    def _logsumexp(x: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
        return torch.logsumexp(x, dim=dim, keepdim=keepdim)

    def forward_algorithm(self, observations: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized forward algorithm for batched variable-length sequences.

        Args:
            observations: (B, T_max) padded observation tensor
            lengths: (B,) tensor of actual sequence lengths

        Returns:
            log_alpha: (B, T_max, n_states) forward variables
            log_evidence: (B,) log P(O) per sequence
        """
        B, T_max = observations.shape
        n_states = self.log_initial.shape[0]

        log_emission = self._gaussian_log_pdf(observations)

        log_alpha = torch.full((B, T_max, n_states), -float("inf"), dtype=torch.float64, device=self.device)
        log_alpha[:, 0, :] = self.log_initial.unsqueeze(0) + log_emission[:, 0, :]

        for t in range(1, T_max):
            prev = log_alpha[:, t - 1, :].unsqueeze(-1)
            trans = self.log_trans.unsqueeze(0)
            scores = self._logsumexp(prev + trans, dim=1)
            log_alpha[:, t, :] = scores + log_emission[:, t, :]

            mask = (t < lengths).unsqueeze(-1)
            log_alpha[:, t, :] = torch.where(mask, log_alpha[:, t, :], torch.full_like(log_alpha[:, t, :], -float("inf")))

        batch_idx = torch.arange(B, device=self.device)
        final_alpha = log_alpha[batch_idx, lengths - 1, :]
        log_evidence = self._logsumexp(final_alpha, dim=1)

        return log_alpha, log_evidence

    def backward_algorithm(self, observations: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Vectorized backward algorithm.

        Args:
            observations: (B, T_max) padded tensor
            lengths: (B,) tensor

        Returns:
            log_beta: (B, T_max, n_states) backward variables
        """
        B, T_max = observations.shape
        n_states = self.log_initial.shape[0]

        log_emission = self._gaussian_log_pdf(observations)

        log_beta = torch.full((B, T_max, n_states), -float("inf"), dtype=torch.float64, device=self.device)

        batch_idx = torch.arange(B, device=self.device)
        log_beta[batch_idx, lengths - 1, :] = 0.0

        for t in range(T_max - 2, -1, -1):
            next_emit = log_emission[:, t + 1, :].unsqueeze(1)
            next_beta = log_beta[:, t + 1, :].unsqueeze(1)
            trans = self.log_trans.unsqueeze(0)
            log_beta[:, t, :] = self._logsumexp(trans + next_emit + next_beta, dim=2)

            mask = (t < lengths).unsqueeze(-1)
            log_beta[:, t, :] = torch.where(mask, log_beta[:, t, :], torch.full_like(log_beta[:, t, :], -float("inf")))

        return log_beta

    def compute_posterior(self, observations: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute posterior probabilities via Forward-Backward.

        Returns:
            posteriors: (B, T_max, n_states)
            log_alpha: (B, T_max, n_states)
            log_evidence: (B,)
        """
        log_emission = self._gaussian_log_pdf(observations)
        log_alpha, log_evidence = self.forward_algorithm(observations, lengths)
        log_beta = self.backward_algorithm(observations, lengths)

        log_posterior = log_alpha + log_beta - log_evidence.unsqueeze(-1).unsqueeze(-1)
        log_posterior = log_posterior - self._logsumexp(log_posterior, dim=2, keepdim=True)
        posteriors = torch.exp(log_posterior)

        return posteriors, log_alpha, log_evidence

    def viterbi_batch(self, observations: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batched Viterbi decoding.

        Returns:
            states: (B, T_max) most likely state sequences
            log_delta: (B, T_max, n_states)
        """
        B, T_max = observations.shape
        n_states = self.log_initial.shape[0]

        log_emission = self._gaussian_log_pdf(observations)

        log_delta = torch.full((B, T_max, n_states), -float("inf"), dtype=torch.float64, device=self.device)
        psi = torch.zeros((B, T_max, n_states), dtype=torch.long, device=self.device)

        log_delta[:, 0, :] = self.log_initial.unsqueeze(0) + log_emission[:, 0, :]

        for t in range(1, T_max):
            prev = log_delta[:, t - 1, :].unsqueeze(-1)
            trans = self.log_trans.unsqueeze(0)
            scores = prev + trans
            best_scores, psi[:, t, :] = torch.max(scores, dim=1)
            log_delta[:, t, :] = best_scores + log_emission[:, t, :]

            mask = (t < lengths).unsqueeze(-1)
            log_delta[:, t, :] = torch.where(mask, log_delta[:, t, :], torch.full_like(log_delta[:, t, :], -float("inf")))

        states = torch.zeros((B, T_max), dtype=torch.long, device=self.device)
        batch_idx = torch.arange(B, device=self.device)
        final_delta = log_delta[batch_idx, lengths - 1, :]
        states[batch_idx, lengths - 1] = torch.argmax(final_delta, dim=1)

        for t in range(T_max - 2, -1, -1):
            next_states = states[:, t + 1]
            states[:, t] = psi[batch_idx, t + 1, next_states]

            mask = (t < lengths)
            states[:, t] = torch.where(mask, states[:, t], torch.zeros_like(states[:, t]))

        return states, log_delta

    @torch.no_grad()
    def predict_batch(
        self,
        observations: np.ndarray,
        lengths: Optional[np.ndarray] = None,
        threshold: float = 0.5,
    ) -> List[Dict]:
        """
        Predict methylation states for a batch of sequences.

        Args:
            observations: numpy array (B, T) or list of 1D arrays
            lengths: optional (B,) array of lengths (for padded batches)
            threshold: probability threshold for methylation call

        Returns:
            list of dicts with keys: states, viterbi_states, methylation_prob, posteriors
        """
        if isinstance(observations, np.ndarray):
            B, T = observations.shape
            if lengths is None:
                lengths = np.full(B, T, dtype=np.int64)
            obs_tensor = torch.tensor(observations, dtype=torch.float64, device=self.device)
            len_tensor = torch.tensor(lengths, dtype=torch.long, device=self.device)
        else:
            B = len(observations)
            T = max(len(o) for o in observations)
            padded = np.zeros((B, T), dtype=np.float64)
            lengths_arr = np.zeros(B, dtype=np.int64)
            for i, o in enumerate(observations):
                padded[i, :len(o)] = o
                lengths_arr[i] = len(o)
            obs_tensor = torch.tensor(padded, dtype=torch.float64, device=self.device)
            len_tensor = torch.tensor(lengths_arr, dtype=torch.long, device=self.device)
            lengths = lengths_arr

        posteriors, _, log_evidence = self.compute_posterior(obs_tensor, len_tensor)
        viterbi_states, _ = self.viterbi_batch(obs_tensor, len_tensor)

        posteriors_np = posteriors.cpu().numpy()
        viterbi_np = viterbi_states.cpu().numpy()
        log_evidence_np = log_evidence.cpu().numpy()

        results = []
        for i in range(B):
            L = lengths[i]
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
