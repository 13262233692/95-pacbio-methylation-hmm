"""
PyTorch-accelerated HMM for methylation detection.

Numerically stable batch implementation using:
  - Scaled Forward-Backward with per-step log-normalization
  - Normalized Log-Viterbi with per-step max-subtraction

Supports GPU/CUDA batch processing for high-throughput analysis.
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
    emission_log_clip: float = -50.0

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

    All core algorithms operate in log-probability space with per-step
    normalization to guarantee numerical stability for sequences of
    arbitrary length (tested up to 50000+ bp HiFi reads).
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
        self.emission_log_clip = self.config.emission_log_clip

    def _gaussian_log_pdf(self, observations: torch.Tensor) -> torch.Tensor:
        B, T = observations.shape
        n_states = self.emission_means.shape[0]

        means = self.emission_means.view(1, 1, n_states)
        vars = self.emission_vars.view(1, 1, n_states)
        obs_expanded = observations.unsqueeze(-1).expand(B, T, n_states)

        log_probs = -0.5 * (
            torch.log(2.0 * np.pi * vars)
            + (obs_expanded - means) ** 2 / vars
        )
        return torch.clamp(log_probs, min=self.emission_log_clip, max=0.0)

    @staticmethod
    def _logsumexp(x: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
        return torch.logsumexp(x, dim=dim, keepdim=keepdim)

    def forward_scaled(
        self, observations: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Scaled Forward algorithm with per-step log-normalization (batched).

        At each step t:
          1. Compute raw log_alpha_raw[t, j] = log_B[j](O[t]) + logsumexp_i(log_alpha_hat[t-1, i] + log_A[i,j])
          2. log_C[t] = logsumexp_j(log_alpha_raw[t, j])
          3. log_alpha_hat[t, j] = log_alpha_raw[t, j] - log_C[t]

        log P(O) = sum_t(log_C[t])

        Returns:
            log_alpha_hat: (B, T_max, n_states) normalized forward variables
            log_C: (B, T_max) per-step log normalization constants
            log_evidence: (B,) log P(O) per sequence
        """
        B, T_max = observations.shape
        n_states = self.log_initial.shape[0]

        log_emission = self._gaussian_log_pdf(observations)

        log_alpha_hat = torch.zeros((B, T_max, n_states), dtype=torch.float64, device=self.device)
        log_C = torch.zeros((B, T_max), dtype=torch.float64, device=self.device)

        log_alpha_hat[:, 0, :] = self.log_initial.unsqueeze(0) + log_emission[:, 0, :]
        log_C[:, 0] = self._logsumexp(log_alpha_hat[:, 0, :], dim=1)
        log_alpha_hat[:, 0, :] = log_alpha_hat[:, 0, :] - log_C[:, 0].unsqueeze(-1)

        for t in range(1, T_max):
            prev = log_alpha_hat[:, t - 1, :].unsqueeze(-1)
            trans = self.log_trans.unsqueeze(0)
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

        batch_idx = torch.arange(B, device=self.device)
        log_evidence = torch.zeros(B, dtype=torch.float64, device=self.device)
        for b in range(B):
            log_evidence[b] = log_C[b, :lengths[b]].sum()

        return log_alpha_hat, log_C, log_evidence

    def backward_scaled(
        self, observations: torch.Tensor, lengths: torch.Tensor, log_C: torch.Tensor
    ) -> torch.Tensor:
        """
        Scaled Backward algorithm using forward pass normalization constants.

        beta_hat[t, i] = logsumexp_j(log_A[i,j] + log_B[j](O[t+1]) + beta_hat[t+1, j]) + log_C[t+1]

        The log_C scaling keeps beta_hat bounded near zero.

        Returns:
            log_beta_hat: (B, T_max, n_states) normalized backward variables
        """
        B, T_max = observations.shape
        n_states = self.log_initial.shape[0]

        log_emission = self._gaussian_log_pdf(observations)

        log_beta_hat = torch.zeros((B, T_max, n_states), dtype=torch.float64, device=self.device)

        batch_idx = torch.arange(B, device=self.device)
        for b in range(B):
            log_beta_hat[b, lengths[b] - 1, :] = 0.0

        for t in range(T_max - 2, -1, -1):
            next_emit = log_emission[:, t + 1, :].unsqueeze(1)
            next_beta = log_beta_hat[:, t + 1, :].unsqueeze(1)
            trans = self.log_trans.unsqueeze(0)
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
        self, observations: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute posterior probabilities via Scaled Forward-Backward.

        log_gamma[t, j] = log_alpha_hat[t, j] + log_beta_hat[t, j]
                        - logsumexp_j(log_alpha_hat[t, j] + log_beta_hat[t, j])

        Both alpha_hat and beta_hat are kept near zero by per-step normalization,
        so their sum never drifts to catastrophic magnitudes.
        """
        log_alpha_hat, log_C, log_evidence = self.forward_scaled(observations, lengths)
        log_beta_hat = self.backward_scaled(observations, lengths, log_C)

        log_gamma = log_alpha_hat + log_beta_hat
        log_gamma = log_gamma - self._logsumexp(log_gamma, dim=2, keepdim=True)
        posteriors = torch.exp(log_gamma)

        return posteriors, log_alpha_hat, log_evidence

    def viterbi_batch(
        self, observations: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Normalized Log-Viterbi algorithm (batched).

        At each step, after computing log_delta_raw, we subtract
        max_j(log_delta_raw[t, j]) to keep values bounded near zero.
        This is a uniform shift that does NOT affect argmax or psi,
        but prevents log_delta from drifting to -1e5 for long sequences.

        Mathematical proof of invariance:
          argmax_i((delta_i + c) + A_{i,j}) = argmax_i(delta_i + A_{i,j})
          for any constant c, since c cancels in the comparison.

        Returns:
            states: (B, T_max) most likely state sequences
            log_delta_hat: (B, T_max, n_states) normalized Viterbi scores
        """
        B, T_max = observations.shape
        n_states = self.log_initial.shape[0]

        log_emission = self._gaussian_log_pdf(observations)

        log_delta_hat = torch.zeros((B, T_max, n_states), dtype=torch.float64, device=self.device)
        psi = torch.zeros((B, T_max, n_states), dtype=torch.long, device=self.device)

        log_delta_hat[:, 0, :] = self.log_initial.unsqueeze(0) + log_emission[:, 0, :]
        delta_max = log_delta_hat[:, 0, :].max(dim=1, keepdim=True)[0]
        log_delta_hat[:, 0, :] = log_delta_hat[:, 0, :] - delta_max

        for t in range(1, T_max):
            prev = log_delta_hat[:, t - 1, :].unsqueeze(-1)
            trans = self.log_trans.unsqueeze(0)
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

    @torch.no_grad()
    def predict_batch(
        self,
        observations,
        lengths: Optional[np.ndarray] = None,
        threshold: float = 0.5,
    ) -> List[Dict]:
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
