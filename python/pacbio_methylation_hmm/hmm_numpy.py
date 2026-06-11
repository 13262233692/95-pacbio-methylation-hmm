"""
Core HMM implementation for methylation detection (NumPy version).

Numerically stable implementation using:
  - Scaled Forward-Backward with per-step log-normalization
  - Normalized Log-Viterbi with per-step max-subtraction
  - Logsumexp with finite-guaranteed arithmetic

States:
    - 0: Unmethylated (U)
    - 1: Methylated (M)

Observations: Continuous IPD values modeled as Gaussian distributions.
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass, field


@dataclass
class HMMConfig:
    n_states: int = 2

    initial_probs: np.ndarray = field(default_factory=lambda: np.array([0.8, 0.2]))

    trans_probs: np.ndarray = field(
        default_factory=lambda: np.array([
            [0.95, 0.05],
            [0.10, 0.90],
        ]))

    emission_means: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.5]))

    emission_vars: np.ndarray = field(default_factory=lambda: np.array([0.5, 1.0]))

    emission_log_var_clip: float = -50.0

    def __post_init__(self):
        self.initial_probs = np.asarray(self.initial_probs, dtype=np.float64)
        self.trans_probs = np.asarray(self.trans_probs, dtype=np.float64)
        self.emission_means = np.asarray(self.emission_means, dtype=np.float64)
        self.emission_vars = np.asarray(self.emission_vars, dtype=np.float64)
        self.log_initial = np.log(self.initial_probs + 1e-300)
        self.log_trans = np.log(self.trans_probs + 1e-300)


class MethylationHMM:
    """
    Hidden Markov Model for methylation state decoding from PacBio IPD signals.

    All core algorithms operate entirely in log-probability space:
      - Forward-Backward: per-step logsumexp normalization prevents drift
      - Viterbi: per-step max-subtraction keeps delta bounded near zero
    """

    STATE_UNMETHYLATED = 0
    STATE_METHYLATED = 1

    def __init__(self, config: Optional[HMMConfig] = None):
        self.config = config if config is not None else HMMConfig()
        self._validate_config()

    def _validate_config(self):
        c = self.config
        assert c.n_states == 2
        assert c.initial_probs.shape == (c.n_states,)
        assert c.trans_probs.shape == (c.n_states, c.n_states)
        assert c.emission_means.shape == (c.n_states,)
        assert c.emission_vars.shape == (c.n_states,)

    def _gaussian_log_pdf(self, observations: np.ndarray) -> np.ndarray:
        T = observations.shape[0]
        log_probs = np.zeros((T, self.config.n_states), dtype=np.float64)
        for s in range(self.config.n_states):
            mean = self.config.emission_means[s]
            var = self.config.emission_vars[s]
            log_probs[:, s] = -0.5 * (
                np.log(2.0 * np.pi * var)
                + (observations - mean) ** 2 / var
            )
        clip = self.config.emission_log_var_clip
        np.clip(log_probs, clip, 0.0, out=log_probs)
        return log_probs

    @staticmethod
    def _logsumexp(a: np.ndarray, axis: Optional[int] = None, keepdims: bool = False) -> np.ndarray:
        a_max = np.max(a, axis=axis, keepdims=True)
        a_max_safe = np.where(np.isfinite(a_max), a_max, 0.0)
        sumexp = np.sum(np.exp(a - a_max_safe), axis=axis, keepdims=True)
        sumexp = np.maximum(sumexp, 1e-300)
        out = np.log(sumexp)
        result = a_max_safe + out
        if not keepdims:
            result = np.squeeze(result, axis=axis)
        return result

    def forward_scaled(
        self, observations: np.ndarray, log_emission: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Scaled Forward algorithm with per-step log-normalization.

        At each step t, after computing raw log_alpha, we subtract
        logsumexp_j(log_alpha_raw[t,j]) so that the normalized values
        sum to 1 in probability space. The scaling constants log_C are
        accumulated to recover log P(O).

        This prevents log_alpha from drifting to -1e5 for long sequences.

        Returns:
            log_alpha_hat: (T, n_states) normalized forward variables
            log_C: (T,) per-step log normalization constants
            log_evidence: log P(O) = sum(log_C)
        """
        T = observations.shape[0]
        S = self.config.n_states
        if log_emission is None:
            log_emission = self._gaussian_log_pdf(observations)

        log_alpha_hat = np.zeros((T, S), dtype=np.float64)
        log_C = np.zeros(T, dtype=np.float64)

        log_alpha_hat[0] = self.config.log_initial + log_emission[0]
        log_C[0] = self._logsumexp(log_alpha_hat[0])
        log_alpha_hat[0] -= log_C[0]

        for t in range(1, T):
            for j in range(S):
                log_alpha_hat[t, j] = log_emission[t, j] + self._logsumexp(
                    log_alpha_hat[t - 1] + self.config.log_trans[:, j]
                )
            log_C[t] = self._logsumexp(log_alpha_hat[t])
            log_alpha_hat[t] -= log_C[t]

        log_evidence = float(np.sum(log_C))
        return log_alpha_hat, log_C, log_evidence

    def backward_scaled(
        self, observations: np.ndarray, log_C: np.ndarray,
        log_emission: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Scaled Backward algorithm using the same per-step normalization
        constants log_C from the forward pass.

        beta_hat[t, i] = logsumexp_j(log_A[i,j] + log_B[j](O[t+1]) + beta_hat[t+1, j]) + log_C[t+1]

        The log_C[t+1] scaling keeps beta_hat bounded just like alpha_hat.

        Returns:
            log_beta_hat: (T, n_states) normalized backward variables
        """
        T = observations.shape[0]
        S = self.config.n_states
        if log_emission is None:
            log_emission = self._gaussian_log_pdf(observations)

        log_beta_hat = np.zeros((T, S), dtype=np.float64)
        log_beta_hat[-1] = 0.0

        for t in range(T - 2, -1, -1):
            for i in range(S):
                log_beta_hat[t, i] = self._logsumexp(
                    self.config.log_trans[i, :]
                    + log_emission[t + 1, :]
                    + log_beta_hat[t + 1, :]
                )
            log_beta_hat[t] += log_C[t + 1]

        return log_beta_hat

    def compute_posterior(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute posterior state probabilities using Scaled Forward-Backward.

        gamma[t, j] = alpha_hat[t, j] * beta_hat[t, j]
                    / sum_j(alpha_hat[t, j] * beta_hat[t, j])

        In log space:
        log_gamma[t, j] = log_alpha_hat[t, j] + log_beta_hat[t, j]
                        - logsumexp_j(log_alpha_hat[t, j] + log_beta_hat[t, j])

        Because alpha_hat and beta_hat are kept near zero by per-step
        normalization, their sum never drifts to catastrophic magnitudes.
        """
        log_emission = self._gaussian_log_pdf(observations)
        log_alpha_hat, log_C, log_evidence = self.forward_scaled(observations, log_emission)
        log_beta_hat = self.backward_scaled(observations, log_C, log_emission)

        log_gamma = log_alpha_hat + log_beta_hat
        log_gamma = log_gamma - self._logsumexp(log_gamma, axis=1, keepdims=True)
        posteriors = np.exp(log_gamma)

        return posteriors, log_alpha_hat, log_evidence

    def viterbi(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalized Log-Viterbi algorithm.

        At each step, after computing log_delta_raw, we subtract
        max_j(log_delta_raw[t,j]) to keep values near zero.
        This does NOT affect argmax or psi because it's a uniform shift,
        but it prevents log_delta from drifting to -1e5 for long sequences.

        The backtracking pointer psi is completely unaffected by normalization
        because argmax_i(a_i + c) = argmax_i(a_i) for any constant c.

        Returns:
            states: (T,) most likely state sequence
            log_delta_hat: (T, n_states) normalized Viterbi scores
        """
        T = observations.shape[0]
        S = self.config.n_states
        log_emission = self._gaussian_log_pdf(observations)

        log_delta_hat = np.zeros((T, S), dtype=np.float64)
        psi = np.zeros((T, S), dtype=np.int32)

        log_delta_hat[0] = self.config.log_initial + log_emission[0]
        log_delta_hat[0] -= np.max(log_delta_hat[0])

        for t in range(1, T):
            for j in range(S):
                scores = log_delta_hat[t - 1] + self.config.log_trans[:, j]
                psi[t, j] = int(np.argmax(scores))
                log_delta_hat[t, j] = scores[psi[t, j]] + log_emission[t, j]

            log_delta_hat[t] -= np.max(log_delta_hat[t])

        states = np.zeros(T, dtype=np.int32)
        states[-1] = int(np.argmax(log_delta_hat[-1]))

        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states, log_delta_hat

    def predict(self, observations: np.ndarray, threshold: float = 0.5) -> dict:
        if observations.ndim == 1:
            return self._predict_single(observations, threshold)
        else:
            return [self._predict_single(observations[i], threshold) for i in range(observations.shape[0])]

    def _predict_single(self, observations: np.ndarray, threshold: float) -> dict:
        posteriors, log_alpha_hat, log_evidence = self.compute_posterior(observations)
        states = (posteriors[:, self.STATE_METHYLATED] >= threshold).astype(np.int32)
        viterbi_states, _ = self.viterbi(observations)

        return {
            "states": states,
            "viterbi_states": viterbi_states,
            "posteriors": posteriors,
            "methylation_prob": posteriors[:, self.STATE_METHYLATED],
            "log_evidence": log_evidence,
        }
