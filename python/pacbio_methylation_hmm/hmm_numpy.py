"""
Core HMM implementation for methylation detection (NumPy version).

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

    use_log_space: bool = True

    def __post_init__(self):
        self.initial_probs = np.asarray(self.initial_probs, dtype=np.float64)
        self.trans_probs = np.asarray(self.trans_probs, dtype=np.float64)
        self.emission_means = np.asarray(self.emission_means, dtype=np.float64)
        self.emission_vars = np.asarray(self.emission_vars, dtype=np.float64)

        if self.use_log_space:
            self.log_initial = np.log(self.initial_probs + 1e-300)
            self.log_trans = np.log(self.trans_probs + 1e-300)


class MethylationHMM:
    """
    Hidden Markov Model for methylation state decoding from PacBio IPD signals.
    """

    STATE_UNMETHYLATED = 0
    STATE_METHYLATED = 1

    def __init__(self, config: Optional[HMMConfig] = None):
        self.config = config if config is not None else HMMConfig()
        self._validate_config()

    def _validate_config(self):
        c = self.config
        assert c.n_states == 2, "Currently only 2-state HMM is supported"
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
                np.log(2 * np.pi * var)
                + (observations - mean) ** 2 / var
            )
        return log_probs

    @staticmethod
    def _logsumexp(a: np.ndarray, axis: Optional[int] = None, keepdims: bool = False) -> np.ndarray:
        a_max = np.max(a, axis=axis, keepdims=True)
        a_max_safe = np.where(np.isfinite(a_max), a_max, 0.0)
        out = np.log(np.sum(np.exp(a - a_max_safe), axis=axis, keepdims=True))
        result = a_max_safe + out
        if not keepdims:
            result = np.squeeze(result, axis=axis)
        return result

    def forward(self, observations: np.ndarray, log_emission: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        T = observations.shape[0]
        if log_emission is None:
            log_emission = self._gaussian_log_pdf(observations)

        log_alpha = np.full((T, self.config.n_states), -np.inf, dtype=np.float64)
        log_alpha[0] = self.config.log_initial + log_emission[0]

        for t in range(1, T):
            for j in range(self.config.n_states):
                log_alpha[t, j] = log_emission[t, j] + self._logsumexp(
                    log_alpha[t - 1] + self.config.log_trans[:, j]
                )

        log_evidence = float(self._logsumexp(log_alpha[-1]))
        return log_alpha, log_evidence

    def backward(self, observations: np.ndarray, log_emission: Optional[np.ndarray] = None) -> np.ndarray:
        T = observations.shape[0]
        if log_emission is None:
            log_emission = self._gaussian_log_pdf(observations)

        log_beta = np.full((T, self.config.n_states), -np.inf, dtype=np.float64)
        log_beta[-1] = 0.0

        for t in range(T - 2, -1, -1):
            for i in range(self.config.n_states):
                log_beta[t, i] = self._logsumexp(
                    self.config.log_trans[i, :]
                    + log_emission[t + 1, :]
                    + log_beta[t + 1, :]
                )

        return log_beta

    def compute_posterior(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        log_emission = self._gaussian_log_pdf(observations)
        log_alpha, log_evidence = self.forward(observations, log_emission)
        log_beta = self.backward(observations, log_emission)

        log_posterior = log_alpha + log_beta - log_evidence
        log_posterior = log_posterior - self._logsumexp(log_posterior, axis=1, keepdims=True)
        posteriors = np.exp(log_posterior)
        return posteriors, log_alpha, log_evidence

    def viterbi(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        T = observations.shape[0]
        log_emission = self._gaussian_log_pdf(observations)

        log_delta = np.full((T, self.config.n_states), -np.inf, dtype=np.float64)
        psi = np.zeros((T, self.config.n_states), dtype=np.int32)

        log_delta[0] = self.config.log_initial + log_emission[0]

        for t in range(1, T):
            for j in range(self.config.n_states):
                scores = log_delta[t - 1] + self.config.log_trans[:, j]
                psi[t, j] = int(np.argmax(scores))
                log_delta[t, j] = scores[psi[t, j]] + log_emission[t, j]

        states = np.zeros(T, dtype=np.int32)
        states[-1] = int(np.argmax(log_delta[-1]))

        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states, log_delta

    def predict(self, observations: np.ndarray, threshold: float = 0.5) -> dict:
        if observations.ndim == 1:
            return self._predict_single(observations, threshold)
        else:
            return [self._predict_single(observations[i], threshold) for i in range(observations.shape[0])]

    def _predict_single(self, observations: np.ndarray, threshold: float) -> dict:
        posteriors, log_alpha, log_evidence = self.compute_posterior(observations)
        states = (posteriors[:, self.STATE_METHYLATED] >= threshold).astype(np.int32)
        viterbi_states, _ = self.viterbi(observations)

        return {
            "states": states,
            "viterbi_states": viterbi_states,
            "posteriors": posteriors,
            "methylation_prob": posteriors[:, self.STATE_METHYLATED],
            "log_evidence": log_evidence,
        }
