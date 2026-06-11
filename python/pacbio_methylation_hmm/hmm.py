"""
Unified HMM predictor and helper classes.
"""

import numpy as np
from typing import Optional, List, Dict, Union
import warnings

from .hmm_numpy import MethylationHMM, HMMConfig

try:
    import torch
    from .hmm_torch import MethylationHMMPyTorch, TorchHMMConfig
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class HMMPredictor:
    """
    Unified interface for methylation HMM prediction.

    Automatically selects PyTorch backend if available and requested.
    """

    def __init__(
        self,
        use_torch: bool = True,
        device: str = "cpu",
        config: Optional[dict] = None,
    ):
        self.use_torch = use_torch and _TORCH_AVAILABLE

        if config is None:
            config = {}

        if self.use_torch:
            torch_cfg = TorchHMMConfig(device=device, **config)
            self.model = MethylationHMMPyTorch(torch_cfg)
            if device != "cpu":
                self.model.to_device(device)
        else:
            numpy_cfg = HMMConfig(**{k: v for k, v in config.items() if k in HMMConfig.__dataclass_fields__})
            self.model = MethylationHMM(numpy_cfg)

    def predict(
        self,
        observations: Union[np.ndarray, List[np.ndarray]],
        threshold: float = 0.5,
    ) -> Union[Dict, List[Dict]]:
        """
        Predict methylation states from IPD observations.

        Args:
            observations: Single (T,) array, (B, T) batch array, or list of arrays
            threshold: Methylation probability threshold

        Returns:
            Prediction dict or list of dicts
        """
        if isinstance(observations, list):
            return self._predict_list(observations, threshold)
        elif observations.ndim == 1:
            return self._predict_single(observations, threshold)
        else:
            return self._predict_batch(observations, threshold)

    def _predict_single(self, obs: np.ndarray, threshold: float) -> Dict:
        if self.use_torch:
            results = self.model.predict_batch(obs.reshape(1, -1), threshold=threshold)
            return results[0]
        else:
            return self.model.predict(obs, threshold)

    def _predict_batch(self, obs: np.ndarray, threshold: float) -> List[Dict]:
        if self.use_torch:
            return self.model.predict_batch(obs, threshold=threshold)
        else:
            return self.model.predict(obs, threshold)

    def _predict_list(self, obs_list: List[np.ndarray], threshold: float) -> List[Dict]:
        if self.use_torch:
            return self.model.predict_batch(obs_list, threshold=threshold)
        else:
            return [self.model.predict(o, threshold) for o in obs_list]

    def set_emission_params(
        self,
        unmethylated_mean: float,
        unmethylated_var: float,
        methylated_mean: float,
        methylated_var: float,
    ):
        """
        Set Gaussian emission parameters for the two states.
        """
        means = np.array([unmethylated_mean, methylated_mean])
        vars = np.array([unmethylated_var, methylated_var])

        if self.use_torch:
            import torch
            with torch.no_grad():
                self.model.emission_means.copy_(
                    torch.tensor(means, dtype=torch.float64, device=self.model.device)
                )
                self.model.emission_vars.copy_(
                    torch.tensor(vars, dtype=torch.float64, device=self.model.device)
                )
        else:
            self.model.config.emission_means = means
            self.model.config.emission_vars = vars

    def set_transition_params(
        self,
        stay_unmethylated: float,
        meth_to_unmeth: float,
        unmeth_to_meth: float,
        stay_methylated: float,
    ):
        """
        Set state transition probabilities.
        """
        trans = np.array([
            [stay_unmethylated, unmeth_to_meth],
            [meth_to_unmeth, stay_methylated],
        ])

        if self.use_torch:
            import torch
            with torch.no_grad():
                self.model.log_trans.copy_(
                    torch.log(torch.tensor(trans, dtype=torch.float64, device=self.model.device) + 1e-300)
                )
        else:
            self.model.config.trans_probs = trans
            self.model.config.log_trans = np.log(trans + 1e-300)

    @property
    def is_torch_backend(self) -> bool:
        return self.use_torch
