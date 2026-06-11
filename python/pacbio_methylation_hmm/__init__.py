"""PacBio Methylation HMM - Epigenetics Analysis Engine

A high-performance engine for detecting DNA methylation from PacBio SMRT sequencing data,
combining C++-based BAM parsing, PyTorch-accelerated HMM decoding, and Bi-LSTM neural
emission modeling.
"""

__version__ = "0.2.0"

from .hmm import (
    MethylationHMM,
    MethylationHMMPyTorch,
    HMMPredictor,
)

from .bam_reader import BamReader
from .pipeline import MethylationPipeline

try:
    from .neural_emission import NeuralEmissionNetwork, NeuralEmissionConfig
    from .neural_hmm import NeuralHMM, NeuralHMMConfig
    from .trainer import (
        NeuralHMMTrainer,
        TrainingConfig,
        MethylationDataset,
        generate_synthetic_training_data,
    )
    _NEURAL_AVAILABLE = True
except ImportError:
    _NEURAL_AVAILABLE = False

try:
    from . import _cpp_bindings
    _CPP_AVAILABLE = True
except ImportError:
    _CPP_AVAILABLE = False
    _cpp_bindings = None

__all__ = [
    "MethylationHMM",
    "MethylationHMMPyTorch",
    "HMMPredictor",
    "BamReader",
    "MethylationPipeline",
    "NeuralEmissionNetwork",
    "NeuralHMM",
    "NeuralHMMTrainer",
    "TrainingConfig",
    "MethylationDataset",
    "generate_synthetic_training_data",
    "_CPP_AVAILABLE",
    "_NEURAL_AVAILABLE",
]
