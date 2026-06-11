"""PacBio Methylation HMM - Epigenetics Analysis Engine

A high-performance engine for detecting DNA methylation from PacBio SMRT sequencing data,
combining C++-based BAM parsing and PyTorch-accelerated HMM decoding.
"""

__version__ = "0.1.0"

from .hmm import (
    MethylationHMM,
    MethylationHMMPyTorch,
    HMMPredictor,
)

from .bam_reader import BamReader
from .pipeline import MethylationPipeline

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
    "_CPP_AVAILABLE",
]
