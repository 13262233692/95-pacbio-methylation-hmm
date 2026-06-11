"""
End-to-end methylation analysis pipeline.

Combines BAM parsing and HMM decoding into a single, easy-to-use API.
"""

import os
import numpy as np
from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict

from .bam_reader import BamReader, ReadData
from .hmm import HMMPredictor


@dataclass
class MethylationCall:
    chrom: str
    position: int
    strand: str
    read_count: int
    methylated_count: int
    unmethylated_count: int
    mean_ipd: float
    mean_meth_prob: float
    methylation_level: float

    @property
    def coverage(self) -> int:
        return self.read_count


@dataclass
class PipelineConfig:
    min_mapq: int = 10
    min_baseq: int = 10
    hmm_threshold: float = 0.5
    use_torch: bool = True
    device: str = "cpu"
    batch_size: int = 32
    target_base: Optional[bytes] = b'C'
    normalize_ipd: bool = True
    ipd_mean: float = 0.5
    ipd_std: float = 0.8


class MethylationPipeline:
    """
    Complete PacBio methylation analysis pipeline.

    Usage:
        >>> pipeline = MethylationPipeline()
        >>> calls = pipeline.run_bam("sample.bam")
        >>> for c in calls[:10]:
        ...     print(c.chrom, c.position, c.methylation_level)
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config if config is not None else PipelineConfig()
        self.predictor = HMMPredictor(
            use_torch=self.config.use_torch,
            device=self.config.device,
        )

    def _normalize_ipd(self, ipd: np.ndarray) -> np.ndarray:
        if not self.config.normalize_ipd:
            return ipd
        return (ipd - self.config.ipd_mean) / (self.config.ipd_std + 1e-8)

    def _process_reads(
        self, reads: List[ReadData]
    ) -> List[Dict]:
        if not reads:
            return []

        ipd_sequences = []
        for r in reads:
            mask = (r.ref_positions >= 0) & (r.base_qualities >= self.config.min_baseq)
            if self.config.target_base is not None:
                mask = mask & (r.read_bases == self.config.target_base[0])
            if np.any(mask):
                ipd_seq = self._normalize_ipd(r.ipd_values[mask].astype(np.float64))
                if len(ipd_seq) > 0:
                    ipd_sequences.append(ipd_seq)

        if not ipd_sequences:
            return []

        return self.predictor.predict(ipd_sequences, threshold=self.config.hmm_threshold)

    def run_bam(
        self,
        bam_path: str,
        region: Optional[Tuple[str, int, int]] = None,
        max_reads: Optional[int] = None,
    ) -> List[MethylationCall]:
        """
        Run the full pipeline on a BAM file.

        Args:
            bam_path: Path to the BAM file.
            region: Optional (chrom, start, end) region to process.
            max_reads: Maximum number of reads to process (for testing).

        Returns:
            List of per-site MethylationCall objects.
        """
        all_predictions = []
        all_read_data = []

        with BamReader(bam_path, min_mapq=self.config.min_mapq) as reader:
            if region:
                reader.set_region(*region)

            processed = 0
            while True:
                batch = reader.parse_batch(self.config.batch_size * 10)
                if not batch:
                    break

                preds = self._process_reads(batch)
                all_predictions.extend(preds)
                all_read_data.extend(batch)

                processed += len(batch)
                if max_reads is not None and processed >= max_reads:
                    break

        return self._aggregate_calls(all_read_data, all_predictions)

    def run_bam_stream(
        self,
        bam_path: str,
        region: Optional[Tuple[str, int, int]] = None,
    ) -> Dict[Tuple[str, int, str], Dict]:
        """
        Stream and accumulate per-position methylation statistics.

        Returns a dict keyed by (chrom, position, strand).
        """
        site_stats = defaultdict(lambda: {
            "meth_count": 0,
            "unmeth_count": 0,
            "ipd_sum": 0.0,
            "prob_sum": 0.0,
        })

        with BamReader(bam_path, min_mapq=self.config.min_mapq) as reader:
            if region:
                reader.set_region(*region)

            while True:
                batch = reader.parse_batch(self.config.batch_size * 10)
                if not batch:
                    break

                preds = self._process_reads(batch)

                pred_idx = 0
                for read in batch:
                    mask = (
                        (read.ref_positions >= 0)
                        & (read.base_qualities >= self.config.min_baseq)
                    )
                    if self.config.target_base is not None:
                        mask = mask & (read.read_bases == self.config.target_base[0])
                    if not np.any(mask):
                        continue

                    if pred_idx >= len(preds):
                        break

                    pred = preds[pred_idx]
                    pred_idx += 1

                    positions = read.ref_positions[mask]
                    ipds = read.ipd_values[mask]
                    probs = pred["methylation_prob"]
                    states = pred["states"]

                    strand = "-" if read.is_reverse else "+"

                    n = min(len(positions), len(states))
                    for i in range(n):
                        pos = int(positions[i])
                        key = (read.chrom, pos, strand)
                        stats = site_stats[key]
                        stats["ipd_sum"] += float(ipds[i]) if i < len(ipds) else 0.0
                        stats["prob_sum"] += float(probs[i]) if i < len(probs) else 0.0
                        if states[i] == 1:
                            stats["meth_count"] += 1
                        else:
                            stats["unmeth_count"] += 1

        return dict(site_stats)

    def _aggregate_calls(
        self,
        reads: List[ReadData],
        predictions: List[Dict],
    ) -> List[MethylationCall]:
        site_data = defaultdict(lambda: {
            "meth": 0,
            "unmeth": 0,
            "ipds": [],
            "probs": [],
            "strand": "+",
            "chrom": "",
        })

        pred_idx = 0
        for read in reads:
            mask = (
                (read.ref_positions >= 0)
                & (read.base_qualities >= self.config.min_baseq)
            )
            if self.config.target_base is not None:
                mask = mask & (read.read_bases == self.config.target_base[0])
            if not np.any(mask):
                continue

            if pred_idx >= len(predictions):
                break

            pred = predictions[pred_idx]
            pred_idx += 1

            positions = read.ref_positions[mask]
            ipds = read.ipd_values[mask]
            probs = pred["methylation_prob"]
            states = pred["states"]
            strand = "-" if read.is_reverse else "+"

            n = min(len(positions), len(states))
            for i in range(n):
                key = (read.chrom, int(positions[i]), strand)
                d = site_data[key]
                d["chrom"] = read.chrom
                d["strand"] = strand
                if states[i] == 1:
                    d["meth"] += 1
                else:
                    d["unmeth"] += 1
                if i < len(ipds):
                    d["ipds"].append(float(ipds[i]))
                if i < len(probs):
                    d["probs"].append(float(probs[i]))

        calls = []
        for (chrom, pos, strand), d in site_data.items():
            total = d["meth"] + d["unmeth"]
            if total == 0:
                continue
            calls.append(MethylationCall(
                chrom=chrom,
                position=pos,
                strand=strand,
                read_count=total,
                methylated_count=d["meth"],
                unmethylated_count=d["unmeth"],
                mean_ipd=float(np.mean(d["ipds"])) if d["ipds"] else 0.0,
                mean_meth_prob=float(np.mean(d["probs"])) if d["probs"] else 0.0,
                methylation_level=d["meth"] / total,
            ))

        calls.sort(key=lambda c: (c.chrom, c.position, c.strand))
        return calls

    def set_hmm_params(
        self,
        unmethylated_mean: float = 0.0,
        unmethylated_var: float = 0.5,
        methylated_mean: float = 1.5,
        methylated_var: float = 1.0,
        stay_unmethylated: float = 0.95,
        stay_methylated: float = 0.90,
    ):
        """
        Configure the HMM emission and transition parameters.
        """
        self.predictor.set_emission_params(
            unmethylated_mean, unmethylated_var,
            methylated_mean, methylated_var,
        )
        self.predictor.set_transition_params(
            stay_unmethylated=stay_unmethylated,
            meth_to_unmeth=1.0 - stay_methylated,
            unmeth_to_meth=1.0 - stay_unmethylated,
            stay_methylated=stay_methylated,
        )
