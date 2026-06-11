"""
BAM file reader with C++ acceleration fallback.

Attempts to use the C++ htslib-backed parser; falls back to pysam if available.
"""

import os
import numpy as np
from typing import Optional, List, Dict, Iterator, Union, Tuple
import warnings


class ReadData:
    """Container for per-read methylation signal data."""

    __slots__ = (
        "read_id", "chrom", "ref_start", "ref_end", "read_length",
        "mapq", "is_reverse", "ref_positions", "read_bases",
        "base_qualities", "ipd_values", "pulse_width_values",
    )

    def __init__(
        self,
        read_id: str,
        chrom: str,
        ref_start: int,
        ref_end: int,
        read_length: int,
        mapq: int,
        is_reverse: bool,
        ref_positions: np.ndarray,
        read_bases: np.ndarray,
        base_qualities: np.ndarray,
        ipd_values: np.ndarray,
        pulse_width_values: np.ndarray,
    ):
        self.read_id = read_id
        self.chrom = chrom
        self.ref_start = ref_start
        self.ref_end = ref_end
        self.read_length = read_length
        self.mapq = mapq
        self.is_reverse = is_reverse
        self.ref_positions = ref_positions
        self.read_bases = read_bases
        self.base_qualities = base_qualities
        self.ipd_values = ipd_values
        self.pulse_width_values = pulse_width_values

    @property
    def valid_mask(self) -> np.ndarray:
        """Return mask of positions aligned to reference and with good quality."""
        return (self.ref_positions >= 0) & (self.base_qualities >= 0)

    def get_aligned_ipd(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get (reference_positions, ipd_values) for aligned bases only."""
        mask = self.valid_mask
        return self.ref_positions[mask], self.ipd_values[mask]

    def __repr__(self) -> str:
        return (
            f"ReadData(id={self.read_id[:20]}..., chrom={self.chrom}, "
            f"pos={self.ref_start}-{self.ref_end}, len={self.read_length})"
        )


class BamReader:
    """
    High-performance BAM reader for PacBio methylation analysis.

    Supports both C++ native bindings and pysam fallback.
    """

    def __init__(
        self,
        bam_path: str,
        use_cpp: bool = True,
        min_mapq: int = 0,
        min_baseq: int = 0,
    ):
        if not os.path.exists(bam_path):
            raise FileNotFoundError(f"BAM file not found: {bam_path}")

        self.bam_path = bam_path
        self.min_mapq = min_mapq
        self.min_baseq = min_baseq
        self._use_cpp = use_cpp
        self._backend = None
        self._cpp_parser = None
        self._pysam_af = None

        self._init_backend()

    def _init_backend(self):
        if self._use_cpp:
            try:
                from . import _cpp_bindings
                self._cpp_parser = _cpp_bindings.BamParser(self.bam_path)
                self._cpp_parser.set_min_mapq(self.min_mapq)
                self._cpp_parser.set_min_baseq(self.min_baseq)
                self._backend = "cpp"
                return
            except Exception as e:
                warnings.warn(f"C++ backend failed: {e}. Trying pysam...")

        try:
            import pysam
            self._pysam_af = pysam.AlignmentFile(self.bam_path, "rb")
            self._backend = "pysam"
            return
        except Exception as e:
            raise RuntimeError(
                f"No BAM backend available. Tried C++ bindings and pysam. "
                f"Last error: {e}"
            )

    @property
    def backend(self) -> str:
        return self._backend

    def set_region(self, chrom: str, start: int = 0, end: int = -1):
        if self._backend == "cpp":
            from . import _cpp_bindings
            self._cpp_parser.set_region(_cpp_bindings.Region(chrom, start, end))
        elif self._backend == "pysam":
            self._iter_region = (chrom, start, end)

    def reset(self):
        if self._backend == "cpp":
            self._cpp_parser.reset()
        elif self._backend == "pysam":
            self._pysam_af.reset()

    def _from_cpp_record(self, rec) -> ReadData:
        return ReadData(
            read_id=rec.read_id,
            chrom=rec.chrom,
            ref_start=int(rec.ref_start),
            ref_end=int(rec.ref_end),
            read_length=int(rec.read_length),
            mapq=int(rec.mapq),
            is_reverse=bool(rec.is_reverse),
            ref_positions=np.asarray(rec.ref_positions, dtype=np.int64),
            read_bases=np.asarray(rec.read_bases, dtype=np.uint8),
            base_qualities=np.asarray(rec.base_qualities, dtype=np.uint8),
            ipd_values=np.asarray(rec.ipd_values, dtype=np.float32),
            pulse_width_values=np.asarray(rec.pulse_width_values, dtype=np.float32),
        )

    def _from_pysam_read(self, read) -> Optional[ReadData]:
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            return None
        if read.mapping_quality < self.min_mapq:
            return None

        read_length = read.query_length
        ref_positions = np.full(read_length, -1, dtype=np.int64)

        for query_pos, ref_pos in read.get_aligned_pairs(matches_only=False, with_seq=False):
            if query_pos is not None and ref_pos is not None:
                ref_positions[query_pos] = ref_pos

        read_bases = np.frombuffer(
            read.query_sequence.encode("ascii"), dtype=np.uint8
        ) if read.query_sequence else np.zeros(read_length, dtype=np.uint8)

        base_qualities = np.asarray(
            read.query_qualities, dtype=np.uint8
        ) if read.query_qualities is not None else np.zeros(read_length, dtype=np.uint8)

        ipd_values = None
        pw_values = None

        if read.has_tag("ip"):
            ipd_tag = read.get_tag("ip")
            if isinstance(ipd_tag, (list, tuple)):
                ipd_values = np.asarray(ipd_tag, dtype=np.float32)
            else:
                ipd_values = np.full(read_length, float(ipd_tag), dtype=np.float32)

        if read.has_tag("pw"):
            pw_tag = read.get_tag("pw")
            if isinstance(pw_tag, (list, tuple)):
                pw_values = np.asarray(pw_tag, dtype=np.float32)
            else:
                pw_values = np.full(read_length, float(pw_tag), dtype=np.float32)

        if ipd_values is None:
            ipd_values = np.zeros(read_length, dtype=np.float32)
        if pw_values is None:
            pw_values = np.zeros(read_length, dtype=np.float32)

        if len(ipd_values) != read_length:
            if len(ipd_values) > read_length:
                ipd_values = ipd_values[:read_length]
            else:
                ipd_values = np.pad(ipd_values, (0, read_length - len(ipd_values)))
        if len(pw_values) != read_length:
            if len(pw_values) > read_length:
                pw_values = pw_values[:read_length]
            else:
                pw_values = np.pad(pw_values, (0, read_length - len(pw_values)))

        return ReadData(
            read_id=read.query_name,
            chrom=read.reference_name if read.reference_name else "*",
            ref_start=int(read.reference_start) if read.reference_start is not None else -1,
            ref_end=int(read.reference_end) if read.reference_end is not None else -1,
            read_length=read_length,
            mapq=int(read.mapping_quality),
            is_reverse=read.is_reverse,
            ref_positions=ref_positions,
            read_bases=read_bases,
            base_qualities=base_qualities,
            ipd_values=ipd_values,
            pulse_width_values=pw_values,
        )

    def __iter__(self) -> Iterator[ReadData]:
        if self._backend == "cpp":
            self._cpp_parser.reset()
            while True:
                batch = self._cpp_parser.parse_next_batch(100)
                if not batch:
                    break
                for rec in batch:
                    yield self._from_cpp_record(rec)
        elif self._backend == "pysam":
            if hasattr(self, "_iter_region"):
                iterable = self._pysam_af.fetch(*self._iter_region)
            else:
                iterable = self._pysam_af
            for read in iterable:
                data = self._from_pysam_read(read)
                if data is not None:
                    yield data

    def parse_all(self) -> List[ReadData]:
        return list(self.__iter__())

    def parse_batch(self, batch_size: int = 100) -> List[ReadData]:
        if self._backend == "cpp":
            batch = self._cpp_parser.parse_next_batch(batch_size)
            return [self._from_cpp_record(r) for r in batch]
        elif self._backend == "pysam":
            results = []
            for _ in range(batch_size):
                try:
                    if hasattr(self, "_iter_region"):
                        read = next(self._pysam_af.fetch(*self._iter_region))
                    else:
                        read = next(self._pysam_af)
                except StopIteration:
                    break
                data = self._from_pysam_read(read)
                if data is not None:
                    results.append(data)
            return results

    def extract_ipd_sequences(
        self,
        target_base: Optional[bytes] = None,
        min_quality: int = 10,
    ) -> List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Extract IPD sequences for HMM input.

        Args:
            target_base: If set (e.g., b'C'), only return positions matching this base.
            min_quality: Minimum base quality.

        Returns:
            List of (read_id, ref_positions, ipd_values, mask) tuples
        """
        results = []
        for read in self:
            mask = (
                (read.ref_positions >= 0)
                & (read.base_qualities >= min_quality)
            )
            if target_base is not None:
                mask = mask & (read.read_bases == target_base[0])

            if np.any(mask):
                results.append((
                    read.read_id,
                    read.ref_positions[mask],
                    read.ipd_values[mask],
                    mask,
                ))

        return results

    def close(self):
        if self._backend == "pysam" and self._pysam_af is not None:
            self._pysam_af.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
