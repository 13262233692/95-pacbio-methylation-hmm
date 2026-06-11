#!/usr/bin/env python
"""
Example: Analyze methylation from a real PacBio BAM file.

Usage:
    python analyze_bam.py input.bam --output calls.tsv --region chr1:1000-2000
"""

import sys
import argparse
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from pacbio_methylation_hmm.pipeline import MethylationPipeline, PipelineConfig


def parse_region(region_str: str):
    """Parse 'chr1:1000-2000' format."""
    if not region_str:
        return None
    try:
        chrom, coords = region_str.split(':')
        start, end = coords.split('-')
        return (chrom, int(start), int(end))
    except Exception:
        print(f"Warning: Could not parse region '{region_str}', ignoring")
        return None


def main():
    parser = argparse.ArgumentParser(description='Detect methylation from PacBio BAM')
    parser.add_argument('bam', help='Input BAM file (with .bai index)')
    parser.add_argument('--output', '-o', default='methylation_calls.tsv', help='Output TSV file')
    parser.add_argument('--region', '-r', help='Region in format chr:start-end')
    parser.add_argument('--max-reads', type=int, default=None, help='Max reads to process (for testing)')
    parser.add_argument('--device', default='cpu', help='Device: cpu or cuda')
    parser.add_argument('--min-mapq', type=int, default=10, help='Minimum mapping quality')
    parser.add_argument('--threshold', type=float, default=0.5, help='Methylation probability threshold')
    args = parser.parse_args()

    if not os.path.exists(args.bam):
        print(f"Error: BAM file not found: {args.bam}")
        sys.exit(1)

    config = PipelineConfig(
        min_mapq=args.min_mapq,
        hmm_threshold=args.threshold,
        use_torch=True,
        device=args.device,
    )

    pipeline = MethylationPipeline(config)

    region = parse_region(args.region)
    region_str = f" in region {args.region}" if region else ""
    print(f"Processing {args.bam}{region_str}...")

    calls = pipeline.run_bam(args.bam, region=region, max_reads=args.max_reads)

    print(f"Found {len(calls)} methylated positions")

    with open(args.output, 'w') as f:
        f.write("chrom\tposition\tstrand\tcoverage\tmethylated\tunmethylated\t"
                "methylation_level\tmean_ipd\tmean_meth_prob\n")
        for c in calls:
            f.write(f"{c.chrom}\t{c.position}\t{c.strand}\t{c.read_count}\t"
                    f"{c.methylated_count}\t{c.unmethylated_count}\t"
                    f"{c.methylation_level:.4f}\t{c.mean_ipd:.4f}\t"
                    f"{c.mean_meth_prob:.4f}\n")

    print(f"Results written to {args.output}")

    total_cov = sum(c.read_count for c in calls)
    avg_level = sum(c.methylation_level for c in calls) / max(len(calls), 1)
    print(f"Total coverage: {total_cov}, Average methylation level: {avg_level:.4f}")


if __name__ == '__main__':
    main()
