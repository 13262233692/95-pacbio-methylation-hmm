#!/usr/bin/env python
"""
Example: End-to-end methylation detection pipeline.

This script demonstrates the complete workflow:
1. Generate synthetic PacBio-like IPD data with known methylation states
2. Run the HMM decoder
3. Output methylation calls
"""

import sys
import numpy as np

sys.path.insert(0, 'python')

from pacbio_methylation_hmm.hmm import HMMPredictor
from pacbio_methylation_hmm.pipeline import MethylationPipeline, PipelineConfig


def generate_synthetic_reads(n_reads=50, read_length=500):
    """
    Generate synthetic PacBio reads with methylation marks.

    Creates reads where certain CpG-like regions have elevated IPD values
    representing 5mC methylation.
    """
    reads = []
    for i in range(n_reads):
        states = np.zeros(read_length, dtype=int)
        # Simulate a methylated region in the middle
        meth_start = read_length // 4
        meth_end = read_length // 2
        for t in range(1, read_length):
            if meth_start <= t < meth_end:
                states[t] = 1 if np.random.rand() < 0.90 else 0
            else:
                states[t] = 1 if np.random.rand() < 0.05 else 0

        obs = np.where(states == 0,
                       np.random.normal(0.0, np.sqrt(0.5), read_length),
                       np.random.normal(1.5, np.sqrt(1.0), read_length))

        reads.append({
            'id': f'synth_read_{i}',
            'observations': obs,
            'true_states': states,
        })
    return reads


def main():
    print("=" * 60)
    print("PacBio Methylation HMM - Synthetic Data Demo")
    print("=" * 60)

    reads = generate_synthetic_reads(n_reads=20, read_length=300)
    print(f"\nGenerated {len(reads)} synthetic reads")

    predictor = HMMPredictor(use_torch=True, device="cpu")

    obs_list = [r['observations'] for r in reads]
    print("Running HMM decoding (PyTorch backend)...")
    results = predictor.predict(obs_list, threshold=0.5)

    print("\n" + "-" * 60)
    print("Per-read methylation detection results:")
    print("-" * 60)

    for i, (read, res) in enumerate(zip(reads, results)):
        true_meth = read['true_states'].sum()
        called_meth = res['states'].sum()
        acc = np.mean(res['viterbi_states'] == read['true_states'])
        print(
            f"  {read['id']:20s} | "
            f"True meth: {true_meth:4d} | "
            f"Called: {called_meth:4d} | "
            f"Accuracy: {acc:.4f} | "
            f"LogEvidence: {res['log_evidence']:8.2f}"
        )

    avg_acc = np.mean([
        np.mean(r['viterbi_states'] == reads[i]['true_states'])
        for i, r in enumerate(results)
    ])
    print("-" * 60)
    print(f"Average Viterbi accuracy across all reads: {avg_acc:.4f}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
