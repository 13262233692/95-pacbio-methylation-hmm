"""
Stress test: Ultra-long HiFi reads (50000 bp) numerical stability.

Verifies that the Scaled Forward-Backward and Normalized Log-Viterbi
algorithms remain numerically stable for sequences of arbitrary length.

Before the fix, log_alpha/log_delta would drift to -1e5 magnitudes,
causing catastrophic loss of floating-point precision. After the fix,
per-step normalization keeps all intermediate values bounded near zero.
"""

import sys
import time
import numpy as np

sys.path.insert(0, 'python')

from pacbio_methylation_hmm.hmm_numpy import MethylationHMM, HMMConfig


def generate_long_read(T, seed=42):
    np.random.seed(seed)
    true_states = np.zeros(T, dtype=int)
    for t in range(1, T):
        if true_states[t - 1] == 0:
            true_states[t] = 0 if np.random.rand() < 0.95 else 1
        else:
            true_states[t] = 1 if np.random.rand() < 0.90 else 0

    obs = np.where(
        true_states == 0,
        np.random.normal(0.0, np.sqrt(0.5), T),
        np.random.normal(1.5, np.sqrt(1.0), T),
    )
    return obs, true_states


def check_numerical_health(result, label):
    issues = []
    post = result['posteriors']
    if np.any(np.isnan(post)):
        issues.append(f"[{label}] NaN in posteriors!")
    if np.any(post < 0.0) or np.any(post > 1.0 + 1e-10):
        issues.append(f"[{label}] Posteriors out of [0,1] range!")
    if np.any(np.isnan(result['viterbi_states'])):
        issues.append(f"[{label}] NaN in Viterbi states!")

    state_set = set(result['viterbi_states'])
    if len(state_set) == 1:
        issues.append(f"[{label}] Viterbi states ALL IDENTICAL: {list(state_set)} — underflow collapse!")

    prob = result['methylation_prob']
    if np.all(prob < 0.01) or np.all(prob > 0.99):
        issues.append(f"[{label}] Methylation probabilities are degenerate (all near 0 or 1)")

    return issues


def run_numpy_test(T, label=""):
    print(f"\n{'='*60}", flush=True)
    print(f"  NumPy HMM Test: T = {T:,} bp  {label}", flush=True)
    print(f"{'='*60}", flush=True)

    obs, true_states = generate_long_read(T)
    hmm = MethylationHMM()

    t0 = time.time()
    result = hmm.predict(obs, threshold=0.5)
    elapsed = time.time() - t0

    viterbi_acc = np.mean(result['viterbi_states'] == true_states)
    posterior_acc = np.mean(result['states'] == true_states)
    true_meth = true_states.sum()
    called_meth = result['viterbi_states'].sum()

    print(f"  Time: {elapsed:.2f}s", flush=True)
    print(f"  Viterbi accuracy:    {viterbi_acc:.4f}", flush=True)
    print(f"  Posterior accuracy:  {posterior_acc:.4f}", flush=True)
    print(f"  True methylated:     {true_meth:,}", flush=True)
    print(f"  Called methylated:   {called_meth:,}", flush=True)
    print(f"  Log evidence:        {result['log_evidence']:.2f}", flush=True)

    post = result['posteriors']
    print(f"  Posterior range:     [{post.min():.6f}, {post.max():.6f}]", flush=True)
    print(f"  Posterior sum(0):    mean={post[:, 0].mean():.6f}", flush=True)
    print(f"  Posterior sum(1):    mean={post[:, 1].mean():.6f}", flush=True)

    issues = check_numerical_health(result, f"NumPy T={T}")
    if issues:
        print(f"\n  *** NUMERICAL ISSUES DETECTED ***", flush=True)
        for issue in issues:
            print(f"  {issue}", flush=True)
        return False
    else:
        print(f"\n  PASSED — No numerical issues detected", flush=True)
        return True


def run_torch_stress_test(T_values):
    try:
        import torch
        from pacbio_methylation_hmm.hmm_torch import MethylationHMMPyTorch, TorchHMMConfig
        from pacbio_methylation_hmm.hmm import HMMPredictor
    except ImportError:
        print("\nPyTorch not available, skipping torch stress test", flush=True)
        return True

    print(f"\n{'='*60}", flush=True)
    print(f"  PyTorch HMM Stress Test: Ultra-long reads", flush=True)
    print(f"{'='*60}", flush=True)

    predictor = HMMPredictor(use_torch=True, device="cpu")
    all_passed = True

    for T in T_values:
        obs, true_states = generate_long_read(T)

        t0 = time.time()
        result = predictor.predict(obs, threshold=0.5)
        elapsed = time.time() - t0

        viterbi_acc = np.mean(result['viterbi_states'] == true_states)
        true_meth = true_states.sum()
        called_meth = result['viterbi_states'].sum()

        print(f"\n  T={T:>7,}bp | {elapsed:.2f}s | Viterbi acc={viterbi_acc:.4f} | "
              f"True meth={true_meth:,} | Called={called_meth:,}", flush=True)

        issues = check_numerical_health(result, f"PyTorch T={T}")
        if issues:
            for issue in issues:
                print(f"    {issue}", flush=True)
            all_passed = False
        else:
            print(f"    PASSED", flush=True)

    return all_passed


def main():
    print("=" * 60, flush=True)
    print("  ULTRA-LONG READ NUMERICAL STABILITY STRESS TEST", flush=True)
    print("  Testing Scaled Forward-Backward + Normalized Log-Viterbi", flush=True)
    print("=" * 60, flush=True)

    numpy_lengths = [200, 5000, 10000, 25000, 50000]
    torch_lengths = [5000, 10000, 25000, 50000]

    all_passed = True

    print("\n" + "=" * 60, flush=True)
    print("  PART 1: NumPy single-read stress test", flush=True)
    print("=" * 60, flush=True)
    for T in numpy_lengths:
        if T > 25000:
            label = "(HiFi-scale ultra-long)"
        elif T > 10000:
            label = "(HiFi-scale long)"
        else:
            label = ""
        ok = run_numpy_test(T, label)
        all_passed = all_passed and ok

    print("\n" + "=" * 60, flush=True)
    print("  PART 2: PyTorch batch stress test", flush=True)
    print("=" * 60, flush=True)
    ok = run_torch_stress_test(torch_lengths)
    all_passed = all_passed and ok

    print("\n" + "=" * 60, flush=True)
    if all_passed:
        print("  ALL STRESS TESTS PASSED", flush=True)
    else:
        print("  SOME TESTS FAILED — NUMERICAL ISSUES DETECTED", flush=True)
    print("=" * 60, flush=True)

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
