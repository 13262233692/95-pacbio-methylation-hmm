import numpy as np
import sys

sys.path.insert(0, 'python')

try:
    import torch
    from pacbio_methylation_hmm.hmm_torch import MethylationHMMPyTorch, TorchHMMConfig
    from pacbio_methylation_hmm.hmm import HMMPredictor
    print("PyTorch available", flush=True)
except ImportError as e:
    print(f"PyTorch not available: {e}", flush=True)
    sys.exit(0)

np.random.seed(42)
torch.manual_seed(42)

B = 8
T = 200
true_states_all = np.zeros((B, T), dtype=int)
obs_all = np.zeros((B, T), dtype=np.float64)

for b in range(B):
    true_states = np.zeros(T, dtype=int)
    for t in range(1, T):
        if true_states[t-1] == 0:
            true_states[t] = 0 if np.random.rand() < 0.95 else 1
        else:
            true_states[t] = 1 if np.random.rand() < 0.90 else 0
    true_states_all[b] = true_states
    obs_all[b] = np.where(true_states == 0,
                          np.random.normal(0.0, np.sqrt(0.5), T),
                          np.random.normal(1.5, np.sqrt(1.0), T))

print(f"Batch shape: {obs_all.shape}", flush=True)

predictor = HMMPredictor(use_torch=True, device="cpu")
results = predictor.predict(obs_all, threshold=0.5)

total_acc = 0.0
for i, res in enumerate(results):
    acc = np.mean(res['viterbi_states'] == true_states_all[i])
    total_acc += acc
    print(f"  Sample {i}: Viterbi acc={acc:.4f}, meth calls={res['states'].sum()}", flush=True)

print(f"Average Viterbi accuracy: {total_acc / B:.4f}", flush=True)

obs_list = [obs_all[i] for i in range(B)]
results2 = predictor.predict(obs_list, threshold=0.5)
print(f"List input works: {len(results2)} results", flush=True)

print('HMM PyTorch test PASSED', flush=True)
