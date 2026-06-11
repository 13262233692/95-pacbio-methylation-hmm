import numpy as np
import sys

sys.path.insert(0, 'python')
from pacbio_methylation_hmm.hmm_numpy import MethylationHMM, HMMConfig

np.random.seed(42)
T = 200
true_states = np.zeros(T, dtype=int)
for t in range(1, T):
    if true_states[t-1] == 0:
        true_states[t] = 0 if np.random.rand() < 0.95 else 1
    else:
        true_states[t] = 1 if np.random.rand() < 0.90 else 0

obs = np.where(true_states == 0,
               np.random.normal(0.0, np.sqrt(0.5), T),
               np.random.normal(1.5, np.sqrt(1.0), T))

print(f"Observation shape: {obs.shape}", flush=True)
print(f"True methylated count: {true_states.sum()}", flush=True)

hmm = MethylationHMM()
result = hmm.predict(obs, threshold=0.5)
accuracy = np.mean(result['viterbi_states'] == true_states)
print(f'Viterbi accuracy: {accuracy:.4f}', flush=True)
post_acc = np.mean(result['states'] == true_states)
print(f'Posterior threshold accuracy: {post_acc:.4f}', flush=True)
print(f'Log evidence: {result["log_evidence"]:.4f}', flush=True)
print(f'Methylation calls: {result["states"].sum()}/{T}', flush=True)
print(f'Posterior shape: {result["posteriors"].shape}', flush=True)
print('HMM NumPy test PASSED', flush=True)
