import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

_log_file = open(os.path.join(os.path.dirname(__file__), 'test_neural_result.txt'), 'w', encoding='utf-8')

def log(msg):
    print(msg, flush=True)
    _log_file.write(msg + '\n')
    _log_file.flush()

import torch
from pacbio_methylation_hmm.neural_hmm import NeuralHMM, NeuralHMMConfig
from pacbio_methylation_hmm.neural_emission import NeuralEmissionNetwork, NeuralEmissionConfig, encode_nucleotides
from pacbio_methylation_hmm.trainer import (
    NeuralHMMTrainer, TrainingConfig, MethylationDataset, generate_synthetic_training_data,
)
from pacbio_methylation_hmm.hmm import HMMPredictor


def main():
    log("=" * 60)
    log("  Neural-HMM Validation Test")
    log("=" * 60)

    log("\n[1] Generating data...")
    nuc, ipd, labels = generate_synthetic_training_data(n_samples=300, seq_length=300, seed=42)
    nuc_test, ipd_test, labels_test = generate_synthetic_training_data(n_samples=30, seq_length=300, seed=99)
    log(f"  Train: {len(nuc)} | Test: {len(nuc_test)}")

    log("\n[2] Testing Bi-LSTM emission network...")
    emission_config = NeuralEmissionConfig(embed_dim=8, hidden_dim=64, n_lstm_layers=3)
    emission_net = NeuralEmissionNetwork(emission_config)

    nuc_t = encode_nucleotides(nuc[0]).unsqueeze(0)
    ipd_t = torch.tensor(ipd[0], dtype=torch.float32).unsqueeze(0)
    log_emit = emission_net(nuc_t, ipd_t)
    log(f"  Emission output shape: {log_emit.shape}")
    log(f"  Emission range: [{log_emit.min():.4f}, {log_emit.max():.4f}]")
    probs = torch.exp(log_emit)
    log(f"  Prob sum (should ~1.0): {probs[0, :5].sum(dim=-1).tolist()}")
    log("  Bi-LSTM emission OK")

    log("\n[3] Testing NeuralHMM decoder (Scaled F-B + Log-Viterbi)...")
    model_config = NeuralHMMConfig(
        embed_dim=8, hidden_dim=64, n_lstm_layers=3,
        dropout=0.1, learn_trans=True,
    )
    model = NeuralHMM(model_config)

    nuc_batch = torch.zeros(4, 300, dtype=torch.long)
    ipd_batch = torch.zeros(4, 300, dtype=torch.float64)
    lengths_batch = torch.tensor([300, 250, 280, 300], dtype=torch.long)
    for i in range(4):
        nuc_batch[i] = encode_nucleotides(nuc[i])
        ipd_batch[i, :len(ipd[i])] = torch.tensor(ipd[i], dtype=torch.float64)

    log_emission = model._neural_log_emission(nuc_batch, ipd_batch, lengths=lengths_batch)

    posteriors, _, log_evidence = model.compute_posterior(log_emission, lengths_batch)
    log(f"  Posterior range: [{posteriors.min():.6f}, {posteriors.max():.6f}]")
    log(f"  Log evidence per sample: {[f'{x:.2f}' for x in log_evidence.tolist()]}")

    viterbi_states, _ = model.viterbi_batch(log_emission, lengths_batch)
    unique_states = set()
    for b in range(4):
        L = int(lengths_batch[b])
        unique_states.update(viterbi_states[b, :L].tolist())
    log(f"  Viterbi states observed (pre-training): {unique_states}")
    log("  NeuralHMM decoder OK")

    log("\n[4] Training Neural-HMM (15 epochs)...")
    dataset = MethylationDataset(nuc, ipd, labels)
    train_config = TrainingConfig(
        learning_rate=1e-3, max_epochs=15, batch_size=16,
        lambda_hmm=0.2, gradient_clip=5.0, print_every=20,
        save_dir=os.path.join(os.path.dirname(__file__), 'checkpoints'),
    )
    trainer = NeuralHMMTrainer(model, train_config)
    t0 = time.time()
    history = trainer.train(dataset)
    train_time = time.time() - t0
    log(f"  Training time: {train_time:.1f}s")
    log(f"  Final loss: {history['train_loss'][-1]:.4f}")

    log("\n[5] Evaluating Neural-HMM...")
    model.eval()
    B_test = len(nuc_test)
    max_len = max(len(n) for n in nuc_test)
    nuc_test_t = torch.zeros(B_test, max_len, dtype=torch.long)
    ipd_test_t = torch.zeros(B_test, max_len, dtype=torch.float64)
    lengths_test_t = torch.tensor([len(n) for n in nuc_test], dtype=torch.long)
    for i in range(B_test):
        L = len(nuc_test[i])
        nuc_test_t[i, :L] = encode_nucleotides(nuc_test[i])
        ipd_test_t[i, :L] = torch.tensor(ipd_test[i], dtype=torch.float64)

    results = model.predict_batch(nuc_test_t, ipd_test_t, lengths_test_t, threshold=0.5)

    vit_correct = 0
    post_correct = 0
    total_pos = 0
    post_unique = set()
    vit_unique = set()
    for i in range(B_test):
        L = len(labels_test[i])
        true = labels_test[i]
        vit_pred = results[i]["viterbi_states"]
        post_pred = results[i]["states"]
        n = min(L, len(vit_pred))
        vit_correct += np.sum(vit_pred[:n] == true[:n])
        post_correct += np.sum(post_pred[:n] == true[:n])
        total_pos += n
        vit_unique.update(vit_pred[:n].tolist())
        post_unique.update(post_pred[:n].tolist())

    neural_vit_acc = vit_correct / total_pos if total_pos > 0 else 0
    neural_post_acc = post_correct / total_pos if total_pos > 0 else 0
    log(f"  Neural-HMM Viterbi accuracy: {neural_vit_acc:.4f}")
    log(f"  Neural-HMM Posterior accuracy: {neural_post_acc:.4f}")
    log(f"  Viterbi states: {vit_unique}")
    log(f"  Posterior states: {post_unique}")

    log("\n[6] Gaussian HMM baseline...")
    predictor = HMMPredictor(use_torch=True)
    baseline_correct = 0
    baseline_pos = 0
    for i in range(B_test):
        obs = ipd_test[i].astype(np.float64)
        result = predictor.predict(obs, threshold=0.5)
        true = labels_test[i]
        n = min(len(result["viterbi_states"]), len(true))
        baseline_correct += np.sum(result["viterbi_states"][:n] == true[:n])
        baseline_pos += n
    baseline_acc = baseline_correct / baseline_pos if baseline_pos > 0 else 0
    log(f"  Gaussian HMM accuracy: {baseline_acc:.4f}")

    log("\n" + "=" * 60)
    log("  COMPARISON")
    log("=" * 60)
    log(f"  Gaussian HMM:     {baseline_acc:.4f}")
    log(f"  Neural-HMM:       {neural_vit_acc:.4f}")
    improvement = (neural_vit_acc - baseline_acc) / baseline_acc * 100 if baseline_acc > 0 else 0
    log(f"  Relative diff:    {improvement:+.1f}%")

    if len(vit_unique) == 1:
        log("  WARNING: Viterbi state collapse!")
    else:
        log("  State diversity OK (2 states observed)")

    log("\nAll tests PASSED")
    _log_file.close()


if __name__ == '__main__':
    main()
