#!/usr/bin/env python
"""
Neural-HMM Training and Inference Demo.

Demonstrates the complete workflow:
1. Generate synthetic PacBio methylation training data
2. Train the Bi-LSTM Neural-HMM hybrid model
3. Run inference with neural emission + HMM decoding
4. Compare accuracy against the pure Gaussian HMM baseline
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import torch
from pacbio_methylation_hmm.neural_hmm import NeuralHMM, NeuralHMMConfig
from pacbio_methylation_hmm.neural_emission import encode_nucleotides
from pacbio_methylation_hmm.trainer import (
    NeuralHMMTrainer,
    TrainingConfig,
    MethylationDataset,
    collate_variable_length,
    generate_synthetic_training_data,
)
from pacbio_methylation_hmm.hmm import HMMPredictor


def evaluate_neural_hmm(model, nucleotides, ipd_values, labels, device="cpu"):
    model.eval()
    B = len(nucleotides)
    max_len = max(len(n) for n in nucleotides)
    lengths = torch.tensor([len(n) for n in nucleotides], dtype=torch.long)

    nuc_tensor = torch.zeros(B, max_len, dtype=torch.long)
    ipd_tensor = torch.zeros(B, max_len, dtype=torch.float64)
    for i in range(B):
        L = len(nucleotides[i])
        nuc_tensor[i, :L] = encode_nucleotides(nucleotides[i])
        ipd_tensor[i, :L] = torch.tensor(ipd_values[i], dtype=torch.float64)

    nuc_tensor = nuc_tensor.to(device)
    ipd_tensor = ipd_tensor.to(device)
    lengths = lengths.to(device)

    results = model.predict_batch(nuc_tensor, ipd_tensor, lengths, threshold=0.5)

    total_correct = 0
    total_positions = 0
    viterbi_correct = 0
    for i in range(B):
        L = len(labels[i])
        true = labels[i]
        pred_post = results[i]["states"]
        pred_vit = results[i]["viterbi_states"]
        n = min(L, len(pred_post))
        total_correct += np.sum(pred_post[:n] == true[:n])
        viterbi_correct += np.sum(pred_vit[:n] == true[:n])
        total_positions += n

    posterior_acc = total_correct / total_positions if total_positions > 0 else 0
    viterbi_acc = viterbi_correct / total_positions if total_positions > 0 else 0
    return posterior_acc, viterbi_acc


def evaluate_gaussian_hmm(ipd_values, labels):
    predictor = HMMPredictor(use_torch=True, device="cpu")
    total_correct = 0
    total_positions = 0
    for i in range(len(ipd_values)):
        obs = ipd_values[i].astype(np.float64)
        result = predictor.predict(obs, threshold=0.5)
        true = labels[i]
        pred = result["viterbi_states"]
        n = min(len(pred), len(true))
        total_correct += np.sum(pred[:n] == true[:n])
        total_positions += n
    return total_correct / total_positions if total_positions > 0 else 0


def main():
    print("=" * 70, flush=True)
    print("  Neural-HMM (Bi-LSTM + Scaled HMM) Training & Inference Demo", flush=True)
    print("=" * 70, flush=True)

    # --- Step 1: Generate synthetic data ---
    print("\n[Step 1] Generating synthetic training data...", flush=True)
    t0 = time.time()
    nuc_train, ipd_train, labels_train = generate_synthetic_training_data(
        n_samples=200, seq_length=300, seed=42
    )
    nuc_test, ipd_test, labels_test = generate_synthetic_training_data(
        n_samples=50, seq_length=300, seed=999
    )
    print(f"  Train: {len(nuc_train)} samples | Test: {len(nuc_test)} samples", flush=True)
    print(f"  Generated in {time.time()-t0:.2f}s", flush=True)

    # --- Step 2: Evaluate Gaussian HMM baseline ---
    print("\n[Step 2] Gaussian HMM baseline...", flush=True)
    t0 = time.time()
    baseline_acc = evaluate_gaussian_hmm(ipd_test, labels_test)
    print(f"  Gaussian HMM Viterbi accuracy: {baseline_acc:.4f} ({time.time()-t0:.2f}s)", flush=True)

    # --- Step 3: Build Neural-HMM ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Step 3] Building Neural-HMM on {device}...", flush=True)

    model_config = NeuralHMMConfig(
        device=device,
        embed_dim=16,
        hidden_dim=64,
        n_lstm_layers=2,
        dropout=0.1,
        learn_trans=True,
    )
    model = NeuralHMM(model_config)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,} total, {n_trainable:,} trainable", flush=True)

    # --- Step 4: Train ---
    print("\n[Step 4] Training...", flush=True)
    dataset = MethylationDataset(nuc_train, ipd_train, labels_train)

    train_config = TrainingConfig(
        learning_rate=1e-3,
        weight_decay=1e-5,
        lambda_hmm=0.1,
        max_epochs=10,
        batch_size=16,
        gradient_clip=5.0,
        device=device,
        print_every=5,
        save_dir=os.path.join(os.path.dirname(__file__), '..', 'checkpoints'),
    )

    trainer = NeuralHMMTrainer(model, train_config)
    t0 = time.time()
    history = trainer.train(dataset)
    train_time = time.time() - t0
    print(f"  Training completed in {train_time:.1f}s", flush=True)

    # --- Step 5: Evaluate Neural-HMM ---
    print("\n[Step 5] Evaluating Neural-HMM...", flush=True)
    t0 = time.time()
    post_acc, vit_acc = evaluate_neural_hmm(model, nuc_test, ipd_test, labels_test, device)
    eval_time = time.time() - t0
    print(f"  Neural-HMM Posterior accuracy:  {post_acc:.4f}", flush=True)
    print(f"  Neural-HMM Viterbi accuracy:    {vit_acc:.4f} ({eval_time:.2f}s)", flush=True)

    # --- Step 6: Compare ---
    print("\n" + "=" * 70, flush=True)
    print("  RESULTS COMPARISON", flush=True)
    print("=" * 70, flush=True)
    print(f"  Gaussian HMM baseline:   {baseline_acc:.4f}", flush=True)
    print(f"  Neural-HMM (Bi-LSTM):    {vit_acc:.4f}", flush=True)
    improvement = (vit_acc - baseline_acc) / baseline_acc * 100 if baseline_acc > 0 else 0
    print(f"  Relative improvement:    {improvement:+.1f}%", flush=True)

    state_set = set()
    for i in range(len(nuc_test)):
        L = len(labels_test[i])
        nuc_t = encode_nucleotides(nuc_test[i]).unsqueeze(0).to(device)
        ipd_t = torch.tensor(ipd_test[i], dtype=torch.float64).unsqueeze(0).to(device)
        len_t = torch.tensor([L], dtype=torch.long).to(device)
        result = model.predict_batch(nuc_t, ipd_t, len_t, threshold=0.5)
        state_set.update(result[0]["viterbi_states"].tolist())

    if len(state_set) == 1:
        print(f"  *** WARNING: Viterbi states collapsed to single value: {state_set}", flush=True)
    else:
        print(f"  Viterbi state diversity: {len(state_set)} states observed OK", flush=True)

    print("\n  Demo complete!", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
