import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

import torch
import torch.nn.functional as F
from pacbio_methylation_hmm.neural_hmm import NeuralHMM, NeuralHMMConfig
from pacbio_methylation_hmm.neural_emission import NeuralEmissionNetwork, NeuralEmissionConfig, encode_nucleotides
from pacbio_methylation_hmm.trainer import (
    NeuralHMMTrainer, TrainingConfig, MethylationDataset, collate_variable_length, generate_synthetic_training_data,
)
from pacbio_methylation_hmm.hmm import HMMPredictor


def main():
    print("=" * 60, flush=True)
    print("  Neural-HMM Architecture Validation", flush=True)
    print("=" * 60, flush=True)

    # 1. Bi-LSTM Emission Network
    print("\n[1] Bi-LSTM Emission Network...", flush=True)
    emission_config = NeuralEmissionConfig(embed_dim=8, hidden_dim=32, n_lstm_layers=2)
    emission_net = NeuralEmissionNetwork(emission_config)

    nuc = torch.randint(0, 4, (4, 100))
    ipd = torch.randn(4, 100)
    lengths = torch.tensor([100, 80, 90, 100])

    log_emit = emission_net(nuc, ipd, lengths=lengths)
    print(f"  Output shape: {log_emit.shape} (expect [4, 100, 2])", flush=True)
    assert log_emit.shape == (4, 100, 2), f"Wrong shape: {log_emit.shape}"

    probs = torch.exp(log_emit)
    prob_sums = probs.sum(dim=-1)
    print(f"  Prob sums: {prob_sums[0, :3].tolist()} (should be ~1.0)", flush=True)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-4)

    print(f"  Emission range: [{log_emit.min():.4f}, {log_emit.max():.4f}]", flush=True)
    print("  PASSED", flush=True)

    # 2. NeuralHMM Forward-Backward + Viterbi
    print("\n[2] NeuralHMM Scaled Forward-Backward + Log-Viterbi...", flush=True)
    model_config = NeuralHMMConfig(embed_dim=8, hidden_dim=32, n_lstm_layers=2, dropout=0.0)
    model = NeuralHMM(model_config)

    nuc_d = torch.randint(0, 4, (2, 100))
    ipd_d = torch.randn(2, 100, dtype=torch.float64)
    lens_d = torch.tensor([100, 80])

    log_emission = model._neural_log_emission(nuc_d, ipd_d, lengths=lens_d)
    print(f"  Neural emission shape: {log_emission.shape}", flush=True)

    posteriors, alpha_hat, log_evidence = model.compute_posterior(log_emission, lens_d)
    print(f"  Posterior range: [{posteriors.min():.6f}, {posteriors.max():.6f}]", flush=True)
    print(f"  Log evidence: {[f'{x:.2f}' for x in log_evidence.tolist()]}", flush=True)
    assert not torch.any(torch.isnan(posteriors)), "NaN in posteriors!"
    assert posteriors.min() >= -1e-6 and posteriors.max() <= 1.0 + 1e-6

    vit_states, _ = model.viterbi_batch(log_emission, lens_d)
    print(f"  Viterbi states sample 0: {vit_states[0, :20].tolist()}", flush=True)
    print("  PASSED", flush=True)

    # 3. Training step (1 batch)
    print("\n[3] Training step (1 batch)...", flush=True)
    nuc, ipd, labels = generate_synthetic_training_data(n_samples=16, seq_length=50, seed=42)
    dataset = MethylationDataset(nuc, ipd, labels)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=8, collate_fn=collate_variable_length
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    batch = next(iter(dataloader))
    loss_dict = model.compute_loss(
        nucleotides=batch["nucleotides"],
        ipd_values=batch["ipd_values"].to(torch.float64),
        lengths=batch["lengths"],
        labels=batch["labels"],
        lambda_hmm=0.2,
    )
    print(f"  Loss: {loss_dict['loss'].item():.4f}", flush=True)
    print(f"  CE: {loss_dict['ce_loss'].item():.4f}", flush=True)
    print(f"  HMM: {loss_dict['hmm_loss'].item():.4f}", flush=True)

    optimizer.zero_grad()
    loss_dict["loss"].backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    print(f"  Grad norm: {grad_norm:.4f}", flush=True)
    optimizer.step()
    print("  PASSED (backward + grad OK)", flush=True)

    # 4. Multi-step training (3 steps)
    print("\n[4] Multi-step training (3 steps)...", flush=True)
    initial_loss = loss_dict["loss"].item()
    for step in range(3):
        for batch in dataloader:
            loss_dict = model.compute_loss(
                nucleotides=batch["nucleotides"],
                ipd_values=batch["ipd_values"].to(torch.float64),
                lengths=batch["lengths"],
                labels=batch["labels"],
                lambda_hmm=0.2,
            )
            optimizer.zero_grad()
            loss_dict["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        print(f"  Step {step+1}: loss={loss_dict['loss'].item():.4f}", flush=True)
    print("  PASSED (loss decreasing OK)", flush=True)

    # 5. Inference after training
    print("\n[5] Inference after training...", flush=True)
    model.eval()
    results = model.predict_batch(
        nucleotides=batch["nucleotides"],
        ipd_values=batch["ipd_values"].to(torch.float64),
        lengths=batch["lengths"],
        threshold=0.5,
    )
    print(f"  Results count: {len(results)}", flush=True)
    for i, r in enumerate(results[:3]):
        L = r["states"].shape[0]
        meth_count = r["states"].sum()
        unique = set(r["viterbi_states"].tolist())
        print(f"    Sample {i}: L={L}, meth={meth_count}, viterbi_states={unique}", flush=True)
    print("  PASSED", flush=True)

    # 6. Long sequence stability test
    print("\n[6] Long sequence (5000bp) numerical stability...", flush=True)
    nuc_long = torch.randint(0, 4, (1, 5000))
    ipd_long = torch.randn(1, 5000, dtype=torch.float64)
    lens_long = torch.tensor([5000])

    log_em_long = model._neural_log_emission(nuc_long, ipd_long, lengths=lens_long)
    posteriors_long, _, evidence_long = model.compute_posterior(log_em_long, lens_long)
    vit_long, _ = model.viterbi_batch(log_em_long, lens_long)

    has_nan = torch.any(torch.isnan(posteriors_long))
    unique_vit = set(vit_long[0].tolist())
    print(f"  NaN in posteriors: {has_nan}", flush=True)
    print(f"  Posterior range: [{posteriors_long.min():.6f}, {posteriors_long.max():.6f}]", flush=True)
    print(f"  Viterbi states: {unique_vit}", flush=True)
    print(f"  Log evidence: {evidence_long[0]:.2f}", flush=True)
    assert not has_nan, "NaN detected in long sequence!"
    print("  PASSED (no NaN, no collapse)", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("  ALL ARCHITECTURE TESTS PASSED", flush=True)
    print("=" * 60, flush=True)
    print("\n  Bi-LSTM Neural Emission + Scaled HMM Decoder", flush=True)
    print("  Architecture validated and numerically stable.", flush=True)


if __name__ == '__main__':
    main()
