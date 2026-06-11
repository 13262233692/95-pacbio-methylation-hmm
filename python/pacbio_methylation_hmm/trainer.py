"""
Training pipeline for the Neural-HMM methylation detector.

Provides:
  - MethylationDataset: PyTorch Dataset for methylation sequence data
  - NeuralHMMTrainer: Training loop with hybrid CE + HMM evidence loss
  - Synthetic data generation for testing and development
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
import time

from .neural_hmm import NeuralHMM, NeuralHMMConfig
from .neural_emission import encode_nucleotides


@dataclass
class TrainingConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    lambda_hmm: float = 0.1
    max_epochs: int = 30
    batch_size: int = 16
    gradient_clip: float = 5.0
    lr_scheduler: str = "cosine"
    warmup_steps: int = 100
    print_every: int = 10
    save_dir: str = "checkpoints"
    device: str = "cpu"


class MethylationDataset(Dataset):
    """
    Dataset for methylation sequence training.

    Each sample contains:
      - nucleotides: (T,) long tensor of base indices
      - ipd_values: (T,) float tensor of normalized IPD
      - labels: (T,) long tensor (0=unmethylated, 1=methylated)
      - length: int
    """

    def __init__(
        self,
        nucleotides: List[np.ndarray],
        ipd_values: List[np.ndarray],
        labels: List[np.ndarray],
        pulse_widths: Optional[List[np.ndarray]] = None,
    ):
        assert len(nucleotides) == len(ipd_values) == len(labels)
        self.nucleotides = nucleotides
        self.ipd_values = ipd_values
        self.labels = labels
        self.pulse_widths = pulse_widths

    def __len__(self) -> int:
        return len(self.nucleotides)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        nuc = self.nucleotides[idx]
        if isinstance(nuc, np.ndarray):
            nuc = encode_nucleotides(nuc)
        ipd = torch.tensor(self.ipd_values[idx], dtype=torch.float32)
        lab = torch.tensor(self.labels[idx], dtype=torch.long)

        item = {
            "nucleotides": nuc,
            "ipd_values": ipd,
            "labels": lab,
            "length": len(lab),
        }

        if self.pulse_widths is not None:
            item["pulse_width"] = torch.tensor(self.pulse_widths[idx], dtype=torch.float32)

        return item


def collate_variable_length(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for variable-length sequences with padding.
    """
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    max_len = lengths.max().item()

    nuc_padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    ipd_padded = torch.zeros(len(batch), max_len, dtype=torch.float32)
    lab_padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    pw_padded = None

    has_pw = "pulse_width" in batch[0]
    if has_pw:
        pw_padded = torch.zeros(len(batch), max_len, dtype=torch.float32)

    for i, item in enumerate(batch):
        L = item["length"]
        nuc_padded[i, :L] = item["nucleotides"]
        ipd_padded[i, :L] = item["ipd_values"]
        lab_padded[i, :L] = item["labels"]
        if has_pw:
            pw_padded[i, :L] = item["pulse_width"]

    result = {
        "nucleotides": nuc_padded,
        "ipd_values": ipd_padded,
        "labels": lab_padded,
        "lengths": lengths,
    }
    if pw_padded is not None:
        result["pulse_width"] = pw_padded

    return result


class NeuralHMMTrainer:
    """
    Trainer for the Neural-HMM methylation detector.

    Supports:
      - Hybrid CE + HMM evidence loss
      - Gradient clipping
      - Learning rate scheduling
      - Checkpoint saving
    """

    def __init__(
        self,
        model: NeuralHMM,
        config: Optional[TrainingConfig] = None,
    ):
        self.model = model
        self.config = config if config is not None else TrainingConfig()
        self.device = torch.device(self.config.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.history = {
            "train_loss": [],
            "train_ce": [],
            "train_hmm": [],
        }

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int = 0,
    ) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_hmm = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            nucleotides = batch["nucleotides"].to(self.device)
            ipd_values = batch["ipd_values"].to(self.device).to(torch.float64)
            labels = batch["labels"].to(self.device)
            lengths = batch["lengths"].to(self.device)
            pulse_width = None
            if "pulse_width" in batch:
                pulse_width = batch["pulse_width"].to(self.device).to(torch.float64)

            loss_dict = self.model.compute_loss(
                nucleotides=nucleotides,
                ipd_values=ipd_values,
                lengths=lengths,
                labels=labels,
                pulse_width=pulse_width,
                lambda_hmm=self.config.lambda_hmm,
            )

            self.optimizer.zero_grad()
            loss_dict["loss"].backward()

            if self.config.gradient_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip
                )

            self.optimizer.step()

            total_loss += loss_dict["loss"].item()
            total_ce += loss_dict["ce_loss"].item()
            total_hmm += loss_dict["hmm_loss"].item()
            n_batches += 1

            if (batch_idx + 1) % self.config.print_every == 0:
                avg_loss = total_loss / n_batches
                avg_ce = total_ce / n_batches
                avg_hmm = total_hmm / n_batches
                print(
                    f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(dataloader)} | "
                    f"Loss: {avg_loss:.4f} (CE: {avg_ce:.4f}, HMM: {avg_hmm:.4f})",
                    flush=True,
                )

        avg_loss = total_loss / max(n_batches, 1)
        avg_ce = total_ce / max(n_batches, 1)
        avg_hmm = total_hmm / max(n_batches, 1)

        self.history["train_loss"].append(avg_loss)
        self.history["train_ce"].append(avg_ce)
        self.history["train_hmm"].append(avg_hmm)

        return {"loss": avg_loss, "ce_loss": avg_ce, "hmm_loss": avg_hmm}

    def train(
        self,
        dataset: MethylationDataset,
        val_dataset: Optional[MethylationDataset] = None,
    ) -> Dict[str, List[float]]:
        """
        Full training loop.
        """
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_variable_length,
            num_workers=0,
            drop_last=False,
        )

        n_steps = len(dataloader) * self.config.max_epochs
        if self.config.lr_scheduler == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=n_steps
            )
        elif self.config.lr_scheduler == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=max(1, n_steps // 3), gamma=0.1
            )
        else:
            self.scheduler = None

        print(f"Starting training: {self.config.max_epochs} epochs, {len(dataset)} samples", flush=True)
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}", flush=True)

        best_loss = float("inf")
        for epoch in range(self.config.max_epochs):
            t0 = time.time()
            metrics = self.train_epoch(dataloader, epoch)
            elapsed = time.time() - t0

            if self.scheduler is not None:
                self.scheduler.step()

            print(
                f"Epoch {epoch+1}/{self.config.max_epochs} ({elapsed:.1f}s) | "
                f"Loss: {metrics['loss']:.4f} | CE: {metrics['ce_loss']:.4f} | "
                f"HMM: {metrics['hmm_loss']:.4f}",
                flush=True,
            )

            if metrics["loss"] < best_loss:
                best_loss = metrics["loss"]
                self._save_checkpoint(epoch, metrics["loss"])

        return self.history

    def _save_checkpoint(self, epoch: int, loss: float):
        if not os.path.exists(self.config.save_dir):
            os.makedirs(self.config.save_dir, exist_ok=True)
        path = os.path.join(self.config.save_dir, f"neural_hmm_epoch{epoch+1}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
        }, path)

    @staticmethod
    def load_checkpoint(model: NeuralHMM, path: str, device: str = "cpu"):
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model


def generate_synthetic_training_data(
    n_samples: int = 500,
    seq_length: int = 500,
    seed: int = 42,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Generate synthetic methylation data for testing the Neural-HMM.

    Returns:
        nucleotides, ipd_values, labels
    """
    rng = np.random.RandomState(seed)

    nucleotides_list = []
    ipd_list = []
    labels_list = []

    for _ in range(n_samples):
        bases = rng.randint(0, 4, size=seq_length)

        labels = np.zeros(seq_length, dtype=np.int64)
        meth_start = seq_length // 4 + rng.randint(-50, 50)
        meth_end = seq_length // 2 + rng.randint(-50, 50)
        meth_start = max(0, meth_start)
        meth_end = min(seq_length, meth_end)

        for t in range(1, seq_length):
            if meth_start <= t < meth_end:
                labels[t] = 1 if rng.rand() < 0.90 else 0
            else:
                labels[t] = 1 if rng.rand() < 0.05 else 0

        ipd = np.where(
            labels == 0,
            rng.normal(0.0, np.sqrt(0.5), seq_length),
            rng.normal(1.5, np.sqrt(1.0), seq_length),
        )

        cpg_positions = []
        for t in range(seq_length - 1):
            if bases[t] == 1 and bases[t + 1] == 2:
                cpg_positions.append(t)
        for pos in cpg_positions:
            if labels[pos] == 1:
                ipd[pos] += rng.normal(0.5, 0.3)

        nucleotides_list.append(bases.astype(np.uint8))
        ipd_list.append(ipd.astype(np.float32))
        labels_list.append(labels)

    return nucleotides_list, ipd_list, labels_list
