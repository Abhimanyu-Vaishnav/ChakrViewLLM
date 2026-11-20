"""
UCIP-18 Controlled Conversational Learning Pilot (300 Steps).
Strictly bounded experimental pilot testing whether the reconstructed UCIP-17 dataset
and decoupled gradient objective (L = 0.85 * L_SFT + 0.15 * L_replay) can drive
instruction acquisition without catastrophic language degradation.

Hard Constraints:
- Maximum 300 optimizer steps.
- Decoupled token normalization: L_SFT normalized by SFT active tokens, L_replay normalized by replay tokens.
- Loss: 0.85 * (sft_loss_sum / sft_tokens) + 0.15 * (rep_loss_sum / rep_tokens).
- AdamW (beta1=0.9, beta2=0.95, weight_decay=0.01).
- Learning rate: 2.5e-5 peak with 30-step warmup, cosine decay to 2.5e-6.
- Checkpoints saved to experiments/ucip18/pilot/checkpoint_best.pt and checkpoint_step_0300.pt.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.training.safety import assert_safe_checkpoint_target
from chakrview.training.sft_engine import SFTDataset
from chakrview.training.mixture_sft_engine import StageCReplayDataset


@dataclass
class PilotConfig:
    output_dir: Path = ROOT / "experiments" / "ucip18" / "pilot"
    base_checkpoint: Path = ROOT / "experiments" / "ucip13" / "checkpoints" / "primary_seed42" / "checkpoint_step_800.pt"
    train_file: Path = ROOT / "data" / "sft" / "ucip17" / "sft_train_12000.jsonl"
    val_file: Path = ROOT / "data" / "sft" / "ucip17" / "sft_validation_1500.jsonl"
    dev_file: Path = ROOT / "data" / "sft" / "ucip17" / "sft_dev_1500.jsonl"
    stage_c_train_dir: Path = ROOT / "data" / "tokenized" / "stage_c" / "train"
    stage_c_val_shard: Path = ROOT / "data" / "tokenized" / "stage_c" / "validation" / "shard_00000.bin"
    tokenizer_dir: Path = ROOT / "data" / "experiments" / "vocab_4096"
    steps: int = 300
    sft_batch_size: int = 12
    replay_batch_size: int = 1
    sft_weight: float = 0.85
    replay_weight: float = 0.15
    learning_rate: float = 2.5e-5
    min_learning_rate: float = 2.5e-6
    warmup_steps: int = 30
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    val_interval: int = 50
    save_interval: int = 100
    seed: int = 42
    device: str = "cpu"


class UCIP18PilotTrainer:
    def __init__(self, cfg: PilotConfig):
        self.cfg = cfg
        assert_safe_checkpoint_target(cfg.output_dir)
        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        print("[UCIP-18] Loading tokenizer artifacts...")
        self.tokenizer, _ = load_tokenizer_artifacts(cfg.tokenizer_dir)

        print("[UCIP-18] Loading datasets...")
        self.sft_train = SFTDataset(cfg.train_file, self.tokenizer, max_seq_len=512, is_training=True)
        self.sft_val = SFTDataset(cfg.val_file, self.tokenizer, max_seq_len=512, is_training=False)
        self.replay_train = StageCReplayDataset(cfg.stage_c_train_dir, max_seq_len=512)

        # Stage C validation data for retention evaluation
        self.stage_c_val_data = np.fromfile(cfg.stage_c_val_shard, dtype=np.uint16)

        # Load base checkpoint Step 800
        print(f"[UCIP-18] Loading base checkpoint: {cfg.base_checkpoint}")
        ckpt = torch.load(cfg.base_checkpoint, map_location=cfg.device, weights_only=False)
        self.model_config = ModelConfig(**ckpt["config"])
        self.model = ChakrMicro(self.model_config).to(cfg.device)
        self.model.load_state_dict(ckpt["model_state_dict"])

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=cfg.weight_decay,
        )
        self.loss_fn_sft = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
        self.loss_fn_replay = nn.CrossEntropyLoss(reduction="sum")

        self.history: List[Dict[str, Any]] = []

    @torch.no_grad()
    def evaluate_sft_loss(self, num_batches: int = 10, seed: int = 101) -> float:
        self.model.eval()
        rng = np.random.RandomState(seed)
        total_loss = 0.0
        total_tokens = 0
        for _ in range(num_batches):
            x, y, active_count = self.sft_val.get_batch(batch_size=8, rng=rng)
            if active_count == 0:
                continue
            x, y = x.to(self.cfg.device), y.to(self.cfg.device)
            logits = self.model(x)
            loss_sum = self.loss_fn_sft(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            total_loss += loss_sum.item()
            total_tokens += active_count
        self.model.train()
        return round(total_loss / max(1, total_tokens), 4)

    @torch.no_grad()
    def evaluate_retention_loss(self, num_chunks: int = 20) -> float:
        """Evaluate next-token language modeling loss on Stage C validation partition."""
        self.model.eval()
        losses = []
        loss_fn = nn.CrossEntropyLoss()
        for i in range(num_chunks):
            chunk = self.stage_c_val_data[i * 512 : (i + 1) * 512].astype(np.int64)
            x = torch.tensor([chunk[:-1]], dtype=torch.long, device=self.cfg.device)
            y = torch.tensor([chunk[1:]], dtype=torch.long, device=self.cfg.device)
            logits = self.model(x)
            l = loss_fn(logits.view(-1, 4096), y.view(-1))
            losses.append(l.item())
        self.model.train()
        return round(float(np.mean(losses)), 4)

    def train(self) -> Dict[str, Any]:
        rng = np.random.RandomState(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)

        init_sft_val = self.evaluate_sft_loss(seed=101)
        init_ret_val = self.evaluate_retention_loss(num_chunks=20)
        print(f"[Init] Baseline SFT Val Loss: {init_sft_val:.4f} | Stage C Retention Loss: {init_ret_val:.4f}")

        best_val = init_sft_val
        best_ckpt_path = self.cfg.output_dir / "checkpoint_best.pt"

        start_time = time.perf_counter()

        for step in range(1, self.cfg.steps + 1):
            t0 = time.perf_counter()

            # Schedule: linear warmup then cosine decay
            if step <= self.cfg.warmup_steps:
                cur_lr = self.cfg.learning_rate * (step / max(1, self.cfg.warmup_steps))
            else:
                progress = (step - self.cfg.warmup_steps) / max(1, self.cfg.steps - self.cfg.warmup_steps)
                coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
                cur_lr = self.cfg.min_learning_rate + coeff * (self.cfg.learning_rate - self.cfg.min_learning_rate)

            for param_group in self.optimizer.param_groups:
                param_group["lr"] = cur_lr

            self.optimizer.zero_grad()

            # 1. SFT micro-batch
            x_sft, y_sft, sft_tokens = self.sft_train.get_batch(batch_size=self.cfg.sft_batch_size, rng=rng)
            x_sft, y_sft = x_sft.to(self.cfg.device), y_sft.to(self.cfg.device)
            logits_sft = self.model(x_sft)
            sft_loss_sum = self.loss_fn_sft(logits_sft.reshape(-1, 4096), y_sft.reshape(-1))
            sft_mean_loss = sft_loss_sum / max(1, sft_tokens)

            # 2. Stage C Replay micro-batch
            x_rep, y_rep, rep_tokens = self.replay_train.get_batch(batch_size=self.cfg.replay_batch_size, rng=rng)
            x_rep, y_rep = x_rep.to(self.cfg.device), y_rep.to(self.cfg.device)
            logits_rep = self.model(x_rep)
            rep_loss_sum = self.loss_fn_replay(logits_rep.reshape(-1, 4096), y_rep.reshape(-1))
            rep_mean_loss = rep_loss_sum / max(1, rep_tokens)

            # Decoupled gradient objective:
            # L = 0.85 * (sft_loss_sum / sft_tokens) + 0.15 * (rep_loss_sum / rep_tokens)
            loss = self.cfg.sft_weight * sft_mean_loss + self.cfg.replay_weight * rep_mean_loss
            loss.backward()

            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip).item()
            self.optimizer.step()

            dt = time.perf_counter() - t0

            # Logging
            if step % 10 == 0 or step == 1 or step == self.cfg.steps:
                print(
                    f"Step {step:03d}/{self.cfg.steps:03d} | "
                    f"Loss: {loss.item():.4f} (SFT: {sft_mean_loss.item():.4f}, Rep: {rep_mean_loss.item():.4f}) | "
                    f"Grad: {grad_norm:.3f} | LR: {cur_lr:.2e} | {dt:.2f}s"
                )

            # Validation interval
            if step % self.cfg.val_interval == 0 or step == self.cfg.steps:
                val_sft = self.evaluate_sft_loss(seed=101 + step)
                val_ret = self.evaluate_retention_loss(num_chunks=20)
                print(f"--> [Val Step {step:03d}] SFT Val Loss: {val_sft:.4f} | Retention Val Loss: {val_ret:.4f}")

                rec = {
                    "step": step,
                    "train_loss": round(loss.item(), 4),
                    "sft_train_loss": round(sft_mean_loss.item(), 4),
                    "replay_train_loss": round(rep_mean_loss.item(), 4),
                    "val_sft_loss": val_sft,
                    "val_retention_loss": val_ret,
                    "grad_norm": round(grad_norm, 4),
                    "lr": cur_lr,
                    "sft_tokens": sft_tokens,
                    "replay_tokens": rep_tokens,
                }
                self.history.append(rec)

                # Save best checkpoint
                if val_sft < best_val:
                    best_val = val_sft
                    self.save_checkpoint(best_ckpt_path, step, val_sft, val_ret)

            # Periodic checkpoint
            if step % self.cfg.save_interval == 0 or step == self.cfg.steps:
                ckpt_path = self.cfg.output_dir / f"checkpoint_step_{step:04d}.pt"
                self.save_checkpoint(ckpt_path, step, sft_mean_loss.item(), rep_mean_loss.item())

        total_time = time.perf_counter() - start_time
        print(f"[UCIP-18] Training completed in {total_time:.1f}s.")

        summary = {
            "initial_sft_val_loss": init_sft_val,
            "final_sft_val_loss": self.history[-1]["val_sft_loss"] if self.history else None,
            "best_sft_val_loss": best_val,
            "initial_retention_loss": init_ret_val,
            "final_retention_loss": self.history[-1]["val_retention_loss"] if self.history else None,
            "total_steps": self.cfg.steps,
            "total_time_seconds": round(total_time, 2),
            "history": self.history,
        }
        with open(self.cfg.output_dir / "training_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary

    def save_checkpoint(self, path: Path, step: int, sft_val: float, ret_val: float):
        payload = {
            "step": step,
            "model_state_dict": copy.deepcopy(self.model.state_dict()),
            "optimizer_state_dict": copy.deepcopy(self.optimizer.state_dict()),
            "config": self.model_config.to_dict() if hasattr(self.model_config, "to_dict") else vars(self.model_config),
            "sft_val_loss": sft_val,
            "retention_val_loss": ret_val,
            "history": self.history,
            "timestamp": time.time(),
        }
        torch.save(payload, path)
        print(f"[UCIP-18] Checkpoint saved: {path}")


def main():
    cfg = PilotConfig()
    trainer = UCIP18PilotTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
