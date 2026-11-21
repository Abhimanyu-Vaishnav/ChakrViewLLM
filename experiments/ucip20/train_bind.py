"""
UCIP-20 Isolated BIND Trainer (Five Complete Exposures).
Strictly trains ChakrMicro on the isolated BIND training dataset (bind_train_1600.jsonl)
for EXACTLY five complete passes (5 epochs).

Constraints & Invariants:
- Initialization: Original Step 800 (checkpoint_step_800.pt, SHA-256: 8759ccb4...)
- Dataset: data/sft/ucip20_bind/bind_train_1600.jsonl (1,600 records)
- Zero Replay: Stage C replay is disabled.
- Exposure: 5 complete epochs = 5 * 1,600 = 8,000 sequence exposures.
- Batch size: 16 -> Steps per epoch: 100 -> Total optimizer steps: 500.
- AdamW: lr=2.5e-5, warmup=50 steps, cosine decay to 2.5e-6, beta1=0.9, beta2=0.95, weight_decay=0.01.
- Assistant-only loss: prompt tokens masked with -100, normalized by active response tokens.
- Checkpoints saved per pass: checkpoint_pass_1.pt through checkpoint_pass_5.pt, plus checkpoint_best.pt.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.training.prompt_contract import build_causal_sequence, normalize_prompt_response_strings
from experiments.ucip20.eval.bind_evaluator import run_bind_evaluation


class BINDDataset:
    def __init__(self, jsonl_path: Path, tokenizer: Any, max_seq_len: int = 512):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.records = [json.loads(line) for line in open(jsonl_path, encoding="utf-8")]

    def __len__(self) -> int:
        return len(self.records)

    def encode_record(self, prompt: str, response: str) -> Tuple[List[int], List[int]]:
        p_str, r_str = normalize_prompt_response_strings(prompt, response, strip_delimiters=True)
        p_toks = self.tokenizer.encode(p_str)
        r_toks = self.tokenizer.encode(r_str)
        
        # Build causal sequence with assistant-only mask (-100)
        inp, tgt = build_causal_sequence(p_toks, r_toks, bos_id=0, eos_id=1, ignore_index=-100)
        return inp, tgt

    def get_batch(self, indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor, int]:
        batch_inps = []
        batch_tgts = []
        for idx in indices:
            rec = self.records[idx]
            inp, tgt = self.encode_record(rec["prompt"], rec["response"])
            batch_inps.append(inp)
            batch_tgts.append(tgt)

        max_len = max(len(x) for x in batch_inps)
        pad_id = 2
        ignore_index = -100

        padded_inps = [x + [pad_id] * (max_len - len(x)) for x in batch_inps]
        padded_tgts = [y + [ignore_index] * (max_len - len(y)) for y in batch_tgts]

        x_tensor = torch.tensor(padded_inps, dtype=torch.long)
        y_tensor = torch.tensor(padded_tgts, dtype=torch.long)
        active_tokens = int((y_tensor != ignore_index).sum().item())
        return x_tensor, y_tensor, active_tokens


@dataclass
class BINDTrainingConfig:
    output_dir: Path = ROOT / "experiments" / "ucip20" / "checkpoints"
    eval_dir: Path = ROOT / "experiments" / "ucip20" / "eval"
    base_checkpoint: Path = ROOT / "experiments" / "ucip13" / "checkpoints" / "primary_seed42" / "checkpoint_step_800.pt"
    train_file: Path = ROOT / "data" / "sft" / "ucip20_bind" / "bind_train_1600.jsonl"
    val_file: Path = ROOT / "data" / "sft" / "ucip20_bind" / "bind_val_200.jsonl"
    dev_file: Path = ROOT / "data" / "sft" / "ucip20_bind" / "bind_dev_200.jsonl"
    tokenizer_dir: Path = ROOT / "data" / "experiments" / "vocab_4096"
    num_exposures: int = 5
    batch_size: int = 16
    learning_rate: float = 2.5e-5
    min_learning_rate: float = 2.5e-6
    warmup_steps: int = 50
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "cpu"


class UCIP20Trainer:
    def __init__(self, cfg: BINDTrainingConfig):
        self.cfg = cfg
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        cfg.eval_dir.mkdir(parents=True, exist_ok=True)

        self.tokenizer, _ = load_tokenizer_artifacts(cfg.tokenizer_dir)
        self.train_ds = BINDDataset(cfg.train_file, self.tokenizer)
        self.val_ds = BINDDataset(cfg.val_file, self.tokenizer)

        # Load Base Model
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
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
        self.history: List[Dict[str, Any]] = []

    @torch.no_grad()
    def evaluate_val_loss(self) -> float:
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        all_indices = list(range(len(self.val_ds)))
        for i in range(0, len(all_indices), self.cfg.batch_size):
            batch_idx = all_indices[i : i + self.cfg.batch_size]
            x, y, active_count = self.val_ds.get_batch(batch_idx)
            if active_count == 0: continue
            x, y = x.to(self.cfg.device), y.to(self.cfg.device)
            logits = self.model(x)
            loss_sum = self.loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            total_loss += loss_sum.item()
            total_tokens += active_count
        self.model.train()
        return round(total_loss / max(1, total_tokens), 4)

    def train(self) -> Dict[str, Any]:
        N = len(self.train_ds)
        steps_per_epoch = math.ceil(N / self.cfg.batch_size)
        total_steps = steps_per_epoch * self.cfg.num_exposures

        print(f"=== UCIP-20 BIND Training ===")
        print(f"Dataset Records: {N}")
        print(f"Batch Size: {self.cfg.batch_size}")
        print(f"Steps per complete pass (epoch): {steps_per_epoch}")
        print(f"Total passes (exposures): {self.cfg.num_exposures}")
        print(f"Total optimizer steps: {total_steps}")

        init_val_loss = self.evaluate_val_loss()
        print(f"[Init] Validation Loss: {init_val_loss:.4f}")

        best_val_loss = init_val_loss
        best_ckpt_path = self.cfg.output_dir / "checkpoint_best.pt"

        global_step = 0
        total_active_tokens_processed = 0
        start_time = time.perf_counter()

        pass_metrics = []

        for pass_idx in range(1, self.cfg.num_exposures + 1):
            t_pass0 = time.perf_counter()
            # Deterministic permutation for this complete pass
            rng = np.random.RandomState(self.cfg.seed + pass_idx)
            perm = rng.permutation(N).tolist()

            pass_loss_sum = 0.0
            pass_tokens = 0

            for i in range(0, N, self.cfg.batch_size):
                global_step += 1
                batch_indices = perm[i : i + self.cfg.batch_size]
                x, y, active_count = self.train_ds.get_batch(batch_indices)
                x, y = x.to(self.cfg.device), y.to(self.cfg.device)

                # LR schedule: warmup then cosine decay
                if global_step <= self.cfg.warmup_steps:
                    cur_lr = self.cfg.learning_rate * (global_step / max(1, self.cfg.warmup_steps))
                else:
                    progress = (global_step - self.cfg.warmup_steps) / max(1, total_steps - self.cfg.warmup_steps)
                    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
                    cur_lr = self.cfg.min_learning_rate + coeff * (self.cfg.learning_rate - self.cfg.min_learning_rate)

                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = cur_lr

                self.optimizer.zero_grad()
                logits = self.model(x)
                loss_sum = self.loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                loss = loss_sum / max(1, active_count)
                loss.backward()

                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip).item()
                self.optimizer.step()

                pass_loss_sum += loss_sum.item()
                pass_tokens += active_count
                total_active_tokens_processed += active_count

                if global_step % 25 == 0 or global_step == 1:
                    print(
                        f"Pass {pass_idx}/{self.cfg.num_exposures} | "
                        f"Step {global_step:03d}/{total_steps:03d} | "
                        f"Loss: {loss.item():.4f} | Grad: {grad_norm:.3f} | LR: {cur_lr:.2e}"
                    )

            pass_mean_train_loss = pass_loss_sum / max(1, pass_tokens)
            val_loss = self.evaluate_val_loss()
            pass_time = time.perf_counter() - t_pass0

            print(
                f"--> [Pass {pass_idx} Complete] Train Loss: {pass_mean_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Time: {pass_time:.1f}s"
            )

            # Save checkpoint for this complete pass
            pass_ckpt_path = self.cfg.output_dir / f"checkpoint_pass_{pass_idx}.pt"
            self.save_checkpoint(pass_ckpt_path, global_step, pass_idx, pass_mean_train_loss, val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(best_ckpt_path, global_step, pass_idx, pass_mean_train_loss, val_loss)

            pass_metrics.append({
                "pass": pass_idx,
                "step": global_step,
                "train_loss": round(pass_mean_train_loss, 4),
                "val_loss": val_loss,
                "active_tokens": pass_tokens,
                "time_seconds": round(pass_time, 2),
            })

        total_time = time.perf_counter() - start_time
        summary = {
            "initial_val_loss": init_val_loss,
            "best_val_loss": best_val_loss,
            "final_val_loss": pass_metrics[-1]["val_loss"],
            "total_exposures": self.cfg.num_exposures,
            "total_optimizer_steps": total_steps,
            "total_active_tokens_processed": total_active_tokens_processed,
            "total_time_seconds": round(total_time, 2),
            "pass_metrics": pass_metrics,
        }
        with open(self.cfg.output_dir / "training_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary

    def save_checkpoint(self, path: Path, step: int, pass_idx: int, train_loss: float, val_loss: float):
        payload = {
            "step": step,
            "pass": pass_idx,
            "model_state_dict": copy.deepcopy(self.model.state_dict()),
            "optimizer_state_dict": copy.deepcopy(self.optimizer.state_dict()),
            "config": self.model_config.to_dict() if hasattr(self.model_config, "to_dict") else vars(self.model_config),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "timestamp": time.time(),
        }
        torch.save(payload, path)
        print(f"Checkpoint saved: {path.name}")


def main():
    cfg = BINDTrainingConfig()
    trainer = UCIP20Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
