"""
ChakrView Neural Pretraining & Reproducible Experiment Engine (v0.6).

Location: chakrview/training/pretraining_engine.py

Supports:
- Resource-bounded training runner (steps or token budget).
- Configurable batch size, seq_len, seed, device (CPU / CUDA).
- Validation tracking on fixed validation subset and held-out test evaluation.
- Full resumable checkpoints (model, optimizer, RNG, steps, metrics).
- Inference-only checkpoint exports.
- Divergence / non-finite value detection and safe rollback to last valid checkpoint.
- Text generation evaluation with repetition and diversity metrics.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts


@dataclass
class PretrainingMetrics:
    step: int
    train_loss: float
    val_loss: float
    test_loss: Optional[float]
    grad_norm: float
    learning_rate: float
    tokens_processed: int
    elapsed_sec: float
    throughput_tokens_per_sec: float


@dataclass
class PretrainingExperimentResult:
    steps: int
    batch_size: int
    seq_len: int
    device: str
    seed: int
    initial_val_loss: float
    final_train_loss: float
    final_val_loss: float
    test_loss: float
    tokens_processed: int
    elapsed_sec: float
    avg_tokens_per_sec: float
    checkpoint_path: Optional[str]
    checkpoint_sha256: Optional[str]
    inference_checkpoint_path: Optional[str]
    generated_samples: Dict[str, str]
    repetition_ratio: float
    unique_tokens_ratio: float
    history: List[Dict[str, Any]]


class PretrainingExperimentEngine:
    """
    Robust, reproducible pretraining engine for ChakrMicro language modeling.
    """

    @classmethod
    def load_dataset_tokens(
        cls, repo_root: Path, max_train_tokens: int = 100000
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads bounded token arrays from Stage C binary shards.
        """
        stage_c = repo_root / "data" / "tokenized" / "stage_c"
        train_bin = stage_c / "train" / "shard_00000.bin"
        val_bin = stage_c / "validation" / "shard_00000.bin"
        test_bin = stage_c / "test" / "shard_00000.bin"

        with open(train_bin, "rb") as f:
            train_tokens = np.frombuffer(f.read(max_train_tokens * 2), dtype=np.uint16).astype(np.int64)
        with open(val_bin, "rb") as f:
            val_tokens = np.frombuffer(f.read(20000 * 2), dtype=np.uint16).astype(np.int64)
        with open(test_bin, "rb") as f:
            test_tokens = np.frombuffer(f.read(20000 * 2), dtype=np.uint16).astype(np.int64)

        return train_tokens, val_tokens, test_tokens

    @classmethod
    def get_batch(
        cls,
        tokens: np.ndarray,
        batch_size: int = 8,
        seq_len: int = 64,
        rng: Optional[np.random.RandomState] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if rng is None:
            rng = np.random.RandomState(42)
        max_idx = len(tokens) - seq_len - 1
        idxs = rng.randint(0, max_idx, size=batch_size)
        x = torch.stack([torch.tensor(tokens[i : i + seq_len], dtype=torch.long) for i in idxs])
        y = torch.stack([torch.tensor(tokens[i + 1 : i + seq_len + 1], dtype=torch.long) for i in idxs])
        return x, y

    @classmethod
    def run_training(
        cls,
        repo_root: Path,
        steps: int = 30,
        batch_size: int = 8,
        seq_len: int = 64,
        lr: float = 5e-4,
        seed: int = 42,
        device: str = "cpu",
        val_interval: int = 10,
        grad_clip: float = 1.0,
        checkpoint_dir: Optional[Path] = None,
        resume_from: Optional[Path] = None,
        prompts: Optional[List[str]] = None,
    ) -> PretrainingExperimentResult:
        start_time = time.perf_counter()
        torch.manual_seed(seed)
        rng = np.random.RandomState(seed)

        target_device = torch.device(device)
        train_tokens, val_tokens, test_tokens = cls.load_dataset_tokens(repo_root)

        config = ModelConfig(max_seq_len=512)
        model = ChakrMicro(config).to(target_device)
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        loss_fn = nn.CrossEntropyLoss()

        # Fixed validation and test sets
        vx, vy = cls.get_batch(val_tokens, batch_size=16, seq_len=seq_len, rng=np.random.RandomState(101))
        vx, vy = vx.to(target_device), vy.to(target_device)

        tex, tey = cls.get_batch(test_tokens, batch_size=16, seq_len=seq_len, rng=np.random.RandomState(202))
        tex, tey = tex.to(target_device), tey.to(target_device)

        start_step = 0
        history: List[Dict[str, Any]] = []

        if resume_from is not None and resume_from.exists():
            checkpoint_data = torch.load(resume_from, map_location=target_device, weights_only=False)
            model.load_state_dict(checkpoint_data["model_state_dict"])
            if "optimizer_state_dict" in checkpoint_data:
                opt.load_state_dict(checkpoint_data["optimizer_state_dict"])
            start_step = checkpoint_data.get("step", 0)
            if "rng_state" in checkpoint_data:
                torch.set_rng_state(checkpoint_data["rng_state"])
            if "history" in checkpoint_data:
                history = checkpoint_data["history"]

        # Evaluate initial validation
        model.eval()
        with torch.no_grad():
            init_val_logits = model(vx)
            init_val_loss = loss_fn(init_val_logits.reshape(-1, 4096), vy.reshape(-1)).item()
        model.train()

        last_train_loss = init_val_loss
        best_val_loss = float("inf")
        best_model_state = None
        last_valid_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        for step in range(start_step, steps):
            step_start = time.perf_counter()
            x, y = cls.get_batch(train_tokens, batch_size=batch_size, seq_len=seq_len, rng=rng)
            x, y = x.to(target_device), y.to(target_device)

            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, 4096), y.reshape(-1))

            if not torch.isfinite(loss):
                # Safe recovery to last valid state
                model.load_state_dict(last_valid_state)
                raise FloatingPointError(f"Non-finite loss encountered at step {step}: {loss.item()}. Rolled back.")

            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip).item()

            if not math.isfinite(grad_norm):
                model.load_state_dict(last_valid_state)
                raise FloatingPointError(f"Non-finite grad norm encountered at step {step}: {grad_norm}. Rolled back.")

            opt.step()
            last_train_loss = loss.item()
            last_valid_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            step_elapsed = time.perf_counter() - step_start
            step_tokens = batch_size * seq_len
            tps = step_tokens / max(step_elapsed, 1e-6)

            is_val_step = ((step + 1) % val_interval == 0) or (step == steps - 1)
            val_loss = None

            if is_val_step:
                model.eval()
                with torch.no_grad():
                    val_logits = model(vx)
                    val_loss = loss_fn(val_logits.reshape(-1, 4096), vy.reshape(-1)).item()
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                model.train()

            history.append({
                "step": step + 1,
                "train_loss": round(last_train_loss, 4),
                "val_loss": round(val_loss, 4) if val_loss is not None else None,
                "grad_norm": round(grad_norm, 4),
                "tps": round(tps, 2),
            })

        # Restore best validation checkpoint if found
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Final evaluation
        model.eval()
        with torch.no_grad():
            final_val_loss = loss_fn(model(vx).reshape(-1, 4096), vy.reshape(-1)).item()
            final_test_loss = loss_fn(model(tex).reshape(-1, 4096), tey.reshape(-1)).item()

        total_elapsed = time.perf_counter() - start_time
        total_tokens = (steps - start_step) * batch_size * seq_len
        avg_tps = total_tokens / max(total_elapsed, 1e-6)

        # Checkpoints
        ckpt_full_path: Optional[str] = None
        ckpt_infer_path: Optional[str] = None
        ckpt_hash: Optional[str] = None

        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            # Full training resumption checkpoint
            full_ckpt_file = checkpoint_dir / f"training_checkpoint_step_{steps}.pt"
            torch.save({
                "step": steps,
                "seed": seed,
                "config": config,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "rng_state": torch.get_rng_state(),
                "history": history,
                "metrics": {
                    "init_val_loss": init_val_loss,
                    "final_train_loss": last_train_loss,
                    "final_val_loss": final_val_loss,
                    "test_loss": final_test_loss,
                }
            }, full_ckpt_file)
            ckpt_full_path = str(full_ckpt_file)
            ckpt_hash = hashlib.sha256(full_ckpt_file.read_bytes()).hexdigest()

            # Inference-only checkpoint
            infer_ckpt_file = checkpoint_dir / f"inference_model_step_{steps}.pt"
            torch.save(model.state_dict(), infer_ckpt_file)
            ckpt_infer_path = str(infer_ckpt_file)

        # Sample generations
        if prompts is None:
            prompts = ["August is", "The system", "Language learning", "ChakrView native"]

        tok_dir = repo_root / "data" / "experiments" / "vocab_4096"
        tokenizer, _ = load_tokenizer_artifacts(tok_dir)
        samples: Dict[str, str] = {}
        all_gen_token_ids: List[int] = []

        model.eval()
        with torch.no_grad():
            for p in prompts:
                input_ids = torch.tensor([tokenizer.encode(p)], device=target_device)
                for _ in range(16):
                    l = model(input_ids)[:, -1, :]
                    nxt = l.argmax(dim=-1, keepdim=True)
                    input_ids = torch.cat([input_ids, nxt], dim=1)
                tokens_out = input_ids[0].tolist()
                samples[p] = tokenizer.decode(tokens_out)
                all_gen_token_ids.extend(tokens_out[len(tokenizer.encode(p)):])

        # Repetition and diversity diagnostics
        if all_gen_token_ids:
            unique_ratio = len(set(all_gen_token_ids)) / len(all_gen_token_ids)
            # Count consecutive duplicates
            dups = sum(1 for i in range(len(all_gen_token_ids) - 1) if all_gen_token_ids[i] == all_gen_token_ids[i+1])
            rep_ratio = dups / max(len(all_gen_token_ids) - 1, 1)
        else:
            unique_ratio = 1.0
            rep_ratio = 0.0

        return PretrainingExperimentResult(
            steps=steps,
            batch_size=batch_size,
            seq_len=seq_len,
            device=str(target_device),
            seed=seed,
            initial_val_loss=round(init_val_loss, 4),
            final_train_loss=round(last_train_loss, 4),
            final_val_loss=round(final_val_loss, 4),
            test_loss=round(final_test_loss, 4),
            tokens_processed=total_tokens,
            elapsed_sec=round(total_elapsed, 2),
            avg_tokens_per_sec=round(avg_tps, 2),
            checkpoint_path=ckpt_full_path,
            checkpoint_sha256=ckpt_hash,
            inference_checkpoint_path=ckpt_infer_path,
            generated_samples=samples,
            repetition_ratio=round(rep_ratio, 4),
            unique_tokens_ratio=round(unique_ratio, 4),
            history=history,
        )
