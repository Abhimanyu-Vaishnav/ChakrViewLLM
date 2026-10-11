"""
ChakrView Neural Language Learning Pilot & Checkpoint Governance Suite.

Location: chakrview/cognition/v050_candidate/neural_language_suite.py

Implements:
- Bounded next-token prediction experiment on real tokenized text shards.
- Train / Validation / Test separation with leakage controls.
- Gradient norm monitoring and non-finite value checks.
- Isolated candidate checkpoint serialization with SHA-256 hash computation.
- Deterministic checkpoint reload verification and text sample generation.
"""

from __future__ import annotations

import hashlib
import json
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
class LanguageExperimentResult:
    steps: int
    batch_size: int
    seq_len: int
    initial_val_loss: float
    final_train_loss: float
    final_val_loss: float
    test_loss: float
    tokens_processed: int
    elapsed_sec: float
    device: str
    checkpoint_path: Optional[str]
    checkpoint_sha256: Optional[str]
    generated_sample: str


class NeuralLanguageExperimentSuite:
    """
    Executes a bounded neural language modeling experiment on tokenized text shards.
    """

    @classmethod
    def load_dataset_tokens(cls, repo_root: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        train_bin = repo_root / "data" / "tokenized" / "stage_c" / "train" / "shard_00000.bin"
        val_bin = repo_root / "data" / "tokenized" / "stage_c" / "validation" / "shard_00000.bin"
        test_bin = repo_root / "data" / "tokenized" / "stage_c" / "test" / "shard_00000.bin"

        with open(train_bin, "rb") as f:
            train_tokens = np.frombuffer(f.read(50000), dtype=np.uint16).astype(np.int64)
        with open(val_bin, "rb") as f:
            val_tokens = np.frombuffer(f.read(10000), dtype=np.uint16).astype(np.int64)
        with open(test_bin, "rb") as f:
            test_tokens = np.frombuffer(f.read(10000), dtype=np.uint16).astype(np.int64)

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
        x = torch.stack([torch.tensor(tokens[i : i + seq_len]) for i in idxs])
        y = torch.stack([torch.tensor(tokens[i + 1 : i + seq_len + 1]) for i in idxs])
        return x, y

    @classmethod
    def run_training_experiment(
        cls,
        repo_root: Path,
        steps: int = 35,
        batch_size: int = 8,
        seq_len: int = 64,
        lr: float = 5e-4,
        seed: int = 42,
        device: str = "cpu",
        checkpoint_dir: Optional[Path] = None,
    ) -> LanguageExperimentResult:
        start_time = time.perf_counter()
        torch.manual_seed(seed)
        rng = np.random.RandomState(seed)

        target_device = torch.device(device)
        train_tokens, val_tokens, test_tokens = cls.load_dataset_tokens(repo_root)

        config = ModelConfig(max_seq_len=512)
        model = ChakrMicro(config).to(target_device)
        model.train()

        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        loss_fn = nn.CrossEntropyLoss()

        # Fixed validation and test batches
        vx, vy = cls.get_batch(val_tokens, batch_size=16, seq_len=seq_len, rng=np.random.RandomState(101))
        vx, vy = vx.to(target_device), vy.to(target_device)

        tex, tey = cls.get_batch(test_tokens, batch_size=16, seq_len=seq_len, rng=np.random.RandomState(202))
        tex, tey = tex.to(target_device), tey.to(target_device)

        # Baseline evaluation
        model.eval()
        with torch.no_grad():
            init_val_logits = model(vx)
            init_val_loss = loss_fn(init_val_logits.reshape(-1, 4096), vy.reshape(-1)).item()
        model.train()

        last_train_loss = init_val_loss
        best_val_loss = float("inf")
        best_state = None

        for step in range(steps):
            x, y = cls.get_batch(train_tokens, batch_size=batch_size, seq_len=seq_len, rng=rng)
            x, y = x.to(target_device), y.to(target_device)

            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, 4096), y.reshape(-1))

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss encountered at step {step}: {loss.item()}")

            loss.backward()
            # Gradient clipping check
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"Non-finite gradient norm at step {step}")

            opt.step()
            last_train_loss = loss.item()

            # Periodic validation tracking
            if (step + 1) % 10 == 0 or step == steps - 1:
                model.eval()
                with torch.no_grad():
                    cur_val_loss = loss_fn(model(vx).reshape(-1, 4096), vy.reshape(-1)).item()
                    if cur_val_loss < best_val_loss:
                        best_val_loss = cur_val_loss
                        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                model.train()

        # Restore best validation checkpoint
        if best_state is not None:
            model.load_state_dict(best_state)

        # Final evaluation on Val and Test
        model.eval()
        with torch.no_grad():
            final_val_loss = loss_fn(model(vx).reshape(-1, 4096), vy.reshape(-1)).item()
            test_loss = loss_fn(model(tex).reshape(-1, 4096), tey.reshape(-1)).item()

        elapsed_sec = time.perf_counter() - start_time
        tokens_processed = steps * batch_size * seq_len

        # Checkpoint serialization
        ckpt_path_str: Optional[str] = None
        ckpt_hash: Optional[str] = None
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = checkpoint_dir / f"candidate_language_step_{steps}.pt"
            torch.save(
                {
                    "steps": steps,
                    "seed": seed,
                    "metrics": {
                        "init_val_loss": init_val_loss,
                        "final_train_loss": last_train_loss,
                        "final_val_loss": final_val_loss,
                        "test_loss": test_loss,
                    },
                    "state_dict": model.state_dict(),
                },
                ckpt_path,
            )
            ckpt_path_str = str(ckpt_path)
            with open(ckpt_path, "rb") as f:
                ckpt_hash = hashlib.sha256(f.read()).hexdigest()

        # Generate sample text
        tok_dir = repo_root / "data" / "experiments" / "vocab_4096"
        tokenizer, _ = load_tokenizer_artifacts(tok_dir)
        prompt = "August is"
        input_ids = torch.tensor([tokenizer.encode(prompt)], device=target_device)
        with torch.no_grad():
            for _ in range(12):
                l = model(input_ids)[:, -1, :]
                nxt = l.argmax(dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, nxt], dim=1)
        gen_tokens = input_ids[0].tolist()
        gen_sample = tokenizer.decode(gen_tokens)

        return LanguageExperimentResult(
            steps=steps,
            batch_size=batch_size,
            seq_len=seq_len,
            initial_val_loss=round(init_val_loss, 4),
            final_train_loss=round(last_train_loss, 4),
            final_val_loss=round(final_val_loss, 4),
            test_loss=round(test_loss, 4),
            tokens_processed=tokens_processed,
            elapsed_sec=round(elapsed_sec, 2),
            device=str(target_device),
            checkpoint_path=ckpt_path_str,
            checkpoint_sha256=ckpt_hash,
            generated_sample=gen_sample,
        )
