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


class MultiShardDataset:
    """
    Memory-mapped multi-shard dataset reader for Stage C binary shards.
    Preserves bounded memory consumption (uint16 memmap per shard)
    while providing contiguous indexable access across all discovered shards.
    """
    def __init__(self, shard_paths: List[Path], expected_dtype: np.dtype = np.uint16, vocab_size: int = 4096):
        if not shard_paths:
            raise ValueError("No shard paths provided to MultiShardDataset.")
        self.shard_paths = [Path(p) for p in shard_paths]
        self.expected_dtype = expected_dtype
        self.vocab_size = vocab_size

        self.shards: List[np.memmap] = []
        self.shard_lengths: List[int] = []
        self.cumulative_lengths: List[int] = []
        total = 0

        for sp in self.shard_paths:
            if not sp.is_file():
                raise FileNotFoundError(f"Shard file missing: {sp}")
            size_bytes = sp.stat().st_size
            if size_bytes == 0:
                raise ValueError(f"Shard file is empty (0 bytes): {sp}")
            itemsize = np.dtype(expected_dtype).itemsize
            if size_bytes % itemsize != 0:
                raise ValueError(f"Shard file size {size_bytes} is not a multiple of itemsize {itemsize}: {sp}")

            token_count = size_bytes // itemsize
            mmap = np.memmap(sp, dtype=expected_dtype, mode="r")
            self.shards.append(mmap)
            self.shard_lengths.append(token_count)
            total += token_count
            self.cumulative_lengths.append(total)

        self.total_tokens = total

    def __len__(self) -> int:
        return self.total_tokens

    def get_slice(self, start_idx: int, length: int) -> np.ndarray:
        """
        Retrieves a contiguous slice of tokens starting at start_idx with length.
        Handles shard boundaries cleanly without loading whole shards into RAM.
        """
        if start_idx < 0 or start_idx + length > self.total_tokens:
            raise IndexError(f"Slice range [{start_idx}:{start_idx + length}] out of bounds for dataset of length {self.total_tokens}")

        # Find starting shard
        shard_idx = np.searchsorted(self.cumulative_lengths, start_idx, side="right")
        shard_start_offset = start_idx - (self.cumulative_lengths[shard_idx - 1] if shard_idx > 0 else 0)

        # Check if slice fits entirely inside this shard
        tokens_in_shard = self.shard_lengths[shard_idx] - shard_start_offset
        if tokens_in_shard >= length:
            return np.array(self.shards[shard_idx][shard_start_offset : shard_start_offset + length], dtype=np.int64)

        # Crosses shard boundary: assemble across shards
        chunks: List[np.ndarray] = []
        rem = length
        curr_shard = shard_idx
        curr_offset = shard_start_offset

        while rem > 0 and curr_shard < len(self.shards):
            avail = self.shard_lengths[curr_shard] - curr_offset
            take = min(rem, avail)
            chunks.append(np.array(self.shards[curr_shard][curr_offset : curr_offset + take], dtype=np.int64))
            rem -= take
            curr_shard += 1
            curr_offset = 0

        return np.concatenate(chunks)

    def close(self) -> None:
        """Explicitly close memmaps to release Windows file locks."""
        for m in self.shards:
            if hasattr(m, "_mmap") and m._mmap is not None:
                m._mmap.close()
        self.shards.clear()

    def __enter__(self) -> MultiShardDataset:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def get_batch(
        self,
        batch_size: int = 8,
        seq_len: int = 64,
        rng: Optional[np.random.RandomState] = None,
        sampling_mode: str = "natural",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if rng is None:
            rng = np.random.RandomState(42)
        max_idx = self.total_tokens - seq_len - 1
        if max_idx <= 0:
            raise ValueError(f"Dataset total tokens ({self.total_tokens}) too small for seq_len ({seq_len})")

        # Balanced sampling across Indic/Hindi partition (shards 9-19) and English/Code partition (shards 0-8, 20-26)
        # Shards 9-19 span cumulative offsets approximately ~2,250,000 to ~5,000,000 in a 27-shard corpus.
        if sampling_mode == "balanced" and len(self.shards) >= 20:
            indic_start = self.cumulative_lengths[8] if len(self.cumulative_lengths) > 8 else 0
            indic_end = self.cumulative_lengths[19] if len(self.cumulative_lengths) > 19 else self.total_tokens
            indic_max = max(indic_end - seq_len - 1, indic_start + 1)
            other_ranges = []
            if indic_start > seq_len + 1:
                other_ranges.append((0, indic_start - seq_len - 1))
            if indic_end < self.total_tokens - seq_len - 1:
                other_ranges.append((indic_end, self.total_tokens - seq_len - 1))

            idxs = []
            for b in range(batch_size):
                # 50% chance Indic/Hindi partition, 50% chance English/Code partition
                if b % 2 == 0 and indic_max > indic_start:
                    idxs.append(rng.randint(indic_start, indic_max))
                else:
                    if other_ranges:
                        # pick from other partition
                        r_start, r_end = other_ranges[rng.randint(0, len(other_ranges))]
                        idxs.append(rng.randint(r_start, max(r_end, r_start + 1)))
                    else:
                        idxs.append(rng.randint(0, max_idx))
        else:
            # Natural uniform sampling
            idxs = [rng.randint(0, max_idx) for _ in range(batch_size)]

        x_list = []
        y_list = []
        for i in idxs:
            tokens = self.get_slice(int(i), seq_len + 1)
            x_list.append(torch.tensor(tokens[:seq_len], dtype=torch.long))
            y_list.append(torch.tensor(tokens[1 : seq_len + 1], dtype=torch.long))
        return torch.stack(x_list), torch.stack(y_list)


class PretrainingExperimentEngine:
    """
    Robust, reproducible pretraining engine for ChakrMicro language modeling.
    Supports multi-shard datasets, memory mapping, and bit-exact resumption.
    """

    @classmethod
    def load_dataset_tokens(
        cls, repo_root: Path, max_train_tokens: int = 100000
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads bounded token arrays from Stage C binary shards (backward-compatible API).
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
    def discover_split_shards(cls, repo_root: Path, split: str = "train") -> List[Path]:
        """
        Discovers all shards for a split using metadata.json if present, falling back to glob.
        Validates shard existence and ensures deterministic alphabetical order.
        """
        split_dir = repo_root / "data" / "tokenized" / "stage_c" / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory does not exist: {split_dir}")

        meta_path = split_dir / "metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                shards_meta = meta.get("shards", [])
                if shards_meta:
                    paths = [split_dir / s["filename"] for s in shards_meta]
                    for p in paths:
                        if not p.is_file():
                            raise FileNotFoundError(f"Shard listed in metadata missing on disk: {p}")
                    return sorted(paths, key=lambda x: x.name)
            except Exception as e:
                # If metadata read fails, re-raise to fail closed
                if isinstance(e, FileNotFoundError):
                    raise
                pass

        # Fallback to sorted glob
        paths = sorted(list(split_dir.glob("shard_*.bin")), key=lambda x: x.name)
        if not paths:
            raise FileNotFoundError(f"No shard_*.bin files found in {split_dir}")
        return paths

    @classmethod
    def create_multi_shard_dataset(cls, repo_root: Path, split: str = "train") -> MultiShardDataset:
        """
        Creates a MultiShardDataset for the given split (train, validation, test).
        """
        shard_paths = cls.discover_split_shards(repo_root, split)
        return MultiShardDataset(shard_paths)

    @classmethod
    def evaluate_split(
        cls,
        model: nn.Module,
        dataset: Any,
        batch_size: int = 16,
        seq_len: int = 64,
        max_batches: int = 5,
        seed: int = 101,
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, float]:
        """
        Deterministic, representative multi-batch evaluator for validation/test splits.
        Computes weighted loss and perplexity across multiple fixed batches.
        """
        was_training = model.training
        model.eval()
        loss_fn = nn.CrossEntropyLoss()
        rng = np.random.RandomState(seed)

        total_loss = 0.0
        total_tokens = 0
        batches_evaluated = 0

        with torch.no_grad():
            for _ in range(max_batches):
                if hasattr(dataset, "get_batch"):
                    x, y = dataset.get_batch(batch_size=batch_size, seq_len=seq_len, rng=rng)
                else:
                    x, y = cls.get_batch(dataset, batch_size=batch_size, seq_len=seq_len, rng=rng)
                x, y = x.to(device), y.to(device)

                logits = model(x)
                loss = loss_fn(logits.reshape(-1, 4096), y.reshape(-1))
                tokens_in_batch = x.numel()

                total_loss += loss.item() * tokens_in_batch
                total_tokens += tokens_in_batch
                batches_evaluated += 1

        if was_training:
            model.train()

        if total_tokens == 0:
            return {"loss": float("nan"), "perplexity": float("nan"), "tokens_evaluated": 0, "batches_evaluated": 0}

        avg_loss = total_loss / total_tokens
        perp = math.exp(min(20.0, avg_loss)) if not math.isnan(avg_loss) else float("nan")

        return {
            "loss": round(avg_loss, 4),
            "perplexity": round(perp, 2),
            "tokens_evaluated": total_tokens,
            "batches_evaluated": batches_evaluated,
        }

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
        use_multi_shard: bool = True,
        lr_schedule: str = "constant",
        min_lr: float = 5e-5,
        warmup_steps: int = 100,
        sampling_mode: str = "natural",
    ) -> PretrainingExperimentResult:
        start_time = time.perf_counter()
        torch.manual_seed(seed)
        rng = np.random.RandomState(seed)

        target_device = torch.device(device)

        if use_multi_shard:
            train_dataset = cls.create_multi_shard_dataset(repo_root, split="train")
            val_dataset = cls.create_multi_shard_dataset(repo_root, split="validation")
            test_dataset = cls.create_multi_shard_dataset(repo_root, split="test")

            # Deterministic fixed validation and test sets
            vx, vy = val_dataset.get_batch(batch_size=16, seq_len=seq_len, rng=np.random.RandomState(101))
            tex, tey = test_dataset.get_batch(batch_size=16, seq_len=seq_len, rng=np.random.RandomState(202))
        else:
            train_tokens, val_tokens, test_tokens = cls.load_dataset_tokens(repo_root)
            vx, vy = cls.get_batch(val_tokens, batch_size=16, seq_len=seq_len, rng=np.random.RandomState(101))
            tex, tey = cls.get_batch(test_tokens, batch_size=16, seq_len=seq_len, rng=np.random.RandomState(202))

        vx, vy = vx.to(target_device), vy.to(target_device)
        tex, tey = tex.to(target_device), tey.to(target_device)

        config = ModelConfig(max_seq_len=512)
        model = ChakrMicro(config).to(target_device)
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        loss_fn = nn.CrossEntropyLoss()

        start_step = 0
        history: List[Dict[str, Any]] = []

        if resume_from is not None and resume_from.exists():
            checkpoint_data = torch.load(resume_from, map_location=target_device, weights_only=False)
            model.load_state_dict(checkpoint_data["model_state_dict"])
            if "optimizer_state_dict" in checkpoint_data and checkpoint_data["optimizer_state_dict"] and "param_groups" in checkpoint_data["optimizer_state_dict"]:
                opt.load_state_dict(checkpoint_data["optimizer_state_dict"])
            start_step = checkpoint_data.get("step", 0)
            if "rng_state" in checkpoint_data:
                torch.set_rng_state(checkpoint_data["rng_state"])
            if "cuda_rng_state" in checkpoint_data and torch.cuda.is_available() and checkpoint_data["cuda_rng_state"] is not None:
                torch.cuda.set_rng_state(checkpoint_data["cuda_rng_state"])
            if "numpy_rng_state" in checkpoint_data:
                rng.set_state(checkpoint_data["numpy_rng_state"])
            if "history" in checkpoint_data:
                history = checkpoint_data["history"]

        # Evaluate initial validation using multi-batch representative evaluator
        val_eval_ds = val_dataset if use_multi_shard else val_tokens
        init_val_metrics = cls.evaluate_split(model, val_eval_ds, batch_size=batch_size, seq_len=seq_len, max_batches=3, seed=101, device=target_device)
        init_val_loss = init_val_metrics["loss"]

        last_train_loss = init_val_loss
        best_val_loss = float("inf")
        best_model_state = None
        last_valid_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        for step in range(start_step, steps):
            step_start = time.perf_counter()
            if use_multi_shard:
                x, y = train_dataset.get_batch(batch_size=batch_size, seq_len=seq_len, rng=rng, sampling_mode=sampling_mode)
            else:
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

            # Learning rate schedule computation
            if lr_schedule == "cosine":
                cur_step = step - start_step
                total_sched_steps = max(steps - start_step, 1)
                if cur_step < warmup_steps:
                    curr_lr = lr * float(cur_step + 1) / float(max(1, warmup_steps))
                else:
                    progress = float(cur_step - warmup_steps) / float(max(1, total_sched_steps - warmup_steps))
                    curr_lr = min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))
                for param_group in opt.param_groups:
                    param_group["lr"] = curr_lr
            else:
                curr_lr = lr

            opt.step()
            last_train_loss = loss.item()
            last_valid_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            step_elapsed = time.perf_counter() - step_start
            step_tokens = batch_size * seq_len
            tps = step_tokens / max(step_elapsed, 1e-6)

            is_val_step = ((step + 1) % val_interval == 0) or (step == steps - 1)
            val_loss = None
            val_perplexity = None

            if is_val_step:
                val_eval_ds = val_dataset if use_multi_shard else val_tokens
                v_metrics = cls.evaluate_split(model, val_eval_ds, batch_size=batch_size, seq_len=seq_len, max_batches=3, seed=101, device=target_device)
                val_loss = v_metrics["loss"]
                val_perplexity = v_metrics["perplexity"]
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            history.append({
                "step": step + 1,
                "train_loss": round(last_train_loss, 4),
                "val_loss": round(val_loss, 4) if val_loss is not None else None,
                "val_perplexity": round(val_perplexity, 2) if val_perplexity is not None else None,
                "grad_norm": round(grad_norm, 4),
                "learning_rate": round(curr_lr, 6),
                "tps": round(tps, 2),
            })

        # Restore best validation checkpoint if found
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Final evaluation across splits using multi-batch representative evaluator
        val_eval_ds = val_dataset if use_multi_shard else val_tokens
        test_eval_ds = test_dataset if use_multi_shard else test_tokens

        val_metrics = cls.evaluate_split(model, val_eval_ds, batch_size=batch_size, seq_len=seq_len, max_batches=3, seed=101, device=target_device)
        test_metrics = cls.evaluate_split(model, test_eval_ds, batch_size=batch_size, seq_len=seq_len, max_batches=3, seed=202, device=target_device)

        final_val_loss = val_metrics["loss"]
        final_test_loss = test_metrics["loss"]

        total_elapsed = time.perf_counter() - start_time
        total_tokens = (steps - start_step) * batch_size * seq_len
        avg_tps = total_tokens / max(total_elapsed, 1e-6)

        # Checkpoints
        ckpt_full_path: Optional[str] = None
        ckpt_infer_path: Optional[str] = None
        ckpt_hash: Optional[str] = None

        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            # Full training resumption checkpoint with complete RNG states (atomic write)
            full_ckpt_file = checkpoint_dir / f"training_checkpoint_step_{steps}.pt"
            tmp_full_file = checkpoint_dir / f"training_checkpoint_step_{steps}.pt.tmp"
            torch.save({
                "step": steps,
                "seed": seed,
                "config": config,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                "numpy_rng_state": rng.get_state(),
                "use_multi_shard": use_multi_shard,
                "history": history,
                "metrics": {
                    "init_val_loss": init_val_loss,
                    "final_train_loss": last_train_loss,
                    "final_val_loss": final_val_loss,
                    "test_loss": final_test_loss,
                }
            }, tmp_full_file)
            tmp_full_file.replace(full_ckpt_file)
            ckpt_full_path = str(full_ckpt_file)
            ckpt_hash = hashlib.sha256(full_ckpt_file.read_bytes()).hexdigest()

            # Inference-only checkpoint (atomic write)
            infer_ckpt_file = checkpoint_dir / f"inference_model_step_{steps}.pt"
            tmp_infer_file = checkpoint_dir / f"inference_model_step_{steps}.pt.tmp"
            torch.save(model.state_dict(), tmp_infer_file)
            tmp_infer_file.replace(infer_ckpt_file)
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
