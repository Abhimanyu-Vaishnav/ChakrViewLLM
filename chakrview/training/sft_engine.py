"""
Stage D Supervised Fine-Tuning (SFT) Engine for ChakrView.
Supports prompt-masked causal loss, strict response-only gradients,
atomic versioned checkpointing, and bit-exact resumption on CPU.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts


@dataclass
class SFTConfig:
    repo_root: Path
    train_file: Path
    val_file: Path
    output_dir: Path
    base_checkpoint: Path
    steps: int = 50
    batch_size: int = 2
    max_seq_len: int = 512
    learning_rate: float = 5e-5
    min_learning_rate: float = 5e-6
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    val_interval: int = 10
    save_interval: int = 25
    seed: int = 42
    device: str = "cpu"
    tokenizer_dir: str = "release/chakrview-v0.5.0/data/experiments/vocab_4096"


class SFTDataset:
    """
    In-memory validated dataset loader and tensor collator for Stage D SFT.
    """

    def __init__(self, jsonl_path: Path, tokenizer: Any, max_seq_len: int = 512):
        self.jsonl_path = Path(jsonl_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.records: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"SFT dataset file not found: {self.jsonl_path}")

        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

        if not self.records:
            raise ValueError(f"SFT dataset is empty: {self.jsonl_path}")

    def __len__(self) -> int:
        return len(self.records)

    def encode_example(self, prompt: str, response: str) -> Tuple[List[int], List[int]]:
        """
        Tokenizes prompt and response, and builds causal input and label sequences
        with prompt targets masked using -100.
        Format:
          input_seq:  <BOS> + prompt_tokens + response_tokens
          target_seq: [-100]*len(prompt_tokens) + response_tokens + <EOS>
        """
        bos_id = 0
        eos_id = 1
        ignore_index = -100

        p_toks = self.tokenizer.encode(prompt.strip())
        r_toks = self.tokenizer.encode(response.strip())

        if not r_toks:
            raise ValueError(f"Response tokenized to 0 tokens for response: {response}")

        # Check total sequence length
        total_len = 1 + len(p_toks) + len(r_toks)
        if total_len > self.max_seq_len:
            # Truncate prompt from left if necessary, preserving response
            allowed_p = max(0, self.max_seq_len - len(r_toks) - 1)
            p_toks = p_toks[-allowed_p:]

        input_seq = [bos_id] + p_toks + r_toks
        # Target alignment (shifted next-token):
        # Position 0 predicts p_toks[0] -> masked
        # Position len(p_toks)-1 predicts r_toks[0] -> first response target!
        # Thus, prompt positions in target are masked: len(p_toks) masked tokens
        target_seq = ([ignore_index] * len(p_toks)) + r_toks + [eos_id]

        assert len(input_seq) == len(target_seq)
        return input_seq, target_seq

    def get_batch(
        self, batch_size: int, rng: np.random.RandomState
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Samples a batch, pads dynamically to the maximum sequence length in the batch,
        and returns input tensor, target tensor, and active response token count.
        """
        indices = rng.randint(0, len(self.records), size=batch_size)
        batch_inputs = []
        batch_targets = []

        for idx in indices:
            rec = self.records[idx]
            inp, tgt = self.encode_example(rec["prompt"], rec["response"])
            batch_inputs.append(inp)
            batch_targets.append(tgt)

        max_len = max(len(x) for x in batch_inputs)
        pad_id = 2
        ignore_index = -100

        padded_inputs = []
        padded_targets = []
        for inp, tgt in zip(batch_inputs, batch_targets):
            pad_amt = max_len - len(inp)
            padded_inputs.append(inp + [pad_id] * pad_amt)
            padded_targets.append(tgt + [ignore_index] * pad_amt)

        x_tensor = torch.tensor(padded_inputs, dtype=torch.long)
        y_tensor = torch.tensor(padded_targets, dtype=torch.long)
        active_tokens = int((y_tensor != ignore_index).sum().item())

        return x_tensor, y_tensor, active_tokens


class SFTExperimentEngine:
    """
    Autonomous Stage D supervised fine-tuning engine.
    """

    def __init__(self, config: SFTConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        self.tokenizer, self.tokenizer_meta = load_tokenizer_artifacts(config.tokenizer_dir)
        self.train_dataset = SFTDataset(config.train_file, self.tokenizer, config.max_seq_len)
        self.val_dataset = SFTDataset(config.val_file, self.tokenizer, config.max_seq_len)

        # Model and Optimizer setup
        self.device = torch.device(config.device)
        self.model = ChakrMicro(ModelConfig(max_seq_len=512)).to(self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        self.start_step = 0
        self.history: List[Dict[str, Any]] = []
        self._load_base_checkpoint()

    def _load_base_checkpoint(self):
        if not self.config.base_checkpoint.exists():
            raise FileNotFoundError(f"Base checkpoint missing: {self.config.base_checkpoint}")

        ckpt = torch.load(self.config.base_checkpoint, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=True)

        # Note: If resuming an SFT checkpoint, restore step, optimizer, and RNG states
        self.saved_numpy_rng_state = None
        self.saved_torch_rng_state = None
        if "sft_step" in ckpt:
            self.start_step = ckpt["sft_step"]
            if "optimizer_state_dict" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "history" in ckpt:
                self.history = ckpt["history"]
            if "numpy_rng_state" in ckpt:
                self.saved_numpy_rng_state = ckpt["numpy_rng_state"]
            if "rng_state" in ckpt:
                self.saved_torch_rng_state = ckpt["rng_state"]

    @torch.no_grad()
    def evaluate(self, num_batches: int = 5, seed: int = 101) -> Dict[str, float]:
        self.model.eval()
        rng = np.random.RandomState(seed)
        total_loss = 0.0
        total_active_tokens = 0

        for _ in range(num_batches):
            x, y, active_count = self.val_dataset.get_batch(batch_size=self.config.batch_size, rng=rng)
            if active_count == 0:
                continue
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            total_loss += loss.item() * active_count
            total_active_tokens += active_count

        self.model.train()
        if total_active_tokens == 0:
            return {"loss": float("nan"), "perplexity": float("nan"), "tokens": 0}

        mean_loss = total_loss / total_active_tokens
        perp = math.exp(min(20.0, mean_loss))
        return {
            "loss": round(mean_loss, 4),
            "perplexity": round(perp, 2),
            "tokens": total_active_tokens,
        }

    def train(self) -> Dict[str, Any]:
        self.model.train()
        rng = np.random.RandomState(self.config.seed)
        torch.manual_seed(self.config.seed)

        init_val = self.evaluate(seed=101 + self.start_step)

        if self.saved_numpy_rng_state is not None:
            rng.set_state(self.saved_numpy_rng_state)
        if self.saved_torch_rng_state is not None:
            torch.set_rng_state(self.saved_torch_rng_state)

        start_time = time.perf_counter()

        for step in range(self.start_step + 1, self.config.steps + 1):
            step_start = time.perf_counter()

            # Cosine learning rate schedule
            decay_ratio = (step - 1) / max(1, self.config.steps)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            cur_lr = self.config.min_learning_rate + coeff * (
                self.config.learning_rate - self.config.min_learning_rate
            )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = cur_lr

            x, y, active_tokens = self.train_dataset.get_batch(
                batch_size=self.config.batch_size, rng=rng
            )
            # Fail-closed safety validation on training batch
            if active_tokens <= 0:
                raise ValueError(f"Encountered all-masked training batch at step {step}: active_tokens={active_tokens}")
            if (x < 0).any() or (x >= self.model.config.vocab_size).any():
                raise ValueError(f"Input tokens out of bounds [0, {self.model.config.vocab_size}) at step {step}")
            active_targets = y[y != -100]
            if (active_targets < 0).any() or (active_targets >= self.model.config.vocab_size).any():
                raise ValueError(f"Target tokens out of bounds [0, {self.model.config.vocab_size}) at step {step}")

            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            if not math.isfinite(loss.item()):
                raise RuntimeError(f"Non-finite loss detected at step {step}: {loss.item()}")

            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip).item()
            self.optimizer.step()

            step_sec = time.perf_counter() - step_start

            step_record = {
                "step": step,
                "train_loss": round(loss.item(), 4),
                "active_tokens": active_tokens,
                "learning_rate": round(cur_lr, 8),
                "grad_norm": round(grad_norm, 4),
                "step_time_sec": round(step_sec, 4),
            }

            if step % self.config.val_interval == 0 or step == self.config.steps:
                val_metrics = self.evaluate(seed=101 + step)
                step_record["val_loss"] = val_metrics["loss"]
                step_record["val_perplexity"] = val_metrics["perplexity"]

            self.history.append(step_record)

            # Checkpoint save
            if step % self.config.save_interval == 0 or step == self.config.steps:
                self.save_checkpoint(step, rng=rng)

        elapsed = time.perf_counter() - start_time
        final_val = self.evaluate(seed=101 + self.config.steps)

        return {
            "initial_val_loss": init_val["loss"],
            "final_train_loss": self.history[-1]["train_loss"],
            "final_val_loss": final_val["loss"],
            "final_val_perplexity": final_val["perplexity"],
            "total_steps": self.config.steps,
            "elapsed_seconds": round(elapsed, 2),
            "history": self.history,
        }

    def save_checkpoint(self, step: int, rng: Optional[np.random.RandomState] = None):
        ckpt_filename = f"sft_checkpoint_step_{step}.pt"
        target_path = self.config.output_dir / ckpt_filename
        temp_path = self.config.output_dir / f"{ckpt_filename}.tmp"

        payload = {
            "sft_step": step,
            "seed": self.config.seed,
            "config": self.model.config,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "rng_state": torch.get_rng_state(),
            "numpy_rng_state": rng.get_state() if rng is not None else None,
            "history": self.history,
            "base_checkpoint": str(self.config.base_checkpoint),
        }

        torch.save(payload, temp_path)
        if temp_path.exists():
            temp_path.replace(target_path)
