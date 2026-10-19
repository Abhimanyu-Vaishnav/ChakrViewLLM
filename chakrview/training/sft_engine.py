"""
Stage D Supervised Fine-Tuning (SFT) Engine for ChakrView.
Supports prompt-masked causal loss, strict response-only gradients,
atomic versioned checkpointing, and bit-exact resumption on CPU.
"""

from __future__ import annotations

import hashlib
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
from chakrview.training.prompt_contract import (
    normalize_prompt_response_strings,
    build_causal_sequence,
)


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
    allow_exploratory_opt_in: bool = False
    enforce_trust_gate: bool = True
    strip_prompt_delimiters: bool = True
    save_best_checkpoint: bool = True


class SFTDataset:
    """
    In-memory validated dataset loader and tensor collator for Stage D SFT.
    Enforces fail-closed trust gate: records marked PENDING_HUMAN_REVIEW or HELD_OUT_BENCHMARK
    cannot enter gradient updates unless explicit exploratory opt-in is configured.
    """

    def __init__(
        self,
        jsonl_path: Path,
        tokenizer: Any,
        max_seq_len: int = 512,
        allow_exploratory_opt_in: bool = False,
        enforce_trust_gate: bool = True,
        is_training: bool = False,
        strip_prompt_delimiters: bool = True,
    ):
        self.jsonl_path = Path(jsonl_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.allow_exploratory_opt_in = allow_exploratory_opt_in
        self.enforce_trust_gate = enforce_trust_gate
        self.is_training = is_training
        self.strip_prompt_delimiters = strip_prompt_delimiters
        self.records: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"SFT dataset file not found: {self.jsonl_path}")

        with open(self.jsonl_path, "rb") as f:
            self.file_sha256 = hashlib.sha256(f.read()).hexdigest()

        raw_records = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_records.append(json.loads(line))

        if not raw_records:
            raise ValueError(f"SFT dataset is empty: {self.jsonl_path}")

        for rec in raw_records:
            rid = rec.get("id", "UNKNOWN_RECORD")
            role = rec.get("data_role", "TRAIN" if self.is_training else "VALIDATION")

            # Check benchmark leakage into training
            if self.is_training and role in ("HELD_OUT_BENCHMARK", "BENCHMARK"):
                raise ValueError(
                    f"Contamination error: Record {rid} with role {role} cannot enter training dataset."
                )

            # Enforce trust gate
            if self.enforce_trust_gate:
                review_status = rec.get("review_status")
                status = rec.get("status", "")
                if review_status == "PENDING_HUMAN_REVIEW" and not self.allow_exploratory_opt_in:
                    raise PermissionError(
                        f"Trust Gate Refusal: Record {rid} is PENDING_HUMAN_REVIEW. "
                        f"Unreviewed records cannot enter training without explicit allow_exploratory_opt_in=True."
                    )
                if status == "REJECTED":
                    raise ValueError(f"Trust Gate Refusal: Record {rid} is marked REJECTED.")

            self.records.append(rec)

    def __len__(self) -> int:
        return len(self.records)

    def encode_example(
        self, prompt: str, response: str, append_eos: bool = True
    ) -> Tuple[List[int], List[int]]:
        """
        Tokenizes prompt and response, and builds causal input and label sequences
        with prompt targets masked using -100.
        Format:
          Instruction: input_seq:  <BOS> + prompt_tokens + response_tokens
                       target_seq: [-100]*len(prompt_tokens) + response_tokens + <EOS>
          Replay:      input_seq:  <BOS> + prompt_tokens + response_tokens[:-1]
                       target_seq: [-100]*len(prompt_tokens) + response_tokens
        """
        bos_id = 0
        eos_id = 1
        ignore_index = -100

        p_str, r_str = normalize_prompt_response_strings(
            prompt, response, strip_delimiters=self.strip_prompt_delimiters
        )

        p_toks = self.tokenizer.encode(p_str)
        r_toks = self.tokenizer.encode(r_str)

        if not r_toks:
            raise ValueError(f"Response tokenized to 0 tokens for response: {response}")

        # Check total sequence length
        if 1 + len(r_toks) > self.max_seq_len:
            raise ValueError(
                f"Response tokens ({len(r_toks)}) exceed max_seq_len ({self.max_seq_len})."
            )

        total_len = 1 + len(p_toks) + len(r_toks)
        if total_len > self.max_seq_len:
            # Truncate prompt from left if necessary, preserving response
            allowed_p = max(0, self.max_seq_len - len(r_toks) - 1)
            p_toks = p_toks[-allowed_p:]

        input_seq, target_seq = build_causal_sequence(
            p_toks,
            r_toks,
            bos_id=bos_id,
            eos_id=eos_id,
            ignore_index=ignore_index,
            append_eos=append_eos,
        )
        if len(input_seq) > self.max_seq_len:
            raise ValueError(
                f"Encoded sequence length {len(input_seq)} exceeds max_seq_len {self.max_seq_len}."
            )

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
            append_eos = rec.get("append_eos", True)
            inp, tgt = self.encode_example(rec["prompt"], rec["response"], append_eos=append_eos)
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
        self.train_dataset = SFTDataset(
            config.train_file,
            self.tokenizer,
            config.max_seq_len,
            allow_exploratory_opt_in=config.allow_exploratory_opt_in,
            enforce_trust_gate=config.enforce_trust_gate,
            is_training=True,
            strip_prompt_delimiters=config.strip_prompt_delimiters,
        )
        self.val_dataset = SFTDataset(
            config.val_file,
            self.tokenizer,
            config.max_seq_len,
            allow_exploratory_opt_in=config.allow_exploratory_opt_in,
            enforce_trust_gate=config.enforce_trust_gate,
            is_training=False,
            strip_prompt_delimiters=config.strip_prompt_delimiters,
        )

        # Model and Optimizer setup
        self.device = torch.device(config.device)
        self.model = ChakrMicro(ModelConfig(max_seq_len=512)).to(self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        self.start_step = 0
        self.history: List[Dict[str, Any]] = []
        self.best_val_loss: float = float("inf")
        self.best_step: int = 0
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
                cur_vloss = val_metrics["loss"]
                step_record["val_loss"] = cur_vloss
                step_record["val_perplexity"] = val_metrics["perplexity"]

                # Check if new global minimum achieved (strictly less than, resolving ties deterministically to earlier step)
                if self.config.save_best_checkpoint and math.isfinite(cur_vloss) and cur_vloss < self.best_val_loss:
                    self.best_val_loss = cur_vloss
                    self.best_step = step
                    step_record["is_best"] = True
                    self.save_checkpoint(step, rng=rng, is_best=True)

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
            "best_val_loss": self.best_val_loss if math.isfinite(self.best_val_loss) else final_val["loss"],
            "best_val_step": self.best_step,
            "total_steps": self.config.steps,
            "elapsed_seconds": round(elapsed, 2),
            "history": self.history,
        }

    def save_checkpoint(
        self,
        step: int,
        rng: Optional[np.random.RandomState] = None,
        is_best: bool = False,
    ):
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
            "dataset_mode": "EXPLORATORY_OPT_IN" if self.config.allow_exploratory_opt_in else "VERIFIED_TRAINING",
            "allow_exploratory_opt_in": self.config.allow_exploratory_opt_in,
            "enforce_trust_gate": self.config.enforce_trust_gate,
            "train_file": str(self.config.train_file),
            "val_file": str(self.config.val_file),
            "train_file_sha256": self.train_dataset.file_sha256 if hasattr(self.train_dataset, "file_sha256") else None,
            "val_file_sha256": self.val_dataset.file_sha256 if hasattr(self.val_dataset, "file_sha256") else None,
            "is_best": is_best,
            "best_val_loss": self.best_val_loss if math.isfinite(self.best_val_loss) else None,
            "best_step": self.best_step if self.best_step > 0 else step,
        }

        torch.save(payload, temp_path)
        if temp_path.exists():
            temp_path.replace(target_path)

        # If this is the new best checkpoint, write atomic copy to sft_best_checkpoint.pt
        if is_best:
            best_target_path = self.config.output_dir / "sft_best_checkpoint.pt"
            best_temp_path = self.config.output_dir / "sft_best_checkpoint.pt.tmp"
            torch.save(payload, best_temp_path)
            if best_temp_path.exists():
                best_temp_path.replace(best_target_path)
