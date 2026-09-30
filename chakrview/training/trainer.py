"""
Core Pre-Training Engine for ChakrView (Phase 9).

Executes the optimization loop:
- forward pass
- loss computation
- gradient accumulation & backward
- gradient clipping
- optimizer & learning rate scheduling
- metrics logging
- periodic validation evaluation
- atomic checkpointing and resumption
"""

import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any, Iterable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from chakrview.brain.model import ChakrMicro
from chakrview.training.config import PretrainingConfig
from chakrview.training.seed import set_seed, get_rng_state, set_rng_state
from chakrview.training.loss import CausalLoss
from chakrview.training.optimizer import build_optimizer, build_lr_scheduler
from chakrview.training.checkpoint import CheckpointManager
from chakrview.training.metrics import MetricsTracker
from chakrview.training.evaluator import evaluate
from chakrview.training.safety import TrainingSafetyChecker


class Trainer:
    """
    Modular, restartable pre-training coordinator for ChakrMicro on CPU.
    """
    def __init__(
        self,
        config: PretrainingConfig,
        model: Optional[nn.Module] = None,
        train_loader: Optional[Iterable[Dict[str, torch.Tensor]]] = None,
        val_loader: Optional[Iterable[Dict[str, torch.Tensor]]] = None,
        tokenizer_checksum: Optional[str] = None,
        dataset_manifest_hash: Optional[str] = None,
    ) -> None:
        self.config = config
        self.device = "cpu"
        self.tokenizer_checksum = tokenizer_checksum or getattr(config.data, "tokenizer_checksum", None)
        self.dataset_manifest_hash = dataset_manifest_hash or getattr(config.data, "dataset_manifest_hash", None)

        # 1. Determinism
        set_seed(config.training.seed)

        # 2. Model
        self.model = model if model is not None else ChakrMicro(config.model)
        self.model.to(self.device)
        TrainingSafetyChecker.enforce_model_invariants(self.model)

        # 3. Loss
        self.loss_fn = CausalLoss(ignore_index=config.data.pad_token_id)

        # 4. Optimizer & Scheduler
        self.optimizer = build_optimizer(self.model, config.training)
        self.scheduler = build_lr_scheduler(self.optimizer, config.training)

        # 5. Checkpoint Manager & Metrics
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=config.checkpoint.directory,
            keep_last_n=config.checkpoint.keep_last_n,
            save_optimizer=config.checkpoint.save_optimizer,
        )
        self.metrics = MetricsTracker()

        # 6. Data Loaders
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Internal state
        self.current_step = 0
        self.micro_step = 0
        self.latest_grad_norm: Optional[float] = None

    def resume(self, checkpoint_path: Optional[Path | str] = None) -> int:
        """
        Resume trainer state from checkpoint file or latest available checkpoint.
        Validates training checkpoint integrity and compatibility before loading.
        """
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_manager.get_latest_checkpoint_path()

        if checkpoint_path is None:
            raise FileNotFoundError("No valid checkpoint found to resume from.")

        param_count = sum(p.numel() for p in self.model.parameters())
        payload = CheckpointManager.load(
            checkpoint_path,
            validate_training=True,
            expected_tokenizer_checksum=self.tokenizer_checksum,
            expected_param_count=param_count,
        )

        # 1. Restore model state
        self.model.load_state_dict(payload["model_state_dict"])

        # 2. Restore optimizer and scheduler
        if "optimizer_state_dict" in payload and self.optimizer is not None:
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        if "scheduler_state_dict" in payload and self.scheduler is not None:
            self.scheduler.load_state_dict(payload["scheduler_state_dict"])

        # 3. Restore RNG state
        if "rng_state" in payload and payload["rng_state"]:
            set_rng_state(payload["rng_state"])

        # 4. Restore step
        self.current_step = payload.get("step", 0)
        return self.current_step

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Execute a single forward-backward-accumulate optimization step.
        Enforces token range verification, loss finiteness, gradient finiteness,
        and records gradient norm.
        """
        self.model.train()
        input_ids = batch["input_ids"].to(self.device)
        target_ids = batch["target_ids"].to(self.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # 1. Bounds verification
        TrainingSafetyChecker.verify_token_ids(
            input_ids,
            min_id=0,
            max_id=self.config.model.vocab_size - 1,
            allowed_ignore_index=self.config.data.pad_token_id,
        )
        TrainingSafetyChecker.verify_token_ids(
            target_ids,
            min_id=0,
            max_id=self.config.model.vocab_size - 1,
            allowed_ignore_index=self.config.data.pad_token_id,
        )

        # 2. Forward pass
        logits = self.model(input_ids, attention_mask=attention_mask)
        loss = self.loss_fn(logits, target_ids)

        # 3. Safety check: Loss finiteness
        loss_val = TrainingSafetyChecker.check_loss(loss, step=self.current_step)

        # 4. Backward pass
        scaled_loss = loss / self.config.training.gradient_accumulation_steps
        scaled_loss.backward()

        self.micro_step += 1

        # 5. Optimizer step upon completing accumulation window
        if self.micro_step % self.config.training.gradient_accumulation_steps == 0:
            # Gradient verification
            grad_norm = TrainingSafetyChecker.check_gradients(
                self.model, step=self.current_step
            )
            self.latest_grad_norm = grad_norm

            if self.config.training.gradient_clipping > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.gradient_clipping
                )

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            self.current_step += 1
        else:
            self.latest_grad_norm = None

        return loss_val

    def train(self) -> Dict[str, Any]:
        """
        Main training loop running up to config.training.max_steps.
        """
        if self.train_loader is None:
            raise ValueError("train_loader must be provided to run train()")

        self.model.train()
        self.optimizer.zero_grad()
        latest_val_metrics: Dict[str, float] = {}

        # Initial evaluation
        if self.config.evaluation.eval_on_start and self.val_loader is not None:
            latest_val_metrics = evaluate(
                self.model,
                self.val_loader,
                self.loss_fn,
                max_batches=self.config.evaluation.eval_batches,
                device=self.device,
            )

        data_iter = iter(self.train_loader)

        while self.current_step < self.config.training.max_steps:
            self.metrics.start_step()
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            B, T = batch["input_ids"].shape
            tokens_in_step = B * T

            step_loss = self.train_step(batch)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Evaluation interval
            if (
                self.val_loader is not None
                and self.current_step > 0
                and self.current_step % self.config.evaluation.eval_interval == 0
                and self.micro_step % self.config.training.gradient_accumulation_steps == 0
            ):
                latest_val_metrics = evaluate(
                    self.model,
                    self.val_loader,
                    self.loss_fn,
                    max_batches=self.config.evaluation.eval_batches,
                    device=self.device,
                )

            # Record step metrics
            record = self.metrics.step(
                step=self.current_step,
                loss=step_loss,
                lr=current_lr,
                tokens_in_step=tokens_in_step,
                batch_size=B,
                val_loss=latest_val_metrics.get("val_loss"),
                grad_norm=self.latest_grad_norm,
            )

            # Checkpointing interval
            if (
                self.current_step > 0
                and self.current_step % self.config.checkpoint.save_interval == 0
                and self.micro_step % self.config.training.gradient_accumulation_steps == 0
            ):
                self.checkpoint_manager.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    step=self.current_step,
                    epoch=0,
                    config=self.config.to_dict(),
                    model_config=self.model.config.to_dict() if hasattr(self.model, "config") and hasattr(self.model.config, "to_dict") else {},
                    tokenizer_checksum=self.tokenizer_checksum,
                    dataset_manifest_hash=self.dataset_manifest_hash,
                    parameter_count=sum(p.numel() for p in self.model.parameters()),
                    rng_state=get_rng_state(),
                    train_metrics=record,
                    val_metrics=latest_val_metrics,
                )

        # Final checkpoint save at completion
        final_ckpt = self.checkpoint_manager.save(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=self.current_step,
            epoch=0,
            config=self.config.to_dict(),
            model_config=self.model.config.to_dict() if hasattr(self.model, "config") and hasattr(self.model.config, "to_dict") else {},
            tokenizer_checksum=self.tokenizer_checksum,
            dataset_manifest_hash=self.dataset_manifest_hash,
            parameter_count=sum(p.numel() for p in self.model.parameters()),
            rng_state=get_rng_state(),
            train_metrics=self.metrics.history[-1] if self.metrics.history else {},
            val_metrics=latest_val_metrics,
        )

        return {
            "final_step": self.current_step,
            "final_checkpoint": str(final_ckpt),
            "history": self.metrics.history,
        }

