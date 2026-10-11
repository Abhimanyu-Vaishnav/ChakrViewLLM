"""
Offline CPU Training Engine for ChakrMicro (Step 22).

Executes reproducible, bounded neural training on CPU:
1. Enforces frozen ChakrMicro architecture (3,443,136 params, 4096 vocab, 512 context).
2. Operates strictly offline on a dedicated training model instance (zero mutation of inference).
3. Executes causal language modeling optimization with shifted next-token prediction.
4. Integrates hard safety checks (fails closed on NaN/Inf loss or gradients).
5. Provides deterministic gradient accumulation, clipping, checkpointing, and resumption.
6. Emits cryptographically verifiable TrainingRunManifests.
"""

from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Dict, List, Optional, Any, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.tokenizer import BPETokenizer
from chakrview.training.seed import set_seed, get_rng_state, set_rng_state
from chakrview.training.loss import CausalLoss
from chakrview.training.checkpoint import CheckpointManager
from chakrview.training.metrics import MetricsTracker
from chakrview.training.contract import TrainingDatasetManifest, TokenizerFingerprint
from chakrview.training.builder import ChakrOfflineDataset
from chakrview.training.safety import (
    TrainingSafetyChecker,
    TrainingSafetyError,
    NumericalInstabilityError,
    InvariantViolationError,
    CheckpointCorruptionError,
)
from chakrview.training.validation import ValidationEngine, ValidationResult
from chakrview.training.manifest import TrainingRunManifest


@dataclass
class TrainingResult:
    """Outcome package of a completed or paused training run."""
    run_id: str
    steps_completed: int
    total_tokens_trained: int
    final_loss: float
    final_val_loss: Optional[float]
    final_val_perplexity: Optional[float]
    final_checkpoint_path: Optional[str]
    manifest: TrainingRunManifest
    history: List[Dict[str, Any]]
    success: bool
    status_message: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["manifest"] = self.manifest.to_dict()
        return d


class CPUTrainingEngine:
    """
    Dedicated offline CPU training engine for ChakrMicro models.
    """

    def __init__(
        self,
        model: Optional[ChakrMicro] = None,
        model_config: Optional[ModelConfig] = None,
        tokenizer: Optional[BPETokenizer] = None,
        train_dataset: Optional[ChakrOfflineDataset] = None,
        val_dataset: Optional[ChakrOfflineDataset] = None,
        batch_size: int = 4,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        gradient_accumulation_steps: int = 1,
        gradient_clipping: float = 1.0,
        checkpoint_dir: Optional[Path | str] = None,
        seed: int = 42,
    ) -> None:
        self.device = "cpu"
        self.seed = seed
        set_seed(self.seed)

        # 1. Model Configuration & Instantiation
        self.model_config = model_config or ModelConfig()
        if model is not None:
            self.model = model
        else:
            self.model = ChakrMicro(self.model_config)
        self.model.to(self.device)

        # 2. Invariant Verification: must strictly match frozen ChakrMicro invariants
        TrainingSafetyChecker.enforce_model_invariants(self.model)

        # 3. Tokenizer
        self.tokenizer = tokenizer or BPETokenizer()
        self.tok_fingerprint = TokenizerFingerprint.from_tokenizer(self.tokenizer)

        # 4. Datasets
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = max(1, batch_size)

        # 5. Optimization Hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self.gradient_clipping = gradient_clipping

        # Optimizer: exclude 1D biases and layer norms from weight decay
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() >= 2:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        self.optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        # 6. Loss & Evaluation
        self.loss_fn = CausalLoss(ignore_index=2)  # PAD_ID = 2
        self.metrics = MetricsTracker()

        # 7. Checkpointing
        ckpt_dir = checkpoint_dir or Path("checkpoints/offline_training")
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=ckpt_dir,
            keep_last_n=3,
            save_optimizer=True,
        )

        # Runtime Tracking
        self.current_step = 0
        self.micro_step = 0
        self.total_tokens_trained = 0
        self.run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{self.seed}"

    def resume(self, checkpoint_path: Path | str) -> int:
        """
        Resume trainer state from an existing checkpoint.
        Validates checkpoint metadata and parameter invariants before loading.
        """
        payload = CheckpointManager.load(checkpoint_path)
        TrainingSafetyChecker.verify_checkpoint_metadata(payload)

        self.model.load_state_dict(payload["model_state_dict"])
        if "optimizer_state_dict" in payload and self.optimizer is not None:
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        if "rng_state" in payload and payload["rng_state"]:
            set_rng_state(payload["rng_state"])

        self.current_step = payload.get("step", 0)
        return self.current_step

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Execute one micro-step (forward pass, loss check, scaled backward).
        """
        self.model.train()

        input_ids = batch["input_ids"].to(self.device)
        target_ids = batch["target_ids"].to(self.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Safety Check: Verify token IDs are within valid range [0, 4095]
        TrainingSafetyChecker.verify_token_ids(input_ids, min_id=0, max_id=4095)
        TrainingSafetyChecker.verify_token_ids(target_ids, min_id=0, max_id=4095)

        # Forward Pass
        logits = self.model(input_ids, attention_mask=attention_mask)
        loss = self.loss_fn(logits, target_ids)

        # Safety Check: Validate loss is a finite number
        loss_val = TrainingSafetyChecker.check_loss(loss, step=self.current_step)

        # Scaled backward for gradient accumulation
        scaled_loss = loss / self.gradient_accumulation_steps
        scaled_loss.backward()

        self.micro_step += 1

        # Optimizer step upon completing gradient accumulation cycle
        if self.micro_step % self.gradient_accumulation_steps == 0:
            # Safety Check: Validate gradients are finite numbers
            TrainingSafetyChecker.check_gradients(self.model, step=self.current_step)

            if self.gradient_clipping > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clipping
                )

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.current_step += 1

        return loss_val

    def train(
        self,
        max_steps: int = 10,
        eval_interval: int = 5,
        save_interval: int = 5,
        dataset_manifest: Optional[TrainingDatasetManifest] = None,
    ) -> TrainingResult:
        """
        Execute offline CPU training up to max_steps.
        """
        if self.train_dataset is None or len(self.train_dataset) == 0:
            raise ValueError("train_dataset must be provided and non-empty for training.")

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=ChakrOfflineDataset.collate_fn,
        )

        val_loader = None
        if self.val_dataset is not None and len(self.val_dataset) > 0:
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=ChakrOfflineDataset.collate_fn,
            )

        start_time_iso = datetime.now(timezone.utc).isoformat()
        latest_val_res: Optional[ValidationResult] = None
        saved_checkpoints: List[str] = []

        # Initial evaluation
        if val_loader is not None:
            latest_val_res = ValidationEngine.evaluate(
                self.model, val_loader, self.loss_fn, device=self.device
            )

        data_iter = iter(train_loader)
        last_step_loss = 0.0

        while self.current_step < max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            B, T = batch["input_ids"].shape
            tokens_in_batch = int((batch["target_ids"] != 2).sum().item())
            self.total_tokens_trained += tokens_in_batch

            last_step_loss = self.train_step(batch)

            # Periodic Validation
            if (
                val_loader is not None
                and self.current_step > 0
                and self.current_step % eval_interval == 0
                and self.micro_step % self.gradient_accumulation_steps == 0
            ):
                latest_val_res = ValidationEngine.evaluate(
                    self.model, val_loader, self.loss_fn, device=self.device
                )

            # Periodic Checkpoint Saving
            if (
                self.current_step > 0
                and self.current_step % save_interval == 0
                and self.micro_step % self.gradient_accumulation_steps == 0
            ):
                ckpt_p = self.checkpoint_manager.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    step=self.current_step,
                    epoch=0,
                    config=asdict(self.model_config) if hasattr(self.model_config, "__dataclass_fields__") else {},
                    rng_state=get_rng_state(),
                    train_metrics={"loss": last_step_loss, "tokens": self.total_tokens_trained},
                    val_metrics=latest_val_res.to_dict() if latest_val_res else {},
                )
                saved_checkpoints.append(str(ckpt_p))

        # Final Validation Pass
        if val_loader is not None:
            latest_val_res = ValidationEngine.evaluate(
                self.model, val_loader, self.loss_fn, device=self.device
            )

        # Final Checkpoint
        final_ckpt = self.checkpoint_manager.save(
            model=self.model,
            optimizer=self.optimizer,
            step=self.current_step,
            epoch=0,
            config=asdict(self.model_config) if hasattr(self.model_config, "__dataclass_fields__") else {},
            rng_state=get_rng_state(),
            train_metrics={"loss": last_step_loss, "tokens": self.total_tokens_trained},
            val_metrics=latest_val_res.to_dict() if latest_val_res else {},
        )
        saved_checkpoints.append(str(final_ckpt))

        end_time_iso = datetime.now(timezone.utc).isoformat()

        # Build reproducible TrainingRunManifest
        manifest = TrainingRunManifest(
            run_id=self.run_id,
            seed=self.seed,
            model_version="chakrmicro-v0.1-trained",
            param_count=sum(p.numel() for p in self.model.parameters()),
            vocab_size=self.model_config.vocab_size,
            context_length=self.model_config.max_seq_len,
            tokenizer_fingerprint=self.tok_fingerprint.fingerprint_hash,
            dataset_fingerprint=dataset_manifest.dataset_fingerprint if dataset_manifest else "direct_dataset",
            dataset_version=dataset_manifest.dataset_version if dataset_manifest else "v1.0",
            optimizer_name="AdamW",
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            gradient_clipping=self.gradient_clipping,
            total_steps=self.current_step,
            hardware="CPU",
            threads=torch.get_num_threads(),
            framework_version=torch.__version__,
            start_timestamp=start_time_iso,
            end_timestamp=end_time_iso,
            checkpoints_saved=saved_checkpoints,
            final_train_loss=round(last_step_loss, 4),
            final_val_loss=latest_val_res.val_loss if latest_val_res else None,
            final_perplexity=latest_val_res.perplexity if latest_val_res else None,
        )

        return TrainingResult(
            run_id=self.run_id,
            steps_completed=self.current_step,
            total_tokens_trained=self.total_tokens_trained,
            final_loss=round(last_step_loss, 4),
            final_val_loss=latest_val_res.val_loss if latest_val_res else None,
            final_val_perplexity=latest_val_res.perplexity if latest_val_res else None,
            final_checkpoint_path=str(final_ckpt),
            manifest=manifest,
            history=self.metrics.history,
            success=True,
            status_message="Training run completed successfully.",
        )
