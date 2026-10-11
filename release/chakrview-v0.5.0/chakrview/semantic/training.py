"""
Training pipeline for the Sovereign Semantic Encoder (Step 14).

Implements CPU-friendly contrastive fine-tuning using InfoNCE loss with
deterministic reproducibility, learning rate scheduling, validation, and checkpointing.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
import random
import time
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import torch
from torch.utils.data import DataLoader

from chakrview.semantic.config import SemanticEncoderConfig
from chakrview.semantic.encoder import SemanticEncoder
from chakrview.semantic.dataset import SemanticDataset, SemanticCollator
from chakrview.semantic.loss import InfoNCELoss
from chakrview.semantic.serialization import save_semantic_encoder, load_semantic_encoder


@dataclass
class SemanticTrainingConfig:
    """
    Hyperparameters and runtime configuration for semantic encoder training.
    """
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 4
    epochs: int = 5
    temperature: float = 0.05
    seed: int = 42
    device: str = "cpu"
    gradient_clip_norm: float = 1.0
    checkpoint_dir: Optional[str] = None
    save_best: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SemanticTrainingConfig":
        return cls(**data)


@dataclass
class TrainingHistory:
    """
    Per-epoch loss records and optimization metrics.
    """
    epoch_train_losses: List[float] = field(default_factory=list)
    epoch_val_losses: List[float] = field(default_factory=list)
    total_training_time_s: float = 0.0
    best_val_loss: float = float("inf")
    best_epoch: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SemanticTrainer:
    """
    Coordinates training, validation, and checkpointing for SemanticEncoder.
    """

    def __init__(
        self,
        model: SemanticEncoder,
        tokenizer: Any,
        train_config: Optional[SemanticTrainingConfig] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = train_config or SemanticTrainingConfig()
        self.device = torch.device(self.config.device)
        self.model.to(self.device)

        # Loss function
        self.loss_fn = InfoNCELoss(temperature=self.config.temperature)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self._set_seed(self.config.seed)

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Run one full training epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            q_ids = batch["query_ids"].to(self.device)
            q_mask = batch["query_mask"].to(self.device)
            p_ids = batch["positive_ids"].to(self.device)
            p_mask = batch["positive_mask"].to(self.device)

            self.optimizer.zero_grad()

            q_vecs = self.model(q_ids, attention_mask=q_mask)
            p_vecs = self.model(p_ids, attention_mask=p_mask)

            n_vecs = None
            if "negative_ids" in batch:
                n_ids = batch["negative_ids"].to(self.device)
                n_mask = batch["negative_mask"].to(self.device)
                n_vecs = self.model(n_ids, attention_mask=n_mask)

            loss = self.loss_fn(q_vecs, p_vecs, negative_embeddings=n_vecs)
            loss.backward()

            if self.config.gradient_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm,
                )

            self.optimizer.step()

            total_loss += float(loss.item())
            num_batches += 1

        return total_loss / max(1, num_batches)

    def validate(self, dataloader: DataLoader) -> float:
        """Evaluate validation loss without gradient computation."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                q_ids = batch["query_ids"].to(self.device)
                q_mask = batch["query_mask"].to(self.device)
                p_ids = batch["positive_ids"].to(self.device)
                p_mask = batch["positive_mask"].to(self.device)

                q_vecs = self.model(q_ids, attention_mask=q_mask)
                p_vecs = self.model(p_ids, attention_mask=p_mask)

                n_vecs = None
                if "negative_ids" in batch:
                    n_ids = batch["negative_ids"].to(self.device)
                    n_mask = batch["negative_mask"].to(self.device)
                    n_vecs = self.model(n_ids, attention_mask=n_mask)

                loss = self.loss_fn(q_vecs, p_vecs, negative_embeddings=n_vecs)
                total_loss += float(loss.item())
                num_batches += 1

        return total_loss / max(1, num_batches)

    def fit(
        self,
        train_dataset: SemanticDataset,
        val_dataset: Optional[SemanticDataset] = None,
    ) -> TrainingHistory:
        """
        Execute full training loop across configured epochs.
        """
        collator = SemanticCollator(
            tokenizer=self.tokenizer,
            max_seq_len=self.model.config.max_seq_len,
            pad_token_id=self.model.config.pad_token_id,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collator,
        )

        val_loader = None
        if val_dataset is not None and len(val_dataset) > 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=collator,
            )

        history = TrainingHistory()
        t0 = time.perf_counter()

        ckpt_dir = Path(self.config.checkpoint_dir) if self.config.checkpoint_dir else None

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self.train_epoch(train_loader)
            history.epoch_train_losses.append(train_loss)

            val_loss = 0.0
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                history.epoch_val_losses.append(val_loss)

                if val_loss < history.best_val_loss:
                    history.best_val_loss = val_loss
                    history.best_epoch = epoch
                    if ckpt_dir and self.config.save_best:
                        save_semantic_encoder(
                            self.model,
                            ckpt_dir / "best_semantic_encoder.pt",
                            metadata={"val_loss": val_loss, "epoch": epoch},
                        )
            else:
                history.epoch_val_losses.append(train_loss)
                if train_loss < history.best_val_loss:
                    history.best_val_loss = train_loss
                    history.best_epoch = epoch

        history.total_training_time_s = time.perf_counter() - t0

        if ckpt_dir:
            save_semantic_encoder(
                self.model,
                ckpt_dir / "final_semantic_encoder.pt",
                metadata={"final_train_loss": history.epoch_train_losses[-1]},
            )

        return history

    def save_checkpoint(self, path: Union[str, Path], metadata: Optional[Dict[str, Any]] = None) -> Path:
        """Save active model state."""
        return save_semantic_encoder(self.model, path, metadata)

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Load state into active model."""
        loaded_model, _ = load_semantic_encoder(path, device=self.device)
        self.model.load_state_dict(loaded_model.state_dict())
