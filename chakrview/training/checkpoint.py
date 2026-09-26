"""
Atomic Checkpoint System for ChakrView (Phase 10).

Guarantees failure-safe, resumable checkpointing:
1. Writes state dictionary to a temporary file (.pt.tmp).
2. Performs atomic rename (os.replace) so incomplete writes cannot corrupt checkpoints.
3. Updates latest checkpoint pointer (latest_checkpoint.json).
4. Retains keep_last_n checkpoints, safely pruning older ones.
5. Captures model state, optimizer state, scheduler state, RNG state, step, epoch, and metrics.
"""

import os
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class CheckpointManager:
    """
    Atomic checkpoint manager with history pruning and metadata pointers.
    """
    def __init__(
        self,
        checkpoint_dir: Path | str,
        keep_last_n: int = 3,
        save_optimizer: bool = True,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.save_optimizer = save_optimizer
        self.pointer_file = self.checkpoint_dir / "latest_checkpoint.json"

    def save(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[Any] = None,
        step: int = 0,
        epoch: int = 0,
        config: Optional[Dict[str, Any]] = None,
        rng_state: Optional[Dict[str, Any]] = None,
        train_metrics: Optional[Dict[str, Any]] = None,
        val_metrics: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Atomically save complete training state to disk.
        """
        checkpoint_name = f"checkpoint_{step:07d}.pt"
        final_path = self.checkpoint_dir / checkpoint_name
        tmp_path = self.checkpoint_dir / f"{checkpoint_name}.tmp"

        payload = {
            "step": step,
            "epoch": epoch,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_state_dict": model.state_dict(),
            "config": config or {},
            "rng_state": rng_state or {},
            "train_metrics": train_metrics or {},
            "val_metrics": val_metrics or {},
        }

        if self.save_optimizer and optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            payload["scheduler_state_dict"] = scheduler.state_dict()

        # 1. Write to temporary file
        torch.save(payload, tmp_path)

        # 2. Atomic rename to final path
        os.replace(tmp_path, final_path)

        # 3. Update latest checkpoint pointer
        pointer_data = {
            "latest_checkpoint": checkpoint_name,
            "step": step,
            "timestamp": payload["timestamp"],
        }
        pointer_tmp = self.checkpoint_dir / "latest_checkpoint.json.tmp"
        with open(pointer_tmp, "w", encoding="utf-8") as f:
            json.dump(pointer_data, f, indent=2)
        os.replace(pointer_tmp, self.pointer_file)

        # 4. Prune older checkpoints
        self._prune_old_checkpoints()

        return final_path

    def _prune_old_checkpoints(self) -> None:
        """Keep only the most recent keep_last_n checkpoints."""
        checkpoints = sorted(
            list(self.checkpoint_dir.glob("checkpoint_*.pt")),
            key=lambda p: p.stat().st_mtime,
        )
        if len(checkpoints) > self.keep_last_n:
            to_remove = checkpoints[:-self.keep_last_n]
            for ckpt in to_remove:
                try:
                    ckpt.unlink(missing_ok=True)
                except OSError:
                    pass

    def get_latest_checkpoint_path(self) -> Optional[Path]:
        """Resolve path to the latest valid checkpoint."""
        if self.pointer_file.is_file():
            try:
                with open(self.pointer_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                latest_name = data.get("latest_checkpoint")
                if latest_name:
                    path = self.checkpoint_dir / latest_name
                    if path.is_file():
                        return path
            except Exception:
                pass

        # Fallback: find highest step file
        checkpoints = sorted(list(self.checkpoint_dir.glob("checkpoint_*.pt")))
        if checkpoints:
            return checkpoints[-1]
        return None

    @staticmethod
    def load(path: Path | str, weights_only: bool = False) -> Dict[str, Any]:
        """Load state payload from a checkpoint file."""
        ckpt_path = Path(path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
        return torch.load(ckpt_path, map_location="cpu", weights_only=weights_only)
