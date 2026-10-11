"""
Metrics Tracking and Accounting for ChakrView Pre-Training (Phase 11).

Tracks:
- step, training loss, validation loss, learning rate
- total tokens processed, total examples processed
- tokens/sec throughput, step elapsed time
- perplexity: exp(loss)
"""

import time
import math
from typing import Dict, Any, Optional


class MetricsTracker:
    """
    Cumulative and step-wise metrics accumulator for pre-training.
    """
    def __init__(self) -> None:
        self.start_time = time.perf_counter()
        self.step_start_time = self.start_time
        self.total_tokens_processed = 0
        self.total_examples_processed = 0
        self.history: list[Dict[str, Any]] = []

    def start_step(self) -> None:
        """Mark start time of a training step."""
        self.step_start_time = time.perf_counter()

    def step(
        self,
        step: int,
        loss: float,
        lr: float,
        tokens_in_step: int,
        batch_size: int,
        val_loss: Optional[float] = None,
        grad_norm: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Record step metrics and calculate throughput and perplexity.
        """
        now = time.perf_counter()
        step_elapsed = max(1e-6, now - self.step_start_time)
        total_elapsed = now - self.start_time

        self.total_tokens_processed += tokens_in_step
        self.total_examples_processed += batch_size

        tokens_per_sec = tokens_in_step / step_elapsed
        rounded_loss = round(loss, 4)
        train_perplexity = math.exp(min(20.0, rounded_loss)) if not math.isnan(loss) else float("nan")

        record = {
            "step": step,
            "train_loss": rounded_loss,
            "train_perplexity": round(train_perplexity, 2),
            "learning_rate": lr,
            "tokens_in_step": tokens_in_step,
            "tokens_per_sec": round(tokens_per_sec, 1),
            "step_elapsed_sec": round(step_elapsed, 4),
            "total_tokens": self.total_tokens_processed,
            "total_examples": self.total_examples_processed,
            "total_elapsed_sec": round(total_elapsed, 2),
        }

        if grad_norm is not None:
            record["grad_norm"] = round(grad_norm, 4)

        if val_loss is not None:
            val_perplexity = math.exp(min(20.0, val_loss)) if not math.isnan(val_loss) else float("nan")
            record["val_loss"] = round(val_loss, 4)
            record["val_perplexity"] = round(val_perplexity, 2)

        self.history.append(record)
        return record
