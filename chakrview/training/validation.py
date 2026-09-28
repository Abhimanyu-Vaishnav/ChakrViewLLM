"""
Validation & Perplexity Evaluation Engine for ChakrView (Step 22).

Computes rigorous validation loss and perplexity without runtime weight mutation:
1. Evaluates model in torch.no_grad() and model.eval() mode.
2. Computes mean causal cross-entropy loss over unmasked target tokens.
3. Computes perplexity = exp(min(100.0, loss)).
4. Mathematically robust: Handles NaN, Inf, and overflow safely without crashing.
5. NEVER fabricates metrics: If loss is non-finite, perplexity is reported as None/invalid.
"""

from dataclasses import dataclass, asdict, field
import math
import time
from typing import Dict, Optional, Any, Iterable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class ValidationResult:
    """
    Formal evaluation outcome record for model validation.
    """
    val_loss: float
    perplexity: Optional[float]
    perplexity_valid: bool
    total_tokens: int
    evaluated_examples: int
    batches_evaluated: int
    throughput_tokens_per_sec: float
    duration_ms: float
    status_note: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationResult":
        return cls(**data)


class ValidationEngine:
    """
    Independent validation coordinator evaluating ChakrMicro checkpoints on CPU.
    """

    @classmethod
    def evaluate(
        cls,
        model: nn.Module,
        data_loader: Iterable[Dict[str, Any]],
        loss_fn: nn.Module,
        max_batches: Optional[int] = None,
        device: str = "cpu",
    ) -> ValidationResult:
        """
        Evaluate model performance on validation dataset.
        """
        was_training = model.training
        model.eval()

        total_loss = 0.0
        total_tokens = 0
        total_examples = 0
        batches_evaluated = 0

        is_non_finite = False
        non_finite_val = 0.0
        t0 = time.perf_counter()

        with torch.no_grad():
            for batch in data_loader:
                if max_batches is not None and batches_evaluated >= max_batches:
                    break

                input_ids = batch["input_ids"].to(device)
                target_ids = batch["target_ids"].to(device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)

                logits = model(input_ids, attention_mask=attention_mask)
                loss = loss_fn(logits, target_ids)

                loss_val = loss.item()
                batches_evaluated += 1

                # Count valid tokens evaluated (excluding padding positions)
                pad_token_id = getattr(loss_fn, "ignore_index", 2)
                valid_tokens = (target_ids != pad_token_id).sum().item()
                total_tokens += int(valid_tokens)
                total_examples += input_ids.size(0)

                if math.isnan(loss_val) or math.isinf(loss_val):
                    is_non_finite = True
                    non_finite_val = loss_val
                    break
                else:
                    total_loss += loss_val

        duration_s = max(1e-6, time.perf_counter() - t0)
        duration_ms = duration_s * 1000.0

        if was_training:
            model.train()

        throughput = total_tokens / duration_s

        if is_non_finite:
            return ValidationResult(
                val_loss=non_finite_val,
                perplexity=None,
                perplexity_valid=False,
                total_tokens=total_tokens,
                evaluated_examples=total_examples,
                batches_evaluated=batches_evaluated,
                throughput_tokens_per_sec=round(throughput, 2),
                duration_ms=round(duration_ms, 2),
                status_note="Non-finite validation loss; perplexity is mathematically undefined.",
            )

        if batches_evaluated == 0:
            return ValidationResult(
                val_loss=float("nan"),
                perplexity=None,
                perplexity_valid=False,
                total_tokens=0,
                evaluated_examples=0,
                batches_evaluated=0,
                throughput_tokens_per_sec=0.0,
                duration_ms=duration_ms,
                status_note="No validation batches evaluated.",
            )

        mean_loss = total_loss / batches_evaluated
        throughput = total_tokens / duration_s

        # Handle non-finite loss edge cases honestly
        if math.isnan(mean_loss) or math.isinf(mean_loss):
            return ValidationResult(
                val_loss=mean_loss,
                perplexity=None,
                perplexity_valid=False,
                total_tokens=total_tokens,
                evaluated_examples=total_examples,
                batches_evaluated=batches_evaluated,
                throughput_tokens_per_sec=round(throughput, 2),
                duration_ms=round(duration_ms, 2),
                status_note="Non-finite validation loss; perplexity is mathematically undefined.",
            )

        # Check for loss magnitude overflow in exp()
        if mean_loss > 85.0:
            return ValidationResult(
                val_loss=round(mean_loss, 4),
                perplexity=None,
                perplexity_valid=False,
                total_tokens=total_tokens,
                evaluated_examples=total_examples,
                batches_evaluated=batches_evaluated,
                throughput_tokens_per_sec=round(throughput, 2),
                duration_ms=round(duration_ms, 2),
                status_note=f"High loss ({mean_loss:.2f}) causes numerical overflow in perplexity calculation.",
            )

        perplexity = round(math.exp(mean_loss), 4)

        return ValidationResult(
            val_loss=round(mean_loss, 4),
            perplexity=perplexity,
            perplexity_valid=True,
            total_tokens=total_tokens,
            evaluated_examples=total_examples,
            batches_evaluated=batches_evaluated,
            throughput_tokens_per_sec=round(throughput, 2),
            duration_ms=round(duration_ms, 2),
            status_note="Validation completed successfully.",
        )
