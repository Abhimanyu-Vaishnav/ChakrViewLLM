"""
ChakrView Step 146: Controlled Neural Language Learning & Generalization.

Features:
- MicroLanguageCurriculum: Deterministic linguistic sequence generation covering syntax,
  delimiter pairs, and short contextual dependencies.
- ControlledNeuralLanguageTrainer: Trains an isolated candidate model using CPU PyTorch optimization.
- Multi-split evaluation: Measures train loss, validation loss, held-out loss, and next-token accuracy.
- Computes generalization delta: (candidate_heldout_acc - baseline_heldout_acc).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.tokenizer import BPETokenizer
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.tokenizer.special_tokens import PAD_ID


@dataclass
class LanguageLearningResult:
    baseline_train_loss: float
    candidate_train_loss: float
    baseline_val_loss: float
    candidate_val_loss: float
    baseline_heldout_acc: float
    candidate_heldout_acc: float
    generalization_delta: float
    loss_reduction_pct: float
    training_steps: int
    is_improved: bool


class ControlledNeuralLanguageTrainer:
    """
    Executes controlled gradient updates on ChakrMicro candidate on CPU.
    """

    def __init__(
        self,
        tokenizer: Optional[BPETokenizer] = None,
        learning_rate: float = 5e-4,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.learning_rate = learning_rate
        if tokenizer is None:
            tok_dir = Path("data/experiments/vocab_4096")
            tok, _ = load_tokenizer_artifacts(tok_dir)
            self.tokenizer = tok
        else:
            self.tokenizer = tokenizer

    def create_synthetic_language_corpus(self) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Creates train, validation, and held-out token sequences representing structured language syntax.
        """
        # Linguistic patterns: (Subject, Verb, Object, Delimiter)
        patterns_train = [
            "function calculate(a, b) { return a + b; }",
            "function multiply(x, y) { return x * y; }",
            "function divide(val1, val2) { return val1 / val2; }",
            "function subtract(p, q) { return p - q; }",
            "def run_job(worker_id): return status_ok",
            "def fetch_record(key): return store[key]",
            "def init_node(host, port): return socket_bind",
            "def verify_data(payload): return is_valid",
        ]
        patterns_val = [
            "function evaluate(m, n) { return m + n; }",
            "def stop_node(host, port): return socket_close",
        ]
        patterns_heldout = [
            "function aggregate(u, v) { return u + v; }",
            "def sync_record(key): return store[key]",
            "function power(base, exp) { return base * exp; }",
            "def ping_node(host, port): return socket_alive",
        ]

        def to_tensors(patterns: List[str]) -> List[torch.Tensor]:
            tensors = []
            for p in patterns:
                ids = self.tokenizer.encode(p, add_bos=True, add_eos=True)
                # Pad/truncate to 32 tokens
                if len(ids) < 32:
                    ids = ids + [PAD_ID] * (32 - len(ids))
                else:
                    ids = ids[:32]
                tensors.append(torch.tensor(ids, dtype=torch.long, device=self.device))
            return tensors

        return to_tensors(patterns_train), to_tensors(patterns_val), to_tensors(patterns_heldout)

    def evaluate_loss_and_acc(self, model: ChakrMicro, data: List[torch.Tensor]) -> Tuple[float, float]:
        model.eval()
        total_loss = 0.0
        correct_tokens = 0
        total_tokens = 0

        loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)
        with torch.no_grad():
            for seq in data:
                inp = seq[:-1].unsqueeze(0)
                tgt = seq[1:].unsqueeze(0)
                logits = model(inp)
                loss = loss_fn(logits.view(-1, logits.size(-1)), tgt.view(-1))
                total_loss += loss.item()

                preds = torch.argmax(logits, dim=-1)
                mask = (tgt != PAD_ID)
                correct_tokens += ((preds == tgt) & mask).sum().item()
                total_tokens += mask.sum().item()

        avg_loss = total_loss / max(1, len(data))
        acc = correct_tokens / max(1, total_tokens)
        return avg_loss, acc

    def train_candidate(
        self,
        candidate_model: ChakrMicro,
        baseline_model: ChakrMicro,
        steps: int = 15,
    ) -> LanguageLearningResult:
        train_data, val_data, held_data = self.create_synthetic_language_corpus()

        # 1. Baseline Evaluation
        base_train_loss, _ = self.evaluate_loss_and_acc(baseline_model, train_data)
        base_val_loss, _ = self.evaluate_loss_and_acc(baseline_model, val_data)
        _, base_held_acc = self.evaluate_loss_and_acc(baseline_model, held_data)

        # 2. Candidate Training Loop on CPU
        candidate_model.train()
        optimizer = torch.optim.AdamW(candidate_model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)

        for step in range(steps):
            idx = step % len(train_data)
            seq = train_data[idx]
            inp = seq[:-1].unsqueeze(0)
            tgt = seq[1:].unsqueeze(0)

            optimizer.zero_grad()
            logits = candidate_model(inp)
            loss = loss_fn(logits.view(-1, logits.size(-1)), tgt.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(candidate_model.parameters(), 1.0)
            optimizer.step()

        # 3. Candidate Evaluation
        cand_train_loss, _ = self.evaluate_loss_and_acc(candidate_model, train_data)
        cand_val_loss, _ = self.evaluate_loss_and_acc(candidate_model, val_data)
        _, cand_held_acc = self.evaluate_loss_and_acc(candidate_model, held_data)

        loss_reduction = max(0.0, ((base_train_loss - cand_train_loss) / base_train_loss) * 100.0)
        gen_delta = round(cand_held_acc - base_held_acc, 4)

        return LanguageLearningResult(
            baseline_train_loss=round(base_train_loss, 4),
            candidate_train_loss=round(cand_train_loss, 4),
            baseline_val_loss=round(base_val_loss, 4),
            candidate_val_loss=round(cand_val_loss, 4),
            baseline_heldout_acc=round(base_held_acc, 4),
            candidate_heldout_acc=round(cand_held_acc, 4),
            generalization_delta=gen_delta,
            loss_reduction_pct=round(loss_reduction, 2),
            training_steps=steps,
            is_improved=(cand_train_loss < base_train_loss),
        )
