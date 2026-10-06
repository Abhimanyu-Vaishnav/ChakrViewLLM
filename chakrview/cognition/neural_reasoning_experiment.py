"""
ChakrView Step 156: Neural Reasoning Training Experiment.

Executes CPU-based gradient descent optimization on isolated ChakrMicro candidates
using the staged reasoning curriculum from Steps 154-155.
Measures:
- Train reasoning accuracy
- In-distribution validation reasoning accuracy
- Held-out unseen entity reasoning accuracy
- Parameter updates and causal cross-entropy loss curves.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.tokenizer import BPETokenizer
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.tokenizer.special_tokens import PAD_ID
from chakrview.cognition.reasoning_curriculum_ladder import ReasoningLadderItem


@dataclass
class ReasoningTrainingRunMetrics:
    learning_rate: float
    steps: int
    train_loss_initial: float
    train_loss_final: float
    train_reasoning_acc: float
    held_out_reasoning_acc: float
    runtime_seconds: float
    parameter_norm_delta: float


class NeuralReasoningExperimentRunner:
    """
    Executes controlled neural training runs on reasoning tasks using CPU.
    """

    def __init__(
        self,
        tokenizer: Optional[BPETokenizer] = None,
        device: str = "cpu",
    ) -> None:
        self.device = device
        if tokenizer is None:
            tok_dir = Path("data/experiments/vocab_4096")
            tok, _ = load_tokenizer_artifacts(tok_dir)
            self.tokenizer = tok
        else:
            self.tokenizer = tokenizer

    def evaluate_items(self, model: ChakrMicro, items: List[ReasoningLadderItem]) -> float:
        model.eval()
        correct = 0
        total = 0

        for item in items:
            p_ids = self.tokenizer.encode(item.prompt, add_bos=True, add_eos=False)
            t_ids = self.tokenizer.encode(item.target, add_bos=False, add_eos=False)
            if not t_ids:
                continue
            target_id = t_ids[0]

            inp = torch.tensor([p_ids], dtype=torch.long, device=self.device)
            with torch.no_grad():
                logits = model(inp)
                pred_id = torch.argmax(logits[0, -1, :]).item()

            if pred_id == target_id:
                correct += 1
            total += 1

        return correct / max(1, total)

    def train_candidate_reasoning(
        self,
        candidate_model: ChakrMicro,
        train_items: List[ReasoningLadderItem],
        heldout_items: List[ReasoningLadderItem],
        steps: int = 30,
        lr: float = 1e-3,
    ) -> ReasoningTrainingRunMetrics:
        t0 = time.perf_counter()

        # Build tensors
        train_tensors = []
        for item in train_items:
            full_text = item.prompt + item.target
            ids = self.tokenizer.encode(full_text, add_bos=True, add_eos=True)
            if len(ids) < 24:
                ids = ids + [PAD_ID] * (24 - len(ids))
            else:
                ids = ids[:24]
            train_tensors.append(torch.tensor(ids, dtype=torch.long, device=self.device))

        candidate_model.train()
        optimizer = torch.optim.AdamW(candidate_model.parameters(), lr=lr, weight_decay=0.01)
        loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)

        init_loss = 0.0
        final_loss = 0.0

        for s in range(steps):
            seq = train_tensors[s % len(train_tensors)]
            inp = seq[:-1].unsqueeze(0)
            tgt = seq[1:].unsqueeze(0)

            optimizer.zero_grad()
            logits = candidate_model(inp)
            loss = loss_fn(logits.view(-1, logits.size(-1)), tgt.view(-1))
            if s == 0:
                init_loss = loss.item()
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

        elapsed = time.perf_counter() - t0
        train_acc = self.evaluate_items(candidate_model, train_items)
        held_acc = self.evaluate_items(candidate_model, heldout_items)

        # Compute parameter update magnitude
        param_norm_delta = 0.0
        with torch.no_grad():
            for p in candidate_model.parameters():
                if p.grad is not None:
                    param_norm_delta += p.grad.norm().item()

        return ReasoningTrainingRunMetrics(
            learning_rate=lr,
            steps=steps,
            train_loss_initial=round(init_loss, 4),
            train_loss_final=round(final_loss, 4),
            train_reasoning_acc=round(train_acc, 4),
            held_out_reasoning_acc=round(held_acc, 4),
            runtime_seconds=round(elapsed, 2),
            parameter_norm_delta=round(param_norm_delta, 4),
        )
