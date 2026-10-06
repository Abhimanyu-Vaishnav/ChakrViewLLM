"""
ChakrView Step 147: Neural Reasoning Curriculum & Pattern Generalization.

Features:
- Generates synthetic reasoning tasks with structural variations:
  - Transitive orderings: (X > Y, Y > Z => X > Z)
  - Variable substitutions avoiding exact lexical memorization
  - Held-out composition tests evaluating structural pattern learning
- Evaluates baseline vs candidate on structural pattern accuracy.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.tokenizer import BPETokenizer
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.tokenizer.special_tokens import PAD_ID


@dataclass
class NeuralReasoningResult:
    baseline_train_acc: float
    candidate_train_acc: float
    baseline_heldout_acc: float
    candidate_heldout_acc: float
    heldout_reasoning_delta: float
    pattern_learned: bool


class NeuralReasoningCurriculumTrainer:
    """
    Curriculum generator and evaluator for transitive and compositional neural reasoning patterns.
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

    def generate_reasoning_samples(self, seed: int = 42) -> Tuple[List[str], List[str]]:
        """Generates training relations and structurally distinct held-out composition relations."""
        rng = random.Random(seed)
        symbols_train = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
        symbols_heldout = ["phi", "chi", "psi", "omega", "sigma", "theta"]

        train_texts = []
        for _ in range(12):
            a, b, c = rng.sample(symbols_train, 3)
            # A > B, B > C => A > C
            text = f"fact: {a} > {b} and {b} > {c} . therefore {a} > {c}"
            train_texts.append(text)

        heldout_texts = []
        for _ in range(6):
            x, y, z = rng.sample(symbols_heldout, 3)
            text = f"fact: {x} > {y} and {y} > {z} . therefore {x} > {z}"
            heldout_texts.append(text)

        return train_texts, heldout_texts

    def evaluate_reasoning_accuracy(self, model: ChakrMicro, texts: List[str]) -> float:
        model.eval()
        correct = 0
        total = 0

        for t in texts:
            # Predict the token right after 'therefore '
            prefix, expected = t.split(". therefore ")
            prompt = prefix + ". therefore "
            target_symbol = expected.split(" > ")[0]

            p_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
            t_ids = self.tokenizer.encode(target_symbol, add_bos=False, add_eos=False)
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

    def train_and_evaluate_reasoning(
        self,
        candidate_model: ChakrMicro,
        baseline_model: ChakrMicro,
        steps: int = 15,
        lr: float = 5e-4,
    ) -> NeuralReasoningResult:
        train_texts, heldout_texts = self.generate_reasoning_samples()

        # Tokenize train samples
        train_tensors = []
        for t in train_texts:
            ids = self.tokenizer.encode(t, add_bos=True, add_eos=True)
            if len(ids) < 32:
                ids = ids + [PAD_ID] * (32 - len(ids))
            else:
                ids = ids[:32]
            train_tensors.append(torch.tensor(ids, dtype=torch.long, device=self.device))

        # Baseline scores
        base_train_acc = self.evaluate_reasoning_accuracy(baseline_model, train_texts)
        base_held_acc = self.evaluate_reasoning_accuracy(baseline_model, heldout_texts)

        # Train candidate
        candidate_model.train()
        optimizer = torch.optim.AdamW(candidate_model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)

        for s in range(steps):
            seq = train_tensors[s % len(train_tensors)]
            inp = seq[:-1].unsqueeze(0)
            tgt = seq[1:].unsqueeze(0)

            optimizer.zero_grad()
            logits = candidate_model(inp)
            loss = loss_fn(logits.view(-1, logits.size(-1)), tgt.view(-1))
            loss.backward()
            optimizer.step()

        # Candidate scores
        cand_train_acc = self.evaluate_reasoning_accuracy(candidate_model, train_texts)
        cand_held_acc = self.evaluate_reasoning_accuracy(candidate_model, heldout_texts)

        delta = round(cand_held_acc - base_held_acc, 4)
        return NeuralReasoningResult(
            baseline_train_acc=round(base_train_acc, 4),
            candidate_train_acc=round(cand_train_acc, 4),
            baseline_heldout_acc=round(base_held_acc, 4),
            candidate_heldout_acc=round(cand_held_acc, 4),
            heldout_reasoning_delta=delta,
            pattern_learned=(cand_train_acc >= base_train_acc),
        )
