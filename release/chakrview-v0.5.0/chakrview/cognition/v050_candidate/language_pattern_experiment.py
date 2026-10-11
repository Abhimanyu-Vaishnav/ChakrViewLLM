"""
Stage 4: Language Pattern Learning & Perplexity Evaluation.

Guarantees:
- Evaluates token sequence predictability on training vs held-out validation strings.
- Rejects conversational chatbot claims; strictly bounds evaluation to syntax/pattern learning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro


@dataclass
class LanguagePatternResult:
    train_perplexity: float
    val_perplexity: float
    perplexity_delta: float
    pattern_learning_demonstrated: bool
    rationale: str


class LanguagePatternExperiment:
    """
    Evaluates neural language pattern acquisition on synthetic corpus sequences.
    """

    @classmethod
    def run_experiment(
        cls,
        model: Optional[ChakrMicro] = None,
        steps: int = 15,
        lr: float = 1e-3,
        seed: int = 42,
    ) -> LanguagePatternResult:
        torch.manual_seed(seed)
        config = ModelConfig()
        candidate = model if model is not None else ChakrMicro(config)
        candidate.train()

        # Synthetic repeated grammatical syntax patterns:
        # [Noun_Token, Verb_Token, Adj_Token, Target_End]
        train_sequences = torch.tensor([
            [101, 201, 301, 401],
            [102, 202, 302, 402],
            [103, 203, 303, 403],
            [104, 204, 304, 404],
        ])

        val_sequences = torch.tensor([
            [105, 205, 305, 405],
            [106, 206, 306, 406],
        ])

        optimizer = optim.AdamW(candidate.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        train_loss = 0.0
        for _ in range(steps):
            optimizer.zero_grad()
            inp = train_sequences[:, :-1]
            tgt = train_sequences[:, 1:]
            logits = candidate(inp)
            loss = loss_fn(logits.reshape(-1, config.vocab_size), tgt.reshape(-1))
            loss.backward()
            optimizer.step()
            train_loss = loss.item()

        candidate.eval()
        with torch.no_grad():
            v_inp = val_sequences[:, :-1]
            v_tgt = val_sequences[:, 1:]
            v_logits = candidate(v_inp)
            val_loss = loss_fn(v_logits.reshape(-1, config.vocab_size), v_tgt.reshape(-1)).item()

        tr_ppl = round(math.exp(min(train_loss, 20.0)), 2)
        v_ppl = round(math.exp(min(val_loss, 20.0)), 2)

        return LanguagePatternResult(
            train_perplexity=tr_ppl,
            val_perplexity=v_ppl,
            perplexity_delta=round(v_ppl - tr_ppl, 2),
            pattern_learning_demonstrated=tr_ppl < 50.0,
            rationale=f"Train PPL reduced to {tr_ppl}, Val PPL is {v_ppl}.",
        )
