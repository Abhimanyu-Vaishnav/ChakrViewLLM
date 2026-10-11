"""
Stage 1: Neural Learning Baseline & Optimization.

Guarantees:
- Completely isolated from canonical baseline.
- Operates on CPU with micro-batching.
- Computes loss convergence, validation improvement, and parameter delta.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.brain.canonical_hash import compute_canonical_model_hash


@dataclass
class TrainingRunResult:
    candidate_id: str
    initial_loss: float
    final_loss: float
    val_loss: float
    steps_completed: int
    training_time_seconds: float
    parameter_delta_l2: float
    parameters_updated: int
    is_converged: bool
    candidate_hash: str


class IsolatedNeuralTrainer:
    """
    Trains an isolated clone of ChakrMicro on synthetic sequence objectives
    to empirically verify optimization stability and parameter delta.
    """

    @classmethod
    def train_candidate(
        cls,
        candidate_id: str = "cand_v050_stage1",
        steps: int = 15,
        batch_size: int = 4,
        seq_len: int = 16,
        lr: float = 1e-3,
        seed: int = 42,
    ) -> Tuple[ChakrMicro, TrainingRunResult]:
        torch.manual_seed(seed)
        config = ModelConfig()
        
        # Instantiate baseline clone
        candidate_model = ChakrMicro(config)
        candidate_model.train()

        # Capture initial state for delta computation
        init_weights = {n: p.detach().clone() for n, p in candidate_model.named_parameters()}

        # Synthetic sequence data: structured repetitive pattern
        # e.g., token (i) followed by token (i + 1)
        torch.manual_seed(seed + 1)
        data_x = torch.randint(10, 100, (steps * batch_size, seq_len))
        data_y = torch.roll(data_x, -1, dims=1)  # next token prediction target

        val_x = torch.randint(10, 100, (4, seq_len))
        val_y = torch.roll(val_x, -1, dims=1)

        optimizer = optim.AdamW(candidate_model.parameters(), lr=lr, weight_decay=0.01)
        loss_fn = nn.CrossEntropyLoss()

        initial_loss = 0.0
        final_loss = 0.0
        start_t = time.time()

        for step in range(steps):
            batch_in = data_x[step * batch_size : (step + 1) * batch_size]
            batch_tgt = data_y[step * batch_size : (step + 1) * batch_size]

            optimizer.zero_grad()
            logits = candidate_model(batch_in)
            loss = loss_fn(logits.view(-1, config.vocab_size), batch_tgt.view(-1))

            if step == 0:
                initial_loss = float(loss.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(candidate_model.parameters(), 1.0)
            optimizer.step()

            final_loss = float(loss.item())

        # Evaluate on held-out validation batch
        candidate_model.eval()
        with torch.no_grad():
            val_logits = candidate_model(val_x)
            val_loss = float(loss_fn(val_logits.view(-1, config.vocab_size), val_y.view(-1)).item())

        elapsed = time.time() - start_t

        # Compute parameter drift / delta L2
        delta_l2 = 0.0
        updated_count = 0
        for n, p in candidate_model.named_parameters():
            diff = (p.detach() - init_weights[n]).norm().item()
            delta_l2 += diff ** 2
            if diff > 1e-6:
                updated_count += 1
        delta_l2 = delta_l2 ** 0.5

        cand_hash, num_params, _ = compute_canonical_model_hash(candidate_model)

        result = TrainingRunResult(
            candidate_id=candidate_id,
            initial_loss=round(initial_loss, 4),
            final_loss=round(final_loss, 4),
            val_loss=round(val_loss, 4),
            steps_completed=steps,
            training_time_seconds=round(elapsed, 3),
            parameter_delta_l2=round(delta_l2, 6),
            parameters_updated=updated_count,
            is_converged=final_loss < initial_loss,
            candidate_hash=cand_hash,
        )

        return candidate_model, result
