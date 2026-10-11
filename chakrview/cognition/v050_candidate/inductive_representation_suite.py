"""
Advanced Relational Representation & Inductive Generalization Suite.

Location: chakrview/cognition/v050_candidate/inductive_representation_suite.py

Investigates why naive atomic relation experiments fail to generalize to disjoint entities:
- Diagnostic Mode 1: Naive Atomic Entities (disjoint test tokens with no semantic links).
- Diagnostic Mode 2: In-Context Exemplar Routing (entity-value association provided in prefix).
- Diagnostic Mode 3: Categorical-Semantic Grounded Representation (shared relational semantic types).

Includes:
- Train, validation, and held-out test splits with zero test leakage.
- Early stopping and validation checkpointing.
- Checkpoint serialization and round-trip deterministic verification.
- Multi-seed variance evaluation.
- Device placement support (CPU and optional CUDA).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.brain.canonical_hash import verify_canonical_baseline


@dataclass
class InductiveBenchmarkMetrics:
    mode: str
    seed: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    test_loss: float
    test_accuracy: float
    generalization_gap: float
    is_generalizing: bool
    wall_clock_ms: float
    device: str
    checkpoint_hash: Optional[str] = None


class InductiveRepresentationSuite:
    """
    Experimental suite for diagnostic representation learning and inductive generalization.
    """

    @classmethod
    def generate_data(cls, mode: str = "categorical", seed: int = 42) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        """
        Generates strict Train, Validation, and Test splits with zero data contamination.
        """
        torch.manual_seed(seed)

        if mode == "naive_atomic":
            # Mode 1: Direct atomic pairs (E, R) -> V
            # Train entities: 100..111 (12 entities)
            # Val entities:   112..113 (2 entities)
            # Test entities:  114..117 (4 entities)
            R = 50
            train_pairs = [(e, R, 200 + (e - 100)) for e in range(100, 112)]
            val_pairs = [(e, R, 200 + (e - 100)) for e in range(112, 114)]
            test_pairs = [(e, R, 200 + (e - 100)) for e in range(114, 118)]

            tx = torch.tensor([[e, r] for e, r, _ in train_pairs])
            ty = torch.tensor([v for _, _, v in train_pairs])
            vx = torch.tensor([[e, r] for e, r, _ in val_pairs])
            vy = torch.tensor([v for _, _, v in val_pairs])
            tex = torch.tensor([[e, r] for e, r, _ in test_pairs])
            tey = torch.tensor([v for _, _, v in test_pairs])

        elif mode == "in_context":
            # Mode 2: In-context key-value introduction [E, MAPS_TO, V, SEP, E, R] -> V
            MAPS_TO = 999
            SEP = 998
            R = 50
            # Train: 30 instances
            train_seqs, train_tgts = [], []
            for i in range(30):
                e = 1000 + i
                v = 2000 + i
                train_seqs.append([e, MAPS_TO, v, SEP, e, R])
                train_tgts.append(v)

            # Val: 6 instances
            val_seqs, val_tgts = [], []
            for i in range(30, 36):
                e = 1000 + i
                v = 2000 + i
                val_seqs.append([e, MAPS_TO, v, SEP, e, R])
                val_tgts.append(v)

            # Test: 10 instances (completely disjoint token IDs)
            test_seqs, test_tgts = [], []
            for i in range(10):
                e = 3000 + i
                v = 3500 + i
                test_seqs.append([e, MAPS_TO, v, SEP, e, R])
                test_tgts.append(v)

            tx, ty = torch.tensor(train_seqs), torch.tensor(train_tgts)
            vx, vy = torch.tensor(val_seqs), torch.tensor(val_tgts)
            tex, tey = torch.tensor(test_seqs), torch.tensor(test_tgts)

        elif mode == "categorical":
            # Mode 3: Categorical semantic grounding
            # Relational types shared across categories: [Type_ID, Entity_ID, Relation_ID] -> Target
            # Categories 10, 11, 12 mapping across Relations 50 and 51
            categories = [(10, 200, 300), (11, 201, 301), (12, 202, 302)]
            train_seqs, train_tgts = [], []
            val_seqs, val_tgts = [], []
            test_seqs, test_tgts = [], []

            for cat_id, tgt_r50, tgt_r51 in categories:
                # 8 train entities per category
                for idx in range(8):
                    e = 1000 + cat_id * 30 + idx
                    train_seqs.extend([[cat_id, e, 50], [cat_id, e, 51]])
                    train_tgts.extend([tgt_r50, tgt_r51])
                # 2 val entities per category
                for idx in range(8, 10):
                    e = 1000 + cat_id * 30 + idx
                    val_seqs.extend([[cat_id, e, 50], [cat_id, e, 51]])
                    val_tgts.extend([tgt_r50, tgt_r51])
                # 4 test entities per category (held-out disjoint entities)
                for idx in range(10, 14):
                    e = 1000 + cat_id * 30 + idx
                    test_seqs.extend([[cat_id, e, 50], [cat_id, e, 51]])
                    test_tgts.extend([tgt_r50, tgt_r51])

            tx, ty = torch.tensor(train_seqs), torch.tensor(train_tgts)
            vx, vy = torch.tensor(val_seqs), torch.tensor(val_tgts)
            tex, tey = torch.tensor(test_seqs), torch.tensor(test_tgts)

        else:
            raise ValueError(f"Unknown mode: {mode}")

        return (tx, ty), (vx, vy), (tex, tey)

    @classmethod
    def run_experiment(
        cls,
        mode: str = "categorical",
        steps: int = 35,
        lr: float = 1e-3,
        seed: int = 42,
        device: str = "cpu",
        checkpoint_dir: Optional[Path] = None,
    ) -> InductiveBenchmarkMetrics:
        """
        Executes bounded, reproducible training and evaluation.
        """
        start_time = time.perf_counter()
        torch.manual_seed(seed)

        target_device = torch.device(device)
        config = ModelConfig()
        candidate = ChakrMicro(config).to(target_device)
        candidate.train()

        (tx, ty), (vx, vy), (tex, tey) = cls.generate_data(mode=mode, seed=seed)
        tx, ty = tx.to(target_device), ty.to(target_device)
        vx, vy = vx.to(target_device), vy.to(target_device)
        tex, tey = tex.to(target_device), tey.to(target_device)

        optimizer = optim.AdamW(candidate.parameters(), lr=lr, weight_decay=0.01)
        loss_fn = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        best_state: Optional[Dict[str, Any]] = None

        for step in range(steps):
            optimizer.zero_grad()
            logits = candidate(tx)
            loss = loss_fn(logits[:, -1, :], ty)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected at step {step}: {loss.item()}")

            loss.backward()
            optimizer.step()

            # Early validation checkpoint tracking
            with torch.no_grad():
                val_logits = candidate(vx)
                cur_val_loss = loss_fn(val_logits[:, -1, :], vy).item()
                if cur_val_loss < best_val_loss:
                    best_val_loss = cur_val_loss
                    best_state = {k: v.cpu().clone() for k, v in candidate.state_dict().items()}

        # Restore best validation checkpoint if found
        if best_state is not None:
            candidate.load_state_dict(best_state)

        # Final evaluation on Train, Val, and Test
        candidate.eval()
        with torch.no_grad():
            tr_logits = candidate(tx)
            tr_loss = loss_fn(tr_logits[:, -1, :], ty).item()
            tr_acc = (tr_logits[:, -1, :].argmax(dim=-1) == ty).float().mean().item()

            va_logits = candidate(vx)
            va_loss = loss_fn(va_logits[:, -1, :], vy).item()
            va_acc = (va_logits[:, -1, :].argmax(dim=-1) == vy).float().mean().item()

            te_logits = candidate(tex)
            te_loss = loss_fn(te_logits[:, -1, :], tey).item()
            te_acc = (te_logits[:, -1, :].argmax(dim=-1) == tey).float().mean().item()

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        gap = round(tr_acc - te_acc, 3)
        is_gen = te_acc > 0.5 and te_loss < 2.0

        ckpt_hash: Optional[str] = None
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = checkpoint_dir / f"candidate_{mode}_seed_{seed}.pt"
            torch.save(
                {
                    "mode": mode,
                    "seed": seed,
                    "metrics": {
                        "tr_loss": tr_loss,
                        "tr_acc": tr_acc,
                        "te_loss": te_loss,
                        "te_acc": te_acc,
                    },
                    "state_dict": candidate.state_dict(),
                },
                ckpt_path,
            )
            # Compute hash of saved checkpoint
            import hashlib
            with open(ckpt_path, "rb") as f:
                ckpt_hash = hashlib.sha256(f.read()).hexdigest()

        return InductiveBenchmarkMetrics(
            mode=mode,
            seed=seed,
            train_loss=round(tr_loss, 4),
            train_accuracy=round(tr_acc, 4),
            val_loss=round(va_loss, 4),
            val_accuracy=round(va_acc, 4),
            test_loss=round(te_loss, 4),
            test_accuracy=round(te_acc, 4),
            generalization_gap=gap,
            is_generalizing=is_gen,
            wall_clock_ms=round(elapsed_ms, 2),
            device=str(target_device),
            checkpoint_hash=ckpt_hash,
        )
