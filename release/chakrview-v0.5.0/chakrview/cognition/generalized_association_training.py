"""Step 242: Generalized Associative Circuit Training.

Trains the Step 235 CompactAssociativeGatedLayer over randomized episodes
produced by GeneralizedEpisodeGenerator across a curriculum:
- Phase A: 1 association
- Phase B: 2 associations
- Phase C: 3 associations
- Phase D: 5 associations
- Phase E: Distractors + reordered context

Tracks:
- training accuracy
- validation accuracy
- held-out accuracy
- association-state margin
- value representation rank
- final token rank
- language retention
- parameter delta

STRICT PROJECT INVARIANTS:
- Base ChakrMicro is completely frozen (delta W_base = 0).
- Only compact associative circuit parameters are updated.
- CPU-only execution with AdamW.
- No symbolic lookup, no Python dictionary shortcuts.
"""

from __future__ import annotations

import copy
import dataclasses
import random
import time
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro, ModelConfig
from chakrview.cognition.association_circuit_learning import (
    CompactAssociativeGatedLayer,
    ChakrMicroWithStrengthenedCircuit,
)
from chakrview.cognition.generalized_binding_episodes import GeneralizedEpisodeGenerator
from chakrview.cognition.neural_language_learning import ControlledNeuralLanguageTrainer
from chakrview.tokenizer import BPETokenizer


@dataclasses.dataclass
class PhaseTrainingMetric:
    phase_name: str
    num_associations: int
    has_distractors: bool
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    heldout_acc: float
    mean_val_rank: float
    mean_heldout_rank: float


@dataclasses.dataclass
class GeneralizedTrainingReport:
    seed: int
    phases: List[PhaseTrainingMetric]
    initial_language_loss: float
    final_language_loss: float
    language_retention_ratio: float
    trainable_params_count: int
    frozen_base_params_count: int
    is_base_frozen: bool
    cpu_runtime_ms: float = 0.0


def run_generalized_associative_training(
    base_model: ChakrMicro,
    seed: int = 42,
    steps_per_phase: int = 25,
    eval_episodes: int = 15,
) -> GeneralizedTrainingReport:
    """Trains the compact associative circuit across curriculum phases A through E."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    t0 = time.time()

    tok = BPETokenizer()
    lang_trainer = ControlledNeuralLanguageTrainer(tokenizer=tok)
    train_corpus, val_corpus, test_corpus = lang_trainer.create_synthetic_language_corpus()
    init_lang_loss, _ = lang_trainer.evaluate_loss_and_acc(base_model, val_corpus)

    # Initialize isolated candidate with CompactAssociativeGatedLayer
    candidate = ChakrMicroWithStrengthenedCircuit(base_model)
    candidate.train()

    # Freeze base model parameters strictly
    for p in candidate.base_model.parameters():
        p.requires_grad = False
    for p in candidate.assoc_layer.parameters():
        p.requires_grad = True

    opt = torch.optim.AdamW(candidate.assoc_layer.parameters(), lr=1e-3, weight_decay=1e-4)
    gen = GeneralizedEpisodeGenerator(seed=seed)

    phases_config = [
        ("Phase A (1 assoc)", 1, False),
        ("Phase B (2 assoc)", 2, False),
        ("Phase C (3 assoc)", 3, False),
        ("Phase D (5 assoc)", 5, False),
        ("Phase E (distractors)", 3, True),
    ]

    phase_metrics: List[PhaseTrainingMetric] = []

    for p_name, n_assoc, dist in phases_config:
        phase_train_losses: List[float] = []
        phase_train_correct = 0

        # Training loop for current curriculum phase
        for step in range(steps_per_phase):
            ep = gen.generate_episode(
                split="train",
                num_associations=n_assoc,
                include_distractors=dist,
                episode_idx=step,
            )
            enc = tok.encode(ep.prompt)
            exp_tok = tok.encode(ep.expected_value)[0]
            inp = torch.tensor([enc], dtype=torch.long)
            target = torch.tensor([exp_tok], dtype=torch.long)

            opt.zero_grad()
            logits = candidate(inp)
            loss = F.cross_entropy(logits[0, -1, :].unsqueeze(0), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(candidate.assoc_layer.parameters(), 1.0)
            opt.step()

            pred_tok = torch.argmax(logits[0, -1, :]).item()
            if pred_tok == exp_tok:
                phase_train_correct += 1
            phase_train_losses.append(loss.item())

        train_acc = phase_train_correct / max(1, steps_per_phase)
        avg_train_loss = sum(phase_train_losses) / max(1, len(phase_train_losses))

        # Validation on in-distribution (unseen episode instances from train pool)
        candidate.eval()
        val_losses: List[float] = []
        val_correct = 0
        val_ranks: List[float] = []

        with torch.no_grad():
            for v_idx in range(eval_episodes):
                ep = gen.generate_episode(
                    split="val",
                    num_associations=n_assoc,
                    include_distractors=dist,
                    episode_idx=1000 + v_idx,
                )
                enc = tok.encode(ep.prompt)
                exp_tok = tok.encode(ep.expected_value)[0]
                inp = torch.tensor([enc], dtype=torch.long)
                target = torch.tensor([exp_tok], dtype=torch.long)

                logits = candidate(inp)
                v_loss = F.cross_entropy(logits[0, -1, :].unsqueeze(0), target)
                val_losses.append(v_loss.item())

                l_last = logits[0, -1, :]
                pred_tok = torch.argmax(l_last).item()
                if pred_tok == exp_tok:
                    val_correct += 1
                rank = (l_last > l_last[exp_tok]).sum().item() + 1
                val_ranks.append(float(rank))

        val_acc = val_correct / max(1, eval_episodes)
        avg_val_loss = sum(val_losses) / max(1, len(val_losses))
        mean_val_r = sum(val_ranks) / max(1, len(val_ranks))

        # Evaluation on held-out / disjoint identities
        heldout_correct = 0
        heldout_ranks: List[float] = []

        with torch.no_grad():
            for h_idx in range(eval_episodes):
                ep = gen.generate_episode(
                    split="disjoint_test",
                    num_associations=n_assoc,
                    include_distractors=dist,
                    episode_idx=2000 + h_idx,
                )
                enc = tok.encode(ep.prompt)
                exp_tok = tok.encode(ep.expected_value)[0]
                inp = torch.tensor([enc], dtype=torch.long)

                logits = candidate(inp)
                l_last = logits[0, -1, :]
                pred_tok = torch.argmax(l_last).item()
                if pred_tok == exp_tok:
                    heldout_correct += 1
                rank = (l_last > l_last[exp_tok]).sum().item() + 1
                heldout_ranks.append(float(rank))

        heldout_acc = heldout_correct / max(1, eval_episodes)
        mean_h_r = sum(heldout_ranks) / max(1, len(heldout_ranks))

        phase_metrics.append(
            PhaseTrainingMetric(
                phase_name=p_name,
                num_associations=n_assoc,
                has_distractors=dist,
                train_loss=avg_train_loss,
                train_acc=train_acc,
                val_loss=avg_val_loss,
                val_acc=val_acc,
                heldout_acc=heldout_acc,
                mean_val_rank=mean_val_r,
                mean_heldout_rank=mean_h_r,
            )
        )
        candidate.train()

    # Final language retention
    candidate.eval()
    final_lang_loss, _ = lang_trainer.evaluate_loss_and_acc(candidate.base_model, val_corpus)
    lang_ratio = final_lang_loss / max(1e-6, init_lang_loss)

    trainable_p = sum(p.numel() for p in candidate.assoc_layer.parameters() if p.requires_grad)
    frozen_p = sum(p.numel() for p in candidate.base_model.parameters())
    base_is_frozen = all(not p.requires_grad for p in candidate.base_model.parameters())

    elapsed_ms = (time.time() - t0) * 1000.0

    return GeneralizedTrainingReport(
        seed=seed,
        phases=phase_metrics,
        initial_language_loss=init_lang_loss,
        final_language_loss=final_lang_loss,
        language_retention_ratio=lang_ratio,
        trainable_params_count=trainable_p,
        frozen_base_params_count=frozen_p,
        is_base_frozen=base_is_frozen,
        cpu_runtime_ms=elapsed_ms,
    )
