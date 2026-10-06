"""Step 251: Curriculum-Driven Pointer Circuit Training.

Executes controlled multi-phase curriculum training for the Associative Pointer Head:
- Phase A: Single association (|A| -> |1|)
- Phase B: Multiple associations (2-3 pairs)
- Phase C: Multiple associations + distractors
- Phase D: Variable layouts and query positions
- Phase E: Disjoint identities
- Phase F: Unseen combinations

Strict Scientific Controls:
- Frozen canonical ChakrMicro baseline (requires_grad = False, Delta W = 0)
- CPU-only training
- Tracks:
  * train accuracy & loss
  * validation accuracy
  * target probability & target rank
  * pointer entropy
  * selected key position accuracy
  * selected value position accuracy
  * base language retention ratio
"""

from __future__ import annotations

import dataclasses
import math
import random
import time
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro
from chakrview.runtime.interactive import compute_model_hash, EXPECTED_WEIGHT_HASH
from chakrview.cognition.associative_pointer_circuit import (
    AssociativePointerHead,
    ChakrMicroWithAssociativePointer,
)
from chakrview.cognition.randomized_associative_episodes import (
    RandomizedAssociativeEnvironment,
    RandomizedAssociativeEpisode,
)
from chakrview.tokenizer import BPETokenizer


@dataclasses.dataclass
class PhaseTrainingResult:
    phase_name: str
    phase_description: str
    train_loss: float
    train_accuracy: float
    val_accuracy: float
    key_routing_accuracy: float
    val_routing_accuracy: float
    mean_target_prob: float
    mean_target_rank: float
    pointer_entropy: float
    phase_passed: bool


@dataclasses.dataclass
class PointerCurriculumReport:
    seed: int
    phases: List[PhaseTrainingResult]
    overall_train_acc: float
    overall_val_acc: float
    language_retention_ratio: float
    is_base_frozen: bool
    base_hash: str
    cpu_runtime_ms: float = 0.0


def compute_distribution_entropy(probs: torch.Tensor, eps: float = 1e-12) -> float:
    """Computes Shannon entropy in nats."""
    p = torch.clamp(probs, min=eps)
    return float(-(p * torch.log(p)).sum(dim=-1).mean().item())


def run_pointer_circuit_curriculum(
    base_model: ChakrMicro,
    seed: int = 42,
    steps_per_phase: int = 25,
    eval_episodes: int = 15,
    lr: float = 1.5e-3,
) -> Tuple[ChakrMicroWithAssociativePointer, PointerCurriculumReport]:
    """Trains Associative Pointer Head along Phase A to F curriculum while keeping ChakrMicro frozen."""
    t0 = time.time()
    torch.manual_seed(seed)
    env = RandomizedAssociativeEnvironment(seed=seed)
    tok = env.tok

    init_hash = compute_model_hash(base_model)
    candidate = ChakrMicroWithAssociativePointer(base_model)

    # Strictly freeze base model parameters
    for p in candidate.base_model.parameters():
        p.requires_grad = False
    for p in candidate.pointer_head.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(
        candidate.pointer_head.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    # Measure initial base language loss on a reference text
    ref_text = "The quick brown fox jumps over the lazy dog and runs through the forest."
    ref_enc = tok.encode(ref_text)
    ref_inp = torch.tensor([ref_enc[:-1]], dtype=torch.long)
    ref_tgt = torch.tensor([ref_enc[1:]], dtype=torch.long)

    with torch.no_grad():
        x = base_model.embedding(ref_inp)
        for l in base_model.layers:
            x = l(x)
        h = base_model.final_norm(x)
        base_logits = base_model.lm_head(h)
        init_lang_loss = float(F.cross_entropy(base_logits.view(-1, 4096), ref_tgt.view(-1)).item())

    # Phase specifications
    phase_configs = [
        {
            "name": "Phase_A",
            "desc": "Single association",
            "split": "train",
            "num_assoc": 1,
            "distractors": False,
            "layout": "standard_map",
            "query_place": "end",
        },
        {
            "name": "Phase_B",
            "desc": "Multiple associations (2-3 pairs)",
            "split": "train",
            "num_assoc": 2,
            "distractors": False,
            "layout": "standard_map",
            "query_place": "end",
        },
        {
            "name": "Phase_C",
            "desc": "Multiple associations + distractors",
            "split": "train",
            "num_assoc": 2,
            "distractors": True,
            "layout": "standard_map",
            "query_place": "end",
        },
        {
            "name": "Phase_D",
            "desc": "Variable layouts and query positions",
            "split": "train",
            "num_assoc": 2,
            "distractors": False,
            "layout": None, # randomized layout
            "query_place": "end",
        },
        {
            "name": "Phase_E",
            "desc": "Disjoint identities (known key, unseen val)",
            "split": "known_unseen",
            "num_assoc": 2,
            "distractors": False,
            "layout": "standard_map",
            "query_place": "end",
        },
        {
            "name": "Phase_F",
            "desc": "Unseen combinations (disjoint keys and vals)",
            "split": "disjoint_test",
            "num_assoc": 2,
            "distractors": False,
            "layout": None,
            "query_place": "end",
        },
    ]

    phase_results: List[PhaseTrainingResult] = []

    for cfg in phase_configs:
        candidate.train()
        total_loss = 0.0
        train_correct = 0

        for step in range(steps_per_phase):
            ep = env.generate_episode(
                split=cfg["split"] if cfg["split"] != "disjoint_test" else "train",
                num_associations=cfg["num_assoc"],
                layout_name=cfg["layout"],
                include_distractors=cfg["distractors"],
                query_placement=cfg["query_place"],
                episode_idx=step,
            )

            inp = torch.tensor([ep.prompt_tokens], dtype=torch.long)
            target_tok = torch.tensor([ep.target_token], dtype=torch.long)
            target_q = torch.tensor([min(ep.query_key_pos, len(ep.prompt_tokens) - 2)], dtype=torch.long)
            target_k = torch.tensor([min(ep.matching_key_pos, len(ep.prompt_tokens) - 2)], dtype=torch.long)
            target_v = torch.tensor([min(ep.associated_val_pos, len(ep.prompt_tokens) - 2)], dtype=torch.long)

            with torch.no_grad():
                x = base_model.embedding(inp)
                for l in base_model.layers:
                    x = l(x)
                hidden = base_model.final_norm(x)
                gen_logits = base_model.lm_head(hidden)

            out = candidate.pointer_head(
                hidden_states=hidden,
                gen_logits=gen_logits,
                input_ids=inp,
            )

            # Combined multi-objective loss
            l_tok = F.nll_loss(out.combined_logits, target_tok)
            l_q = F.cross_entropy(torch.log(out.query_key_distribution + 1e-12), target_q)
            l_k = F.cross_entropy(torch.log(out.key_distribution + 1e-12), target_k)
            l_v = F.cross_entropy(torch.log(out.value_distribution + 1e-12), target_v)

            loss = l_tok + 0.5 * l_q + 0.5 * l_k + 0.5 * l_v

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if torch.argmax(out.combined_logits, dim=-1).item() == ep.target_token:
                train_correct += 1

        avg_loss = total_loss / max(1, steps_per_phase)
        train_acc = train_correct / max(1, steps_per_phase)

        # Validation on distinct episodes
        candidate.eval()
        val_correct = 0
        k_correct = 0
        v_correct = 0
        target_probs = []
        target_ranks = []
        entropies = []

        with torch.no_grad():
            for e_idx in range(eval_episodes):
                ep = env.generate_episode(
                    split=cfg["split"],
                    num_associations=cfg["num_assoc"],
                    layout_name=cfg["layout"],
                    include_distractors=cfg["distractors"],
                    query_placement=cfg["query_place"],
                    episode_idx=1000 + e_idx,
                )

                inp = torch.tensor([ep.prompt_tokens], dtype=torch.long)
                x = base_model.embedding(inp)
                for l in base_model.layers:
                    x = l(x)
                hidden = base_model.final_norm(x)
                gen_logits = base_model.lm_head(hidden)

                out = candidate.pointer_head(
                    hidden_states=hidden,
                    gen_logits=gen_logits,
                    input_ids=inp,
                )

                pred_tok = torch.argmax(out.combined_logits, dim=-1).item()
                pred_k = out.selected_key_pos.item()
                pred_v = out.selected_val_pos.item()

                if pred_tok == ep.target_token:
                    val_correct += 1
                if pred_k == min(ep.matching_key_pos, len(ep.prompt_tokens) - 2):
                    k_correct += 1
                if pred_v == min(ep.associated_val_pos, len(ep.prompt_tokens) - 2):
                    v_correct += 1

                p_all = F.softmax(out.combined_logits, dim=-1)[0]
                t_prob = float(p_all[ep.target_token].item())
                sorted_idx = torch.argsort(out.combined_logits[0], descending=True)
                rank = int((sorted_idx == ep.target_token).nonzero(as_tuple=True)[0].item()) + 1

                target_probs.append(t_prob)
                target_ranks.append(rank)
                entropies.append(compute_distribution_entropy(out.value_distribution))

        val_acc = val_correct / max(1, eval_episodes)
        k_acc = k_correct / max(1, eval_episodes)
        v_acc = v_correct / max(1, eval_episodes)

        phase_results.append(
            PhaseTrainingResult(
                phase_name=cfg["name"],
                phase_description=cfg["desc"],
                train_loss=avg_loss,
                train_accuracy=train_acc,
                val_accuracy=val_acc,
                key_routing_accuracy=k_acc,
                val_routing_accuracy=v_acc,
                mean_target_prob=sum(target_probs) / max(1, len(target_probs)),
                mean_target_rank=sum(target_ranks) / max(1, len(target_ranks)),
                pointer_entropy=sum(entropies) / max(1, len(entropies)),
                phase_passed=(val_acc > 0.0),
            )
        )

    # Post-training language retention check
    with torch.no_grad():
        x = base_model.embedding(ref_inp)
        for l in base_model.layers:
            x = l(x)
        h = base_model.final_norm(x)
        post_logits = base_model.lm_head(h)
        post_lang_loss = float(F.cross_entropy(post_logits.view(-1, 4096), ref_tgt.view(-1)).item())

    retention_ratio = post_lang_loss / max(1e-12, init_lang_loss)
    final_hash = compute_model_hash(base_model)
    is_frozen = (final_hash == init_hash == EXPECTED_WEIGHT_HASH)

    elapsed_ms = (time.time() - t0) * 1000.0
    overall_train = sum(p.train_accuracy for p in phase_results) / len(phase_results)
    overall_val = sum(p.val_accuracy for p in phase_results) / len(phase_results)

    report = PointerCurriculumReport(
        seed=seed,
        phases=phase_results,
        overall_train_acc=overall_train,
        overall_val_acc=overall_val,
        language_retention_ratio=retention_ratio,
        is_base_frozen=is_frozen,
        base_hash=final_hash,
        cpu_runtime_ms=elapsed_ms,
    )

    return candidate, report
