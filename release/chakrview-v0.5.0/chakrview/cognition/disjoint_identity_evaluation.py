"""Step 254: Disjoint Identity Generalization & I3 Gate Evaluation.

Core capability evaluation across 4 orthogonal identity splits:
A. known key / known value (in-distribution generalization)
B. known key / unseen value (value generalization)
C. unseen key / known value (key generalization)
D. unseen key / unseen value (primary I3 capability gate)

Determinism & Multi-seed Requirement:
Runs across at least 3 deterministic seeds:
- 42
- 101
- 2026

Promotion Target:
D >= 0.50 (50.0%) across all seeds for I3_CANDIDATE_ACHIEVED.
Otherwise:
- 0 < D < 0.50: I3_EMERGING
- D == 0.0: I3_NOT_ACHIEVED
"""

from __future__ import annotations

import dataclasses
import random
import time
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro
from chakrview.runtime.interactive import compute_model_hash, EXPECTED_WEIGHT_HASH
from chakrview.cognition.associative_pointer_circuit import (
    ChakrMicroWithAssociativePointer,
)
from chakrview.cognition.randomized_associative_episodes import (
    RandomizedAssociativeEnvironment,
    RandomizedAssociativeEpisode,
)


@dataclasses.dataclass
class DisjointSplitResult:
    split_name: str
    num_episodes: int
    token_accuracy: float
    key_position_accuracy: float
    val_position_accuracy: float
    mean_target_prob: float
    mean_target_rank: float


@dataclasses.dataclass
class SeedDisjointEvaluationResult:
    seed: int
    splits: Dict[str, DisjointSplitResult]
    unseen_unseen_acc: float
    unseen_unseen_val_pos_acc: float
    unseen_unseen_key_pos_acc: float


@dataclasses.dataclass
class DisjointGeneralizationReport:
    seeds_tested: List[int]
    per_seed_results: Dict[int, SeedDisjointEvaluationResult]
    mean_known_known_acc: float
    mean_known_unseen_acc: float
    mean_unseen_known_acc: float
    mean_unseen_unseen_acc: float
    mean_unseen_unseen_val_pos_acc: float
    mean_unseen_unseen_key_pos_acc: float
    i3_status: str               # "I3_CANDIDATE_ACHIEVED", "I3_EMERGING", "I3_NOT_ACHIEVED"
    i3_promoted: bool
    is_base_immutable: bool
    base_hash: str
    cpu_runtime_ms: float = 0.0


def run_disjoint_identity_generalization(
    base_model: ChakrMicro,
    seeds: Optional[List[int]] = None,
    train_steps_per_seed: int = 40,
    eval_episodes_per_split: int = 15,
    lr: float = 1.5e-3,
) -> DisjointGeneralizationReport:
    """Trains Associative Pointer Head on train split, then tests on 4 orthogonal splits across seeds."""
    if seeds is None:
        seeds = [42, 101, 2026]

    t0 = time.time()
    init_hash = compute_model_hash(base_model)
    per_seed_results: Dict[int, SeedDisjointEvaluationResult] = {}

    splits_to_eval = [
        ("known_known", "train"),
        ("known_unseen", "known_unseen"),
        ("unseen_known", "unseen_known"),
        ("unseen_unseen", "disjoint_test"),
    ]

    for s in seeds:
        torch.manual_seed(s)
        env = RandomizedAssociativeEnvironment(seed=s)

        candidate = ChakrMicroWithAssociativePointer(base_model)
        for p in candidate.base_model.parameters():
            p.requires_grad = False
        for p in candidate.pointer_head.parameters():
            p.requires_grad = True

        optimizer = torch.optim.AdamW(candidate.pointer_head.parameters(), lr=lr, weight_decay=1e-4)

        # Train on train split
        candidate.train()
        for step in range(train_steps_per_seed):
            ep = env.generate_episode(
                split="train",
                num_associations=2,
                layout_name=None,
                include_distractors=False,
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

            l_tok = F.nll_loss(out.combined_logits, target_tok)
            l_q = F.cross_entropy(torch.log(out.query_key_distribution + 1e-12), target_q)
            l_k = F.cross_entropy(torch.log(out.key_distribution + 1e-12), target_k)
            l_v = F.cross_entropy(torch.log(out.value_distribution + 1e-12), target_v)

            loss = l_tok + 0.5 * l_q + 0.5 * l_k + 0.5 * l_v

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate across the 4 splits
        candidate.eval()
        split_evals: Dict[str, DisjointSplitResult] = {}

        with torch.no_grad():
            for sp_key, sp_env in splits_to_eval:
                tok_corr = 0
                k_corr = 0
                v_corr = 0
                probs = []
                ranks = []

                for e_idx in range(eval_episodes_per_split):
                    ep = env.generate_episode(
                        split=sp_env,
                        num_associations=2,
                        layout_name=None,
                        include_distractors=False,
                        episode_idx=5000 + e_idx,
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

                    gt_tok = ep.target_token
                    gt_k = min(ep.matching_key_pos, len(ep.prompt_tokens) - 2)
                    gt_v = min(ep.associated_val_pos, len(ep.prompt_tokens) - 2)

                    if pred_tok == gt_tok:
                        tok_corr += 1
                    if pred_k == gt_k:
                        k_corr += 1
                    if pred_v == gt_v:
                        v_corr += 1

                    p_all = F.softmax(out.combined_logits, dim=-1)[0]
                    probs.append(float(p_all[gt_tok].item()))
                    sorted_idx = torch.argsort(out.combined_logits[0], descending=True)
                    rk = int((sorted_idx == gt_tok).nonzero(as_tuple=True)[0].item()) + 1
                    ranks.append(rk)

                split_evals[sp_key] = DisjointSplitResult(
                    split_name=sp_key,
                    num_episodes=eval_episodes_per_split,
                    token_accuracy=tok_corr / max(1, eval_episodes_per_split),
                    key_position_accuracy=k_corr / max(1, eval_episodes_per_split),
                    val_position_accuracy=v_corr / max(1, eval_episodes_per_split),
                    mean_target_prob=sum(probs) / max(1, len(probs)),
                    mean_target_rank=sum(ranks) / max(1, len(ranks)),
                )

        uu = split_evals["unseen_unseen"]
        per_seed_results[s] = SeedDisjointEvaluationResult(
            seed=s,
            splits=split_evals,
            unseen_unseen_acc=uu.token_accuracy,
            unseen_unseen_val_pos_acc=uu.val_position_accuracy,
            unseen_unseen_key_pos_acc=uu.key_position_accuracy,
        )

    # Compute aggregate means across seeds
    mean_kk = sum(r.splits["known_known"].token_accuracy for r in per_seed_results.values()) / len(per_seed_results)
    mean_ku = sum(r.splits["known_unseen"].token_accuracy for r in per_seed_results.values()) / len(per_seed_results)
    mean_uk = sum(r.splits["unseen_known"].token_accuracy for r in per_seed_results.values()) / len(per_seed_results)
    mean_uu = sum(r.unseen_unseen_acc for r in per_seed_results.values()) / len(per_seed_results)
    mean_uu_val = sum(r.unseen_unseen_val_pos_acc for r in per_seed_results.values()) / len(per_seed_results)
    mean_uu_key = sum(r.unseen_unseen_key_pos_acc for r in per_seed_results.values()) / len(per_seed_results)

    # All seeds must individually achieve >= 0.50 for promotion
    all_seeds_pass = all(r.unseen_unseen_acc >= 0.50 for r in per_seed_results.values())

    if mean_uu >= 0.50 and all_seeds_pass:
        i3_status = "I3_CANDIDATE_ACHIEVED"
        promoted = True
    elif mean_uu > 0.0:
        i3_status = "I3_EMERGING"
        promoted = False
    else:
        i3_status = "I3_NOT_ACHIEVED"
        promoted = False

    final_hash = compute_model_hash(base_model)
    is_base_clean = (final_hash == init_hash == EXPECTED_WEIGHT_HASH)
    elapsed_ms = (time.time() - t0) * 1000.0

    return DisjointGeneralizationReport(
        seeds_tested=seeds,
        per_seed_results=per_seed_results,
        mean_known_known_acc=mean_kk,
        mean_known_unseen_acc=mean_ku,
        mean_unseen_known_acc=mean_uk,
        mean_unseen_unseen_acc=mean_uu,
        mean_unseen_unseen_val_pos_acc=mean_uu_val,
        mean_unseen_unseen_key_pos_acc=mean_uu_key,
        i3_status=i3_status,
        i3_promoted=promoted,
        is_base_immutable=is_base_clean,
        base_hash=final_hash,
        cpu_runtime_ms=elapsed_ms,
    )
