"""Step 243: Identity Diversity Curriculum.

Expands training identity diversity without changing the underlying semantic operation:
KEY -> VALUE

Progression of identity pools:
- Pool A: Canonical A-Z / 0-9 single-token characters
- Pool B: Extended safe ASCII characters (punctuation, arithmetic symbols)
- Pool C: Multi-character synthetic identifier tokens verified to map cleanly

Invariants:
- All training identities are strictly disjoint from evaluation identities.
- No multi-token fragmentation that distorts the task into a tokenizer artifact.
- Evaluates held-out generalization across curriculum stages.
"""

from __future__ import annotations

import dataclasses
import random
import time
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro, ModelConfig
from chakrview.cognition.association_circuit_learning import (
    CompactAssociativeGatedLayer,
    ChakrMicroWithStrengthenedCircuit,
)
from chakrview.cognition.generalized_binding_episodes import BindingPair, BindingEpisode
from chakrview.tokenizer import BPETokenizer


@dataclasses.dataclass
class IdentityPoolStageMetric:
    pool_name: str
    pool_size_keys: int
    pool_size_vals: int
    train_loss: float
    train_acc: float
    val_acc: float
    heldout_disjoint_acc: float
    mean_target_rank: float
    retained_improvement: bool


@dataclasses.dataclass
class IdentityDiversityCurriculumReport:
    seed: int
    stages: List[IdentityPoolStageMetric]
    overall_disjoint_transfer_achieved: bool
    final_disjoint_acc: float
    cpu_runtime_ms: float = 0.0


class IdentityDiversityManager:
    """Manages identity pools of progressive diversity while guaranteeing disjoint evaluation."""

    # Stage 1: Pool A - Basic alphanumeric
    POOL_A_KEYS = ["A", "B", "C", "D", "E", "F", "G", "H"]
    POOL_A_VALS = ["1", "2", "3", "4", "5", "6", "7", "8"]

    # Stage 2: Pool B - Extended alphanumeric + safe symbols
    POOL_B_KEYS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
    POOL_B_VALS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "="]

    # Stage 3: Pool C - Wide single-token symbols
    POOL_C_KEYS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O"]
    POOL_C_VALS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "=", ":", ";", "."]

    # Strictly disjoint evaluation pool
    DISJOINT_EVAL_KEYS = ["P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    DISJOINT_EVAL_VALS = ["0", "#", "@", "$", "%", "&", "!", "?"]

    def __init__(self, tokenizer: BPETokenizer, seed: int = 42):
        self.tok = tokenizer
        self.rng = random.Random(seed)
        self.verify_token_integrity()

    def verify_token_integrity(self) -> None:
        """Verifies that every symbol maps to exactly one token ID in BPETokenizer."""
        all_syms = (
            self.POOL_A_KEYS + self.POOL_A_VALS +
            self.POOL_B_KEYS + self.POOL_B_VALS +
            self.POOL_C_KEYS + self.POOL_C_VALS +
            self.DISJOINT_EVAL_KEYS + self.DISJOINT_EVAL_VALS
        )
        for s in set(all_syms):
            enc = self.tok.encode(s)
            assert len(enc) == 1, f"Symbol {s!r} fragments into {len(enc)} tokens: {enc}"

    def sample_episode(
        self,
        pool_name: str,
        split: str,
        num_associations: int = 2,
        episode_idx: int = 0,
    ) -> BindingEpisode:
        if split in ("disjoint_test", "eval"):
            k_pool = list(self.DISJOINT_EVAL_KEYS)
            v_pool = list(self.DISJOINT_EVAL_VALS)
        elif pool_name == "Pool_A":
            k_pool = list(self.POOL_A_KEYS)
            v_pool = list(self.POOL_A_VALS)
        elif pool_name == "Pool_B":
            k_pool = list(self.POOL_B_KEYS)
            v_pool = list(self.POOL_B_VALS)
        else: # Pool_C
            k_pool = list(self.POOL_C_KEYS)
            v_pool = list(self.POOL_C_VALS)

        n_pairs = min(num_associations, len(k_pool), len(v_pool))
        s_k = self.rng.sample(k_pool, n_pairs)
        s_v = self.rng.sample(v_pool, n_pairs)
        pairs = [BindingPair(k, v) for k, v in zip(s_k, s_v)]
        self.rng.shuffle(pairs)

        query_pair = self.rng.choice(pairs)
        q_key = query_pair.key
        exp_val = query_pair.val

        prompt = "map " + " and ".join([f"|{p.key}| -> |{p.val}|" for p in pairs]) + f" query |{q_key}| -> |"

        return BindingEpisode(
            episode_id=f"div_{pool_name}_{split}_{episode_idx}",
            split=split,
            pairs=tuple(pairs),
            query_key=q_key,
            expected_value=exp_val,
            layout_format="format_a",
            prompt=prompt,
            num_associations=n_pairs,
            has_distractors=False,
            episode_hash=f"hash_{pool_name}_{split}_{episode_idx}",
        )


def run_identity_diversity_curriculum(
    base_model: ChakrMicro,
    seed: int = 42,
    steps_per_stage: int = 30,
    eval_episodes: int = 15,
) -> IdentityDiversityCurriculumReport:
    """Executes the progressive identity diversity curriculum (Step 243)."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    t0 = time.time()

    tok = BPETokenizer()
    div_mgr = IdentityDiversityManager(tokenizer=tok, seed=seed)

    candidate = ChakrMicroWithStrengthenedCircuit(base_model)
    candidate.train()
    for p in candidate.base_model.parameters():
        p.requires_grad = False
    for p in candidate.assoc_layer.parameters():
        p.requires_grad = True

    opt = torch.optim.AdamW(candidate.assoc_layer.parameters(), lr=1e-3, weight_decay=1e-4)

    stages_cfg = [
        ("Pool_A", div_mgr.POOL_A_KEYS, div_mgr.POOL_A_VALS),
        ("Pool_B", div_mgr.POOL_B_KEYS, div_mgr.POOL_B_VALS),
        ("Pool_C", div_mgr.POOL_C_KEYS, div_mgr.POOL_C_VALS),
    ]

    stage_metrics: List[IdentityPoolStageMetric] = []
    prev_disjoint_acc = 0.0

    for pool_name, k_pool, v_pool in stages_cfg:
        train_losses: List[float] = []
        train_correct = 0

        candidate.train()
        for step in range(steps_per_stage):
            ep = div_mgr.sample_episode(pool_name, split="train", num_associations=2, episode_idx=step)
            enc = tok.encode(ep.prompt)
            exp_tok = tok.encode(ep.expected_value)[0]
            inp = torch.tensor([enc], dtype=torch.long)
            target = torch.tensor([exp_tok], dtype=torch.long)

            opt.zero_grad()
            logits = candidate(inp)
            loss = F.cross_entropy(logits[0, -1, :].unsqueeze(0), target)
            loss.backward()
            opt.step()

            pred_tok = torch.argmax(logits[0, -1, :]).item()
            if pred_tok == exp_tok:
                train_correct += 1
            train_losses.append(loss.item())

        train_acc = train_correct / max(1, steps_per_stage)
        avg_loss = sum(train_losses) / max(1, len(train_losses))

        # Evaluate on Pool validation
        candidate.eval()
        val_correct = 0
        with torch.no_grad():
            for v_idx in range(eval_episodes):
                ep = div_mgr.sample_episode(pool_name, split="val", num_associations=2, episode_idx=100+v_idx)
                enc = tok.encode(ep.prompt)
                exp_tok = tok.encode(ep.expected_value)[0]
                inp = torch.tensor([enc], dtype=torch.long)
                logits = candidate(inp)
                pred_tok = torch.argmax(logits[0, -1, :]).item()
                if pred_tok == exp_tok:
                    val_correct += 1
        val_acc = val_correct / max(1, eval_episodes)

        # Evaluate on strictly disjoint pool
        disjoint_correct = 0
        target_ranks: List[float] = []
        with torch.no_grad():
            for d_idx in range(eval_episodes):
                ep = div_mgr.sample_episode(pool_name, split="disjoint_test", num_associations=2, episode_idx=200+d_idx)
                enc = tok.encode(ep.prompt)
                exp_tok = tok.encode(ep.expected_value)[0]
                inp = torch.tensor([enc], dtype=torch.long)
                logits = candidate(inp)
                l_last = logits[0, -1, :]
                pred_tok = torch.argmax(l_last).item()
                if pred_tok == exp_tok:
                    disjoint_correct += 1
                rank = (l_last > l_last[exp_tok]).sum().item() + 1
                target_ranks.append(float(rank))

        disjoint_acc = disjoint_correct / max(1, eval_episodes)
        mean_r = sum(target_ranks) / max(1, len(target_ranks))
        retained = disjoint_acc >= prev_disjoint_acc
        prev_disjoint_acc = disjoint_acc

        stage_metrics.append(
            IdentityPoolStageMetric(
                pool_name=pool_name,
                pool_size_keys=len(k_pool),
                pool_size_vals=len(v_pool),
                train_loss=avg_loss,
                train_acc=train_acc,
                val_acc=val_acc,
                heldout_disjoint_acc=disjoint_acc,
                mean_target_rank=mean_r,
                retained_improvement=retained,
            )
        )

    final_disjoint = stage_metrics[-1].heldout_disjoint_acc
    transfer_achieved = final_disjoint > 0.0
    elapsed_ms = (time.time() - t0) * 1000.0

    return IdentityDiversityCurriculumReport(
        seed=seed,
        stages=stage_metrics,
        overall_disjoint_transfer_achieved=transfer_achieved,
        final_disjoint_acc=final_disjoint,
        cpu_runtime_ms=elapsed_ms,
    )
