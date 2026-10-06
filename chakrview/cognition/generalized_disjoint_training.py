"""Step 245: Disjoint Identity Training and I3 Gate Evaluation.

The core capability-growth step of Wave 241-248:
Trains the compact associative circuit over randomized episodes and
evaluates across 4 orthogonal identity splits:
1. known key / known value (in-distribution generalization)
2. known key / unseen value
3. unseen key / known value
4. unseen key / unseen value (primary I3 criterion)

Runs across deterministic seeds:
- 42
- 101
- 2026

Measures:
- final token retrieval accuracy
- target token rank
- association margin
- consistency across seeds
- I3 promotion candidacy

STRICT RULES:
- If unseen/unseen accuracy >= 0.50 across seeds: I3_CANDIDATE_ACHIEVED.
- If unseen/unseen accuracy > 0.0 but < 0.50: I3_EMERGING.
- If unseen/unseen accuracy == 0.0: I3_NOT_ACHIEVED.
"""

from __future__ import annotations

import dataclasses
import random
import time
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro
from chakrview.cognition.association_circuit_learning import ChakrMicroWithStrengthenedCircuit
from chakrview.cognition.generalized_binding_episodes import BindingPair, BindingEpisode
from chakrview.tokenizer import BPETokenizer


@dataclasses.dataclass
class SplitEvaluationResult:
    split_name: str
    num_episodes: int
    accuracy: float
    mean_target_rank: float
    mean_margin: float


@dataclasses.dataclass
class SeedDisjointRunResult:
    seed: int
    splits: Dict[str, SplitEvaluationResult]
    unseen_unseen_acc: float
    unseen_unseen_mean_rank: float
    unseen_unseen_mean_margin: float


@dataclasses.dataclass
class DisjointIdentityReport:
    seeds_tested: List[int]
    per_seed_results: Dict[int, SeedDisjointRunResult]
    mean_known_known_acc: float
    mean_known_unseen_acc: float
    mean_unseen_known_acc: float
    mean_unseen_unseen_acc: float
    i3_status: str               # "I3_CANDIDATE_ACHIEVED", "I3_EMERGING", "I3_NOT_ACHIEVED"
    is_base_immutable: bool
    cpu_runtime_ms: float = 0.0


class DisjointSplitGenerator:
    """Generates episodes partitioned across the 4 orthogonal identity splits."""

    KNOWN_KEYS = ["A", "B", "C", "D", "E", "F", "G", "H"]
    KNOWN_VALS = ["1", "2", "3", "4", "5", "6", "7", "8"]

    UNSEEN_KEYS = ["P", "Q", "R", "S", "T", "U", "V", "W"]
    UNSEEN_VALS = ["0", "#", "@", "$", "%", "&", "!", "?"]

    def __init__(self, tokenizer: BPETokenizer, seed: int = 42):
        self.tok = tokenizer
        self.rng = random.Random(seed)

    def generate_split_episode(
        self,
        split_name: str,
        num_associations: int = 2,
        episode_idx: int = 0,
    ) -> BindingEpisode:
        if split_name == "known_known":
            k_pool, v_pool = self.KNOWN_KEYS, self.KNOWN_VALS
        elif split_name == "known_unseen":
            k_pool, v_pool = self.KNOWN_KEYS, self.UNSEEN_VALS
        elif split_name == "unseen_known":
            k_pool, v_pool = self.UNSEEN_KEYS, self.KNOWN_VALS
        elif split_name == "unseen_unseen":
            k_pool, v_pool = self.UNSEEN_KEYS, self.UNSEEN_VALS
        else: # train
            k_pool, v_pool = self.KNOWN_KEYS, self.KNOWN_VALS

        n = min(num_associations, len(k_pool), len(v_pool))
        s_k = self.rng.sample(k_pool, n)
        s_v = self.rng.sample(v_pool, n)
        pairs = [BindingPair(k, v) for k, v in zip(s_k, s_v)]
        self.rng.shuffle(pairs)

        q_pair = self.rng.choice(pairs)
        q_k = q_pair.key
        exp_v = q_pair.val

        prompt = "map " + " and ".join([f"|{p.key}| -> |{p.val}|" for p in pairs]) + f" query |{q_k}| -> |"

        return BindingEpisode(
            episode_id=f"split_{split_name}_{episode_idx}",
            split=split_name,
            pairs=tuple(pairs),
            query_key=q_k,
            expected_value=exp_v,
            layout_format="format_a",
            prompt=prompt,
            num_associations=n,
            has_distractors=False,
            episode_hash=f"hash_{split_name}_{episode_idx}",
        )


def run_disjoint_identity_training(
    base_model: ChakrMicro,
    seeds: Optional[List[int]] = None,
    train_steps: int = 30,
    eval_episodes_per_split: int = 15,
) -> DisjointIdentityReport:
    """Executes multi-seed generalized training and disjoint identity evaluation."""
    if seeds is None:
        seeds = [42, 101, 2026]

    t0 = time.time()
    tok = BPETokenizer()
    per_seed_res: Dict[int, SeedDisjointRunResult] = {}

    splits_to_eval = ["known_known", "known_unseen", "unseen_known", "unseen_unseen"]

    for s in seeds:
        torch.manual_seed(s)
        gen = DisjointSplitGenerator(tokenizer=tok, seed=s)

        candidate = ChakrMicroWithStrengthenedCircuit(base_model)
        candidate.train()

        # Strict base freeze
        for p in candidate.base_model.parameters():
            p.requires_grad = False
        for p in candidate.assoc_layer.parameters():
            p.requires_grad = True

        opt = torch.optim.AdamW(candidate.assoc_layer.parameters(), lr=1e-3, weight_decay=1e-4)

        # Train on randomized known_known episodes
        for step in range(train_steps):
            ep = gen.generate_split_episode("train", num_associations=2, episode_idx=step)
            enc = tok.encode(ep.prompt)
            exp_tok = tok.encode(ep.expected_value)[0]
            inp = torch.tensor([enc], dtype=torch.long)
            target = torch.tensor([exp_tok], dtype=torch.long)

            opt.zero_grad()
            logits = candidate(inp)
            loss = F.cross_entropy(logits[0, -1, :].unsqueeze(0), target)
            loss.backward()
            opt.step()

        # Evaluate across the 4 splits
        candidate.eval()
        split_results: Dict[str, SplitEvaluationResult] = {}

        with torch.no_grad():
            for sp in splits_to_eval:
                corr = 0
                ranks: List[float] = []
                margins: List[float] = []

                for e_idx in range(eval_episodes_per_split):
                    ep = gen.generate_split_episode(sp, num_associations=2, episode_idx=1000 + e_idx)
                    enc = tok.encode(ep.prompt)
                    exp_tok = tok.encode(ep.expected_value)[0]
                    inp = torch.tensor([enc], dtype=torch.long)

                    logits = candidate(inp)
                    l_last = logits[0, -1, :]
                    pred_tok = torch.argmax(l_last).item()

                    if pred_tok == exp_tok:
                        corr += 1
                    rank = (l_last > l_last[exp_tok]).sum().item() + 1
                    margin = (l_last[exp_tok] - torch.mean(l_last)).item()

                    ranks.append(float(rank))
                    margins.append(float(margin))

                split_results[sp] = SplitEvaluationResult(
                    split_name=sp,
                    num_episodes=eval_episodes_per_split,
                    accuracy=corr / max(1, eval_episodes_per_split),
                    mean_target_rank=sum(ranks) / max(1, len(ranks)),
                    mean_margin=sum(margins) / max(1, len(margins)),
                )

        uu = split_results["unseen_unseen"]
        per_seed_res[s] = SeedDisjointRunResult(
            seed=s,
            splits=split_results,
            unseen_unseen_acc=uu.accuracy,
            unseen_unseen_mean_rank=uu.mean_target_rank,
            unseen_unseen_mean_margin=uu.mean_margin,
        )

    # Compute aggregate means across seeds
    kk_accs = [r.splits["known_known"].accuracy for r in per_seed_res.values()]
    ku_accs = [r.splits["known_unseen"].accuracy for r in per_seed_res.values()]
    uk_accs = [r.splits["unseen_known"].accuracy for r in per_seed_res.values()]
    uu_accs = [r.unseen_unseen_acc for r in per_seed_res.values()]

    mean_kk = sum(kk_accs) / max(1, len(kk_accs))
    mean_ku = sum(ku_accs) / max(1, len(ku_accs))
    mean_uk = sum(uk_accs) / max(1, len(uk_accs))
    mean_uu = sum(uu_accs) / max(1, len(uu_accs))

    if mean_uu >= 0.50:
        i3_status = "I3_CANDIDATE_ACHIEVED"
    elif mean_uu > 0.0:
        i3_status = "I3_EMERGING"
    else:
        i3_status = "I3_NOT_ACHIEVED"

    elapsed_ms = (time.time() - t0) * 1000.0

    return DisjointIdentityReport(
        seeds_tested=seeds,
        per_seed_results=per_seed_res,
        mean_known_known_acc=mean_kk,
        mean_known_unseen_acc=mean_ku,
        mean_unseen_known_acc=mean_uk,
        mean_unseen_unseen_acc=mean_uu,
        i3_status=i3_status,
        is_base_immutable=True,
        cpu_runtime_ms=elapsed_ms,
    )
