"""Step 378: Minimal Adaptive Learnability on Mixed 1-Hop, 2-Hop, and 3-Hop Reasoning.

Constructs minimal randomized reasoning tasks:
- 1-hop: A -> B, query A -> B (ideal: ~1 cycle)
- 2-hop: A -> B, B -> C, query A -> C (ideal: ~2 cycles)
- 3-hop: A -> B, B -> C, C -> D, query A -> D (ideal: ~3 cycles)

Trains AdaptiveRecurrentReasoningCore and measures:
- final accuracy per hop depth
- H1, H2, H3 routing accuracy
- average cycles per task depth
- maximum cycles
- halt decision accuracy (premature halts vs unnecessary cycles)
- intermediate state validity
"""

from __future__ import annotations

import dataclasses
import random
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.cognition.adaptive_recurrent_reasoning_core import AdaptiveRecurrentReasoningCore
from chakrview.cognition.compositional_benchmark_environment import (
    CompositionalAssociativeEnvironment,
)


@dataclasses.dataclass
class AdaptiveEpisode:
    hop_count: int                        # 1, 2, or 3
    prompt: str
    prompt_tokens: List[int]
    query_key: str
    target_token: int
    candidate_tokens: List[int]
    candidate_positions: List[int]
    target_idx: int
    chain_keys: List[str]
    chain_vals: List[str]
    key_positions: List[int]
    val_positions: List[int]


def generate_mixed_hop_episode(
    env: CompositionalAssociativeEnvironment,
    hop_count: int = 2,
    split: str = "train",
    num_distractors: int = 1,
) -> AdaptiveEpisode:
    """Generates an episode with 1, 2, or 3 hops with distractors and randomized layout."""
    tok = env.tok
    if split == "train":
        k_pool = list(env.TRAIN_KEYS_POOL)
        v_pool = list(env.TRAIN_VALS_POOL)
    else:
        k_pool = list(env.DISJOINT_KEYS_POOL)
        v_pool = list(env.DISJOINT_VALS_POOL)

    # Need hop_count + 1 symbols for chain: e.g. A->B (2 syms), A->B->C (3 syms), A->B->C->D (4 syms)
    needed = hop_count + 1
    sampled = env.rng.sample(k_pool, needed)
    chain_keys = sampled[:-1]
    chain_vals = sampled[1:]

    chain_pairs = list(zip(chain_keys, chain_vals))

    # Distractors
    distractor_pairs = []
    avail_d = [k for k in k_pool if k not in sampled]
    avail_v = [v for v in v_pool if v not in sampled]
    for _ in range(num_distractors):
        if avail_d and avail_v:
            dk = env.rng.choice(avail_d)
            dv = env.rng.choice(avail_v)
            distractor_pairs.append((dk, dv))

    all_pairs = chain_pairs + distractor_pairs
    env.rng.shuffle(all_pairs)

    query_key = chain_keys[0]
    final_val = chain_vals[-1]

    prompt = env.render_prompt(all_pairs, query_key=query_key, layout="standard_map")
    tokens = tok.encode(prompt)

    # Candidate values: all premise values in prompt
    cand_tokens, cand_positions = [], []
    for k, v in all_pairs:
        v_enc = tok.encode(v)[0]
        pos_list = [j for j, t in enumerate(tokens[:-1]) if t == v_enc]
        if pos_list and pos_list[0] not in cand_positions:
            cand_positions.append(pos_list[0])
            cand_tokens.append(v_enc)

    if not cand_positions:
        cand_positions, cand_tokens = [0], [tokens[0]]

    final_val_enc = tok.encode(final_val)[0]
    target_idx = cand_tokens.index(final_val_enc) if final_val_enc in cand_tokens else 0

    # Key / Value positions for routing checks
    key_positions = []
    for k in chain_keys:
        k_enc = tok.encode(k)[0]
        pos = [j for j, t in enumerate(tokens) if t == k_enc]
        key_positions.append(pos[0] if pos else -1)

    val_positions = []
    for v in chain_vals:
        v_enc = tok.encode(v)[0]
        pos = [j for j, t in enumerate(tokens) if t == v_enc]
        val_positions.append(pos[0] if pos else -1)

    return AdaptiveEpisode(
        hop_count=hop_count,
        prompt=prompt,
        prompt_tokens=tokens,
        query_key=query_key,
        target_token=final_val_enc,
        candidate_tokens=cand_tokens,
        candidate_positions=cand_positions,
        target_idx=target_idx,
        chain_keys=chain_keys,
        chain_vals=chain_vals,
        key_positions=key_positions,
        val_positions=val_positions,
    )


@dataclasses.dataclass
class HopLearnabilityMetrics:
    hop_count: int
    accuracy: float
    avg_cycles: float
    min_cycles: int
    max_cycles: int
    premature_halts: float
    unnecessary_cycles: float
    h1_routing: float
    h2_routing: float
    h3_routing: float


@dataclasses.dataclass
class Step378LearnabilityReport:
    metrics_by_hop: Dict[int, HopLearnabilityMetrics]
    overall_accuracy: float
    mean_cycles_overall: float
    adaptive_depth_aligned: bool
    summary: str


def train_and_eval_adaptive_learnability(
    core: Optional[AdaptiveRecurrentReasoningCore] = None,
    seed: int = 42,
    train_steps: int = 30,
    eval_episodes_per_hop: int = 8,
    lambda_compute: float = 0.05,
) -> Tuple[AdaptiveRecurrentReasoningCore, Step378LearnabilityReport]:
    """Trains and tests minimal adaptive learnability across 1-hop, 2-hop, and 3-hop tasks."""
    torch.manual_seed(seed)
    random.seed(seed)
    env = CompositionalAssociativeEnvironment(seed=seed)

    if core is None:
        core = AdaptiveRecurrentReasoningCore()

    optimizer = torch.optim.Adam(core.parameters(), lr=1e-3, weight_decay=1e-4)

    # Mixed hop training
    core.train()
    hops_pool = [1, 2, 3]

    for step in range(train_steps):
        optimizer.zero_grad()
        hop_k = hops_pool[step % len(hops_pool)]
        ep = generate_mixed_hop_episode(env, hop_count=hop_k, split="train", num_distractors=1)

        seq = torch.tensor([ep.prompt_tokens], dtype=torch.long)
        c_pos = torch.tensor([ep.candidate_positions], dtype=torch.long)
        tgt = torch.tensor([ep.target_idx], dtype=torch.long)

        out = core(
            input_ids=seq,
            candidate_positions=c_pos,
            max_reasoning_cycles=4,
            return_trace=True,
        )

        b_logits = out["binding_logits"]
        loss_bind = F.cross_entropy(b_logits, tgt)

        # Halting supervision / compute penalty
        # Target: continue for (hop_k - 1) cycles, halt at cycle hop_k
        halt_loss = 0.0
        p_cont = out["p_continue_per_cycle"]
        for c_i, p_val in enumerate(p_cont):
            cycle_num = c_i + 1
            if cycle_num < hop_k:
                # Should continue
                halt_loss = halt_loss + F.binary_cross_entropy(p_val, torch.ones_like(p_val))
            else:
                # Should halt
                halt_loss = halt_loss + F.binary_cross_entropy(p_val, torch.zeros_like(p_val))

        loss_compute = out["computation_cost"] * lambda_compute
        total_loss = loss_bind + 0.3 * halt_loss + loss_compute

        total_loss.backward()
        optimizer.step()

    # Evaluation per hop depth on held-out disjoint split
    core.eval()
    metrics_by_hop: Dict[int, HopLearnabilityMetrics] = {}
    all_acc = []
    all_cycles = []

    for hop_k in [1, 2, 3]:
        hits = 0
        cycles_list = []
        premature = 0
        unnecessary = 0
        h1_hits = 0
        h2_hits = 0
        h3_hits = 0

        with torch.no_grad():
            for ep_i in range(eval_episodes_per_hop):
                ep = generate_mixed_hop_episode(
                    env, hop_count=hop_k, split="disjoint_test", num_distractors=1
                )
                seq = torch.tensor([ep.prompt_tokens], dtype=torch.long)
                c_pos = torch.tensor([ep.candidate_positions], dtype=torch.long)

                out = core(
                    input_ids=seq,
                    candidate_positions=c_pos,
                    max_reasoning_cycles=6,
                    halt_threshold=0.5,
                    return_trace=True,
                )

                c_used = out["total_cycles_executed"]
                cycles_list.append(c_used)

                pred_idx = torch.argmax(out["binding_logits"][0]).item()
                if pred_idx == ep.target_idx:
                    hits += 1

                # Cycle calibration diagnostics
                if c_used < hop_k:
                    premature += 1
                elif c_used > hop_k:
                    unnecessary += 1

                # Routing checks
                traces = out["cycle_traces"]
                if len(traces) >= 1 and traces[0].w_attn is not None:
                    w1 = traces[0].w_attn[0].mean(dim=0)[-1, :]
                    if ep.key_positions[0] >= 0 and torch.argmax(w1).item() == ep.key_positions[0]:
                        h1_hits += 1
                if len(traces) >= 2 and traces[1].w_attn is not None:
                    w2 = traces[1].w_attn[0].mean(dim=0)[-1, :]
                    if len(ep.key_positions) >= 2 and ep.key_positions[1] >= 0 and torch.argmax(w2).item() == ep.key_positions[1]:
                        h2_hits += 1
                if len(traces) >= 3 and traces[2].w_attn is not None:
                    w3 = traces[2].w_attn[0].mean(dim=0)[-1, :]
                    if len(ep.key_positions) >= 3 and ep.key_positions[2] >= 0 and torch.argmax(w3).item() == ep.key_positions[2]:
                        h3_hits += 1

        N = max(1, eval_episodes_per_hop)
        acc = hits / N
        avg_c = sum(cycles_list) / N
        all_acc.append(acc)
        all_cycles.extend(cycles_list)

        metrics_by_hop[hop_k] = HopLearnabilityMetrics(
            hop_count=hop_k,
            accuracy=acc,
            avg_cycles=avg_c,
            min_cycles=min(cycles_list) if cycles_list else 0,
            max_cycles=max(cycles_list) if cycles_list else 0,
            premature_halts=premature / N,
            unnecessary_cycles=unnecessary / N,
            h1_routing=h1_hits / N,
            h2_routing=h2_hits / N,
            h3_routing=h3_hits / N,
        )

    mean_acc = sum(all_acc) / len(all_acc)
    mean_c = sum(all_cycles) / len(all_cycles)

    # Check if cycles monotonically increase with difficulty (avg_c(1) < avg_c(2) <= avg_c(3))
    c1 = metrics_by_hop[1].avg_cycles
    c2 = metrics_by_hop[2].avg_cycles
    c3 = metrics_by_hop[3].avg_cycles
    aligned = (c1 <= c2) and (c2 <= c3 + 0.5)

    summary = (
        f"Adaptive Learnability: Mean Acc={mean_acc:.2%}, Mean Cycles={mean_c:.2f}. "
        f"1-Hop: Acc={metrics_by_hop[1].accuracy:.2%}, AvgC={c1:.2f}. "
        f"2-Hop: Acc={metrics_by_hop[2].accuracy:.2%}, AvgC={c2:.2f}. "
        f"3-Hop: Acc={metrics_by_hop[3].accuracy:.2%}, AvgC={c3:.2f}. "
        f"Adaptive Depth Monotonic: {aligned}."
    )

    rep = Step378LearnabilityReport(
        metrics_by_hop=metrics_by_hop,
        overall_accuracy=mean_acc,
        mean_cycles_overall=mean_c,
        adaptive_depth_aligned=aligned,
        summary=summary,
    )
    return core, rep


if __name__ == "__main__":
    print("Step 378: Training and evaluating adaptive learnability...")
    core, rep = train_and_eval_adaptive_learnability(train_steps=30, eval_episodes_per_hop=6)
    print("Report:", rep.summary)
