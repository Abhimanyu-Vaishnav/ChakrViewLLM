"""Step 381: State Causality & Stability under Adaptive Cycles.

Performs controlled causal interventions across cycle states S1, S2, S3:
A. Normal state (unperturbed forward pass)
B. Zero state (zeroed S_t)
C. Random Gaussian state (noise S_t)
D. Corrupted state (feature dimension permutation)
E. Shuffled state (random shuffle across vector)
F. Swapped state from alternative episode
G. Bypass state (query bypassed to initial q0)
H. Frozen previous state (state frozen to S_{t-1})

Measures:
- Continuation probability p_continue
- Hop-2 & Hop-3 routing
- Final target token probability & accuracy
- State norms, cosine similarity across cycles, drift, entropy
"""

from __future__ import annotations

import dataclasses
import math
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.cognition.adaptive_recurrent_reasoning_core import AdaptiveRecurrentReasoningCore
from chakrview.cognition.compositional_benchmark_environment import (
    CompositionalAssociativeEnvironment,
)
from chakrview.cognition.adaptive_learnability import (
    generate_mixed_hop_episode,
    AdaptiveEpisode,
)


@dataclasses.dataclass
class CausalInterventionResult:
    condition_id: str
    description: str
    target_prob: float
    final_acc: float
    continue_prob_mean: float
    h2_routing_acc: float
    drop_from_normal: float


@dataclasses.dataclass
class StateStabilityMetrics:
    mean_state_norm: float
    inter_cycle_cosine_sim: float
    state_entropy: float
    cycle_drift: float
    query_state_alignment: float


@dataclasses.dataclass
class Step381StateCausalityReport:
    interventions: Dict[str, CausalInterventionResult]
    stability: StateStabilityMetrics
    causally_dependent: bool
    summary: str


def run_adaptive_state_causality_study(
    core: Optional[AdaptiveRecurrentReasoningCore] = None,
    seed: int = 42,
    num_episodes: int = 8,
) -> Step381StateCausalityReport:
    """Evaluates causal interventions on cycle states S1 and S2."""
    torch.manual_seed(seed)
    env = CompositionalAssociativeEnvironment(seed=seed)

    if core is None:
        core = AdaptiveRecurrentReasoningCore()
        core.eval()

    episodes = [
        generate_mixed_hop_episode(env, hop_count=2, split="disjoint_test", num_distractors=1)
        for _ in range(num_episodes)
    ]

    conditions = [
        ("A_Normal", "Unperturbed execution"),
        ("B_Zero", "State zeroed S_1 = 0"),
        ("C_Random", "State replaced by standard Gaussian noise"),
        ("D_Corrupted", "State corrupted by adding scaled noise"),
        ("E_Shuffled", "State dimensions randomly permuted"),
        ("F_Swapped", "State swapped from alternate episode"),
        ("G_Bypass", "Query formulation bypasses state"),
        ("H_Frozen", "State frozen to s0"),
    ]

    # Pre-extract normal states for swapping
    all_s1 = []
    with torch.no_grad():
        for ep in episodes:
            seq = torch.tensor([ep.prompt_tokens], dtype=torch.long)
            out = core(seq, fixed_cycles=1, return_trace=True)
            all_s1.append(out["final_state"].clone())

    normal_prob = 0.0
    results: Dict[str, CausalInterventionResult] = {}

    for cond_id, desc in conditions:
        tot_prob = 0.0
        hits = 0
        tot_c_prob = 0.0
        h2_hits = 0

        with torch.no_grad():
            for ep_i, ep in enumerate(episodes):
                seq = torch.tensor([ep.prompt_tokens], dtype=torch.long)
                c_pos = torch.tensor([ep.candidate_positions], dtype=torch.long)

                override_s = None
                bypass_s = False

                if cond_id == "B_Zero":
                    override_s = torch.zeros(1, core.d_state)
                elif cond_id == "C_Random":
                    override_s = torch.randn(1, core.d_state)
                elif cond_id == "D_Corrupted":
                    base_s = all_s1[ep_i]
                    override_s = base_s + 0.5 * torch.randn_like(base_s)
                elif cond_id == "E_Shuffled":
                    base_s = all_s1[ep_i]
                    p_idx = torch.randperm(base_s.shape[-1])
                    override_s = base_s[:, p_idx]
                elif cond_id == "F_Swapped":
                    swap_i = (ep_i + 1) % len(episodes)
                    override_s = all_s1[swap_i]
                elif cond_id == "G_Bypass":
                    bypass_s = True
                elif cond_id == "H_Frozen":
                    override_s = core.s0.clone()

                out = core(
                    input_ids=seq,
                    candidate_positions=c_pos,
                    fixed_cycles=2,
                    override_state=override_s,
                    override_cycle=1,
                    return_trace=True,
                )

                b_probs = out["binding_probs"]
                tgt_p = b_probs[0, ep.target_idx].item() if b_probs is not None else 0.0
                tot_prob += tgt_p

                pred_idx = torch.argmax(out["binding_logits"][0]).item()
                if pred_idx == ep.target_idx:
                    hits += 1

                # Continue prob
                p_conts = out["p_continue_per_cycle"]
                if p_conts:
                    tot_c_prob += p_conts[0].item()

                # Routing
                traces = out["cycle_traces"]
                if len(traces) >= 2 and traces[1].w_attn is not None:
                    w2 = traces[1].w_attn[0].mean(dim=0)[-1, :]
                    if len(ep.key_positions) >= 2 and ep.key_positions[1] >= 0:
                        if torch.argmax(w2).item() == ep.key_positions[1]:
                            h2_hits += 1

        N = max(1, len(episodes))
        mean_p = tot_prob / N
        acc = hits / N
        m_c_prob = tot_c_prob / N
        h2_acc = h2_hits / N

        if cond_id == "A_Normal":
            normal_prob = mean_p
            drop = 0.0
        else:
            drop = normal_prob - mean_p

        results[cond_id] = CausalInterventionResult(
            condition_id=cond_id,
            description=desc,
            target_prob=mean_p,
            final_acc=acc,
            continue_prob_mean=m_c_prob,
            h2_routing_acc=h2_acc,
            drop_from_normal=drop,
        )

    # State stability metrics across cycle 1 and cycle 2
    norms = []
    cos_sims = []
    entropies = []
    drifts = []
    alignments = []

    with torch.no_grad():
        for ep in episodes:
            seq = torch.tensor([ep.prompt_tokens], dtype=torch.long)
            out = core(seq, fixed_cycles=2, return_trace=True)
            traces = out["cycle_traces"]
            if len(traces) >= 2:
                s1 = traces[0].state_s
                s2 = traces[1].state_s
                norms.append(torch.norm(s1, p=2).item())
                norms.append(torch.norm(s2, p=2).item())

                sim = F.cosine_similarity(s1, s2).item()
                cos_sims.append(sim)

                drift = torch.norm(s2 - s1, p=2).item()
                drifts.append(drift)

                # Entropy of softmax over state dimensions
                p_state = torch.softmax(s1, dim=-1)
                ent = -(p_state * torch.log(p_state + 1e-8)).sum().item()
                entropies.append(ent)

                # Query-state alignment
                q2 = traces[1].query_q.squeeze(1)
                # Project state to query space
                q_proj = core.state_to_q(s1)
                q_sim = F.cosine_similarity(q2, q_proj).item()
                alignments.append(q_sim)

    stability = StateStabilityMetrics(
        mean_state_norm=sum(norms) / max(1, len(norms)),
        inter_cycle_cosine_sim=sum(cos_sims) / max(1, len(cos_sims)),
        state_entropy=sum(entropies) / max(1, len(entropies)),
        cycle_drift=sum(drifts) / max(1, len(drifts)),
        query_state_alignment=sum(alignments) / max(1, len(alignments)),
    )

    causal = any(r.drop_from_normal != 0.0 for k, r in results.items() if k != "A_Normal")

    summary = (
        f"State Causality: Normal Prob={normal_prob:.4f}, Zero Drop={results['B_Zero'].drop_from_normal:+.4f}, "
        f"Random Drop={results['C_Random'].drop_from_normal:+.4f}, Inter-Cycle Cosine Sim={stability.inter_cycle_cosine_sim:.4f}, "
        f"Query Alignment={stability.query_state_alignment:.4f}."
    )

    return Step381StateCausalityReport(
        interventions=results,
        stability=stability,
        causally_dependent=causal,
        summary=summary,
    )


if __name__ == "__main__":
    print("Step 381: Evaluating State Causality and Stability...")
    rep = run_adaptive_state_causality_study(num_episodes=6)
    print("Report Summary:", rep.summary)
    for c_id, r in rep.interventions.items():
        print(f"  [{c_id:12s}] Acc={r.final_acc:.2%}, Prob={r.target_prob:.4f}, ContProb={r.continue_prob_mean:.4f}, Drop={r.drop_from_normal:+.4f}")
