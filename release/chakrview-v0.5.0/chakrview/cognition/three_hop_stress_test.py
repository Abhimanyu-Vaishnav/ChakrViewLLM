"""Step 287: Three-Hop Compositional Reasoning Stress Test.

Extends the 2-hop compositional reasoning paradigm to 3-hop chains:
A -> B
B -> C
C -> D
Query: A -> ?
Target: D

Measures:
- Hop 1 routing (A -> B)
- Hop 2 routing (B -> C)
- Hop 3 routing (C -> D)
- Cumulative degradation across hops
- Final token accuracy

Determines whether compounding representation error breaks chaining at Hop 3.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro
from chakrview.cognition.two_hop_composition_architecture import (
    ChakrMicroCompositionalReasoningModel,
)
from chakrview.cognition.compositional_benchmark_environment import (
    CompositionalAssociativeEnvironment,
)


@dataclasses.dataclass
class ThreeHopStressResult:
    num_episodes: int
    hop1_routing_acc: float
    hop2_routing_acc: float
    hop3_routing_acc: float
    final_token_acc: float
    first_failure_boundary: str
    survives_three_hops: bool
    summary: str


def run_three_hop_stress_test(
    candidate: ChakrMicroCompositionalReasoningModel,
    env: Optional[CompositionalAssociativeEnvironment] = None,
    seed: int = 42,
    num_episodes: int = 10,
) -> ThreeHopStressResult:
    """Evaluates 3-hop composition chain A->B->C->D."""
    torch.manual_seed(seed)
    if env is None:
        env = CompositionalAssociativeEnvironment(seed=seed)
    tok = env.tok

    candidate.eval()

    h1_corr = 0
    h2_corr = 0
    h3_corr = 0
    tok_corr = 0

    with torch.no_grad():
        for i in range(num_episodes):
            # Sample 4 distinct symbols A, B, C, D
            k_pool = list(env.DISJOINT_KEYS_POOL)
            sampled = env.rng.sample(k_pool, 4)
            A, B, C, D = sampled[0], sampled[1], sampled[2], sampled[3]

            pairs = [(A, B), (B, C), (C, D)]
            env.rng.shuffle(pairs)

            prompt = env.render_prompt(pairs, query_key=A, layout="standard_map")
            tokens = tok.encode(prompt)
            inp = torch.tensor([tokens], dtype=torch.long)

            h_ad, _ = candidate.forward_backbone(inp)
            h_norm = F.normalize(h_ad[0], p=2, dim=-1)

            # Extract premise key positions
            premise_key_pos = []
            for k, v in pairs:
                k_enc = tok.encode(k)[0]
                m = [j for j, t in enumerate(tokens[:-1]) if t == k_enc]
                if m: premise_key_pos.append(m[0])

            premise_val_pos = []
            for k, v in pairs:
                v_enc = tok.encode(v)[0]
                m = [j for j, t in enumerate(tokens[:-1]) if t == v_enc]
                if m: premise_val_pos.append(m[0])

            # Query position
            A_enc = tok.encode(A)[0]
            q_pos = [j for j, t in enumerate(tokens[:-1]) if t == A_enc][-1]

            # Hop 1 (A -> B)
            h_q1 = h_norm[q_pos]
            k1_sims = [float(torch.dot(h_q1, h_norm[kp]).item()) for kp in premise_key_pos]
            pred_k1 = premise_key_pos[k1_sims.index(max(k1_sims))]
            h_mk1 = h_norm[pred_k1]
            v1_sims = [float(torch.dot(h_mk1, h_norm[vp]).item()) for vp in premise_val_pos]
            pred_v1 = premise_val_pos[v1_sims.index(max(v1_sims))]

            B_enc = tok.encode(B)[0]
            if tokens[pred_v1] == B_enc:
                h1_corr += 1

            # Hop 2 (B -> C)
            h_v1 = h_ad[0, pred_v1 : pred_v1 + 1]
            h_q2 = candidate.bridge_intermediate_state(h_v1)
            h_q2_norm = F.normalize(h_q2[0], p=2, dim=-1)

            k2_sims = [float(torch.dot(h_q2_norm, h_norm[kp]).item()) for kp in premise_key_pos]
            pred_k2 = premise_key_pos[k2_sims.index(max(k2_sims))]
            h_mk2 = h_norm[pred_k2]
            v2_sims = [float(torch.dot(h_mk2, h_norm[vp]).item()) for vp in premise_val_pos]
            pred_v2 = premise_val_pos[v2_sims.index(max(v2_sims))]

            C_enc = tok.encode(C)[0]
            if tokens[pred_v2] == C_enc:
                h2_corr += 1

            # Hop 3 (C -> D)
            h_v2 = h_ad[0, pred_v2 : pred_v2 + 1]
            h_q3 = candidate.bridge_intermediate_state(h_v2)
            h_q3_norm = F.normalize(h_q3[0], p=2, dim=-1)

            k3_sims = [float(torch.dot(h_q3_norm, h_norm[kp]).item()) for kp in premise_key_pos]
            pred_k3 = premise_key_pos[k3_sims.index(max(k3_sims))]
            h_mk3 = h_norm[pred_k3]
            v3_sims = [float(torch.dot(h_mk3, h_norm[vp]).item()) for vp in premise_val_pos]
            pred_v3 = premise_val_pos[v3_sims.index(max(v3_sims))]

            D_enc = tok.encode(D)[0]
            if tokens[pred_v3] == D_enc:
                h3_corr += 1

            # Final candidate selection
            cand_positions = list(set(premise_val_pos))
            cand_tokens = [tokens[p] for p in cand_positions]
            cand_states = torch.stack([h_ad[0, p] for p in cand_positions], dim=0).unsqueeze(0)
            cand_mask = torch.ones((1, len(cand_positions)), dtype=torch.bool, device=h_ad.device)

            val_final_rep = h_ad[0, pred_v3 : pred_v3 + 1]
            bind_logits, _ = candidate.compute_binding_scores(val_final_rep, cand_states, cand_mask)
            pred_cand_idx = bind_logits.argmax().item()
            if cand_tokens[pred_cand_idx] == D_enc:
                tok_corr += 1

    h1_acc = h1_corr / num_episodes
    h2_acc = h2_corr / num_episodes
    h3_acc = h3_corr / num_episodes
    t_acc = tok_corr / num_episodes

    boundary = "NONE"
    if h1_acc < 0.50:
        boundary = "HOP_1"
    elif h2_acc < 0.50:
        boundary = "HOP_2"
    elif h3_acc < 0.50:
        boundary = "HOP_3"
    elif t_acc < 0.50:
        boundary = "FINAL_BINDING"

    survives = (t_acc >= 0.50 and h3_acc >= 0.50)

    return ThreeHopStressResult(
        num_episodes=num_episodes,
        hop1_routing_acc=h1_acc,
        hop2_routing_acc=h2_acc,
        hop3_routing_acc=h3_acc,
        final_token_acc=t_acc,
        first_failure_boundary=boundary,
        survives_three_hops=survives,
        summary=f"3-Hop Trace: Hop1={h1_acc*100:.1f}%, Hop2={h2_acc*100:.1f}%, Hop3={h3_acc*100:.1f}%, Final Token={t_acc*100:.1f}%. Boundary: {boundary}",
    )
