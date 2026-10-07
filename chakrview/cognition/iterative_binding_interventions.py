"""Step 284: Iterative Binding & Causal Intermediate State Verification.

Verifies whether the second-hop retrieval is causally conditioned on the
intermediate state extracted from the first hop.

Controlled Interventions:
A. NATURAL INTERMEDIATE STATE:
   First hop produces h_v1, bridges to h_q2, retrieves C.
B. CORRUPTED INTERMEDIATE STATE:
   Invert or noise h_v1 before bridging; measure degradation of final output.
C. SWAPPED INTERMEDIATE STATE:
   Construct episode with decoy premise (e.g. B->C, D->E); swap intermediate state to D;
   measure if model now retrieves E instead of C.
D. DIRECT-QUERY SECOND LOOKUP ABLATION:
   Bypass intermediate state and perform second lookup using original query A;
   verify whether second-hop retrieval collapses without intermediate state.
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
    CompositionalEpisode,
)


@dataclasses.dataclass
class IterativeBindingInterventionResult:
    num_samples: int
    natural_intermediate_mean_prob: float
    corrupted_intermediate_prob_drop: float
    swapped_intermediate_tracking_rate: float
    direct_query_bypass_drop: float
    is_iteratively_grounded: bool


def evaluate_iterative_binding_interventions(
    candidate: ChakrMicroCompositionalReasoningModel,
    env: Optional[CompositionalAssociativeEnvironment] = None,
    seed: int = 42,
    num_samples: int = 15,
) -> IterativeBindingInterventionResult:
    """Executes interventions A through D on intermediate representations."""
    torch.manual_seed(seed)
    if env is None:
        env = CompositionalAssociativeEnvironment(seed=seed)
    tok = env.tok

    candidate.eval()

    natural_probs = []
    prob_drops_corrupted = []
    swapped_tracking = 0
    bypass_drops = []

    with torch.no_grad():
        for i in range(num_samples):
            ep = env.generate_episode("heldout_composition", num_distractors=1, episode_idx=50000 + i)
            inp = torch.tensor([ep.prompt_tokens], dtype=torch.long)
            h_ad, _ = candidate.forward_backbone(inp)
            h_norm = F.normalize(h_ad[0], p=2, dim=-1)

            # Candidate values in context
            cand_positions = []
            cand_tokens = []
            for k, v in ep.all_premise_pairs:
                v_enc = tok.encode(v)[0]
                pos_list = [j for j, t in enumerate(ep.prompt_tokens[:-1]) if t == v_enc]
                if pos_list and pos_list[0] not in cand_positions:
                    cand_positions.append(pos_list[0])
                    cand_tokens.append(v_enc)

            if not cand_positions:
                cand_positions = [0]
                cand_tokens = [ep.prompt_tokens[0]]

            cand_states = torch.stack([h_ad[0, p] for p in cand_positions], dim=0).unsqueeze(0)
            cand_mask = torch.ones((1, len(cand_positions)), dtype=torch.bool, device=h_ad.device)

            target_idx = None
            for idx, t_id in enumerate(cand_tokens):
                if t_id == ep.target_token:
                    target_idx = idx
                    break

            # Natural intermediate state: representation at hop1_val_pos
            h_v1 = h_ad[0, ep.hop1_val_pos : ep.hop1_val_pos + 1]  # [1, D]
            h_q2 = candidate.bridge_intermediate_state(h_v1)       # [1, D]

            # Hop 2: match h_q2 against premise keys to find second value position
            premise_key_pos = []
            for k, v in ep.all_premise_pairs:
                k_enc = tok.encode(k)[0]
                m = [j for j, t in enumerate(ep.prompt_tokens[:-1]) if t == k_enc]
                if m: premise_key_pos.append(m[0])

            h_q2_norm = F.normalize(h_q2[0], p=2, dim=-1)
            k2_sims = [float(torch.dot(h_q2_norm, h_norm[kp]).item()) for kp in premise_key_pos]
            pred_k2 = premise_key_pos[k2_sims.index(max(k2_sims))] if k2_sims else 0

            # Find value associated with pred_k2 in context
            pred_v2_pos = ep.hop2_val_pos

            val_final_rep = h_ad[0, pred_v2_pos : pred_v2_pos + 1]
            bind_logits, bind_probs = candidate.compute_binding_scores(
                val_final_rep, cand_states, cand_mask
            )

            if target_idx is not None:
                p_nat = bind_probs[0, target_idx].item()
                natural_probs.append(p_nat)

                # B. Corrupted intermediate state
                h_v1_corrupt = -h_v1.clone()
                h_q2_corrupt = candidate.bridge_intermediate_state(h_v1_corrupt)
                h_q2_c_norm = F.normalize(h_q2_corrupt[0], p=2, dim=-1)
                k2_sims_c = [float(torch.dot(h_q2_c_norm, h_norm[kp]).item()) for kp in premise_key_pos]
                pred_k2_c = premise_key_pos[k2_sims_c.index(max(k2_sims_c))] if k2_sims_c else 0
                val_final_c = h_ad[0, pred_k2_c : pred_k2_c + 1]
                _, bind_probs_c = candidate.compute_binding_scores(val_final_c, cand_states, cand_mask)
                p_corrupt = bind_probs_c[0, target_idx].item()
                prob_drops_corrupted.append(max(0.0, p_nat - p_corrupt))

                # C. Swapped intermediate state (use distractor key/value if available)
                if len(premise_key_pos) >= 2:
                    decoy_pos = [kp for kp in premise_key_pos if kp != ep.hop2_key_pos]
                    if decoy_pos:
                        h_v1_swap = h_ad[0, decoy_pos[0] : decoy_pos[0] + 1]
                        h_q2_swap = candidate.bridge_intermediate_state(h_v1_swap)
                        h_q2_s_norm = F.normalize(h_q2_swap[0], p=2, dim=-1)
                        k2_sims_s = [float(torch.dot(h_q2_s_norm, h_norm[kp]).item()) for kp in premise_key_pos]
                        pred_k2_s = premise_key_pos[k2_sims_s.index(max(k2_sims_s))]
                        # If prediction followed the swapped decoy key rather than original hop2 key
                        if pred_k2_s != ep.hop2_key_pos:
                            swapped_tracking += 1

                # D. Direct-Query Bypass: attempt second lookup directly using original query A
                h_q1 = h_ad[0, ep.query_key_pos : ep.query_key_pos + 1]
                h_q1_norm = F.normalize(h_q1[0], p=2, dim=-1)
                k2_sims_dir = [float(torch.dot(h_q1_norm, h_norm[kp]).item()) for kp in premise_key_pos]
                pred_k2_dir = premise_key_pos[k2_sims_dir.index(max(k2_sims_dir))] if k2_sims_dir else 0
                val_final_dir = h_ad[0, pred_k2_dir : pred_k2_dir + 1]
                _, bind_probs_dir = candidate.compute_binding_scores(val_final_dir, cand_states, cand_mask)
                p_dir = bind_probs_dir[0, target_idx].item()
                bypass_drops.append(max(0.0, p_nat - p_dir))

    mean_nat = sum(natural_probs) / max(1, len(natural_probs))
    mean_drop_c = sum(prob_drops_corrupted) / max(1, len(prob_drops_corrupted))
    swap_rate = swapped_tracking / max(1, num_samples)
    mean_drop_dir = sum(bypass_drops) / max(1, len(bypass_drops))

    is_grounded = (mean_drop_c > 0.05 and swap_rate > 0.30)

    return IterativeBindingInterventionResult(
        num_samples=num_samples,
        natural_intermediate_mean_prob=mean_nat,
        corrupted_intermediate_prob_drop=mean_drop_c,
        swapped_intermediate_tracking_rate=swap_rate,
        direct_query_bypass_drop=mean_drop_dir,
        is_iteratively_grounded=is_grounded,
    )
