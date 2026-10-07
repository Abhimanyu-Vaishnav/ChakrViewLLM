"""Step 282: Intermediate State Extraction & Compositional Diagnostic Trace.

Instruments the neural model to observe:
1. Query representation: h[query_pos]
2. First-hop key matching: query attending to premise keys
3. First-hop value routing: matched key attending to premise values
4. First retrieved value representation: h_inter = h[hop1_val_pos]
5. Intermediate state quality & identity: does h_inter match token B?
6. Second-hop key matching: h_inter attending to second premise keys
7. Second-hop value routing: second key attending to final premise values
8. Final contextual token binding: compatibility over candidate tokens
9. Emitted token: does it match target token C?

Traces exactly where information is preserved or lost.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro
from chakrview.cognition.compositional_benchmark_environment import (
    CompositionalEpisode,
)
from chakrview.cognition.contextual_token_candidates import (
    extract_contextual_candidates_from_episode,
)


@dataclasses.dataclass
class CompositionalHopTrace:
    episode_id: str
    hop1_key_matched_correctly: bool
    hop1_val_routed_correctly: bool
    intermediate_state_similarity: float
    hop2_key_matched_correctly: bool
    hop2_val_routed_correctly: bool
    final_candidate_selected_correctly: bool
    final_token_correct: bool
    hop1_key_pos_pred: int
    hop1_val_pos_pred: int
    hop2_key_pos_pred: int
    hop2_val_pos_pred: int
    final_pred_token: int
    expected_intermediate_token: int
    expected_target_token: int


def trace_compositional_execution(
    model: Any,
    episode: CompositionalEpisode,
    tok: Any,
) -> CompositionalHopTrace:
    """Traces forward representations through hop 1 and hop 2."""
    inp = torch.tensor([episode.prompt_tokens], dtype=torch.long)
    h_ad, _ = model.forward_backbone(inp)
    h_norm = F.normalize(h_ad[0], p=2, dim=-1)

    # 1. Identify premise key positions in context
    premise_keys = [p[0] for p in episode.all_premise_pairs]
    premise_key_positions = []
    for k in premise_keys:
        k_enc = tok.encode(k)[0]
        matches = [i for i, t in enumerate(episode.prompt_tokens[:-1]) if t == k_enc]
        if matches:
            premise_key_positions.append(matches[0])

    # Hop 1: Query attends to premise keys
    h_q = h_norm[episode.query_key_pos]
    k_sims = [float(torch.dot(h_q, h_norm[kp]).item()) for kp in premise_key_positions]
    pred_h1_k = premise_key_positions[k_sims.index(max(k_sims))] if k_sims else 0
    h1_k_corr = (pred_h1_k == episode.hop1_key_pos)

    # Hop 1: Matched key attends to premise values
    premise_vals = [p[1] for p in episode.all_premise_pairs]
    premise_val_positions = []
    for v in premise_vals:
        v_enc = tok.encode(v)[0]
        matches = [i for i, t in enumerate(episode.prompt_tokens[:-1]) if t == v_enc]
        if matches:
            premise_val_positions.append(matches[0])

    h_mk = h_norm[pred_h1_k]
    v_sims = [float(torch.dot(h_mk, h_norm[vp]).item()) for vp in premise_val_positions]
    pred_h1_v = premise_val_positions[v_sims.index(max(v_sims))] if v_sims else 0
    h1_v_corr = (pred_h1_v == episode.hop1_val_pos)

    # Intermediate state is the retrieved representation at pred_h1_v
    intermediate_rep = h_ad[0, pred_h1_v : pred_h1_v + 1]  # [1, D]
    h_inter_norm = F.normalize(intermediate_rep[0], p=2, dim=-1)

    # Check intermediate state similarity to actual intermediate token embedding
    inter_tok_id = episode.intermediate_token
    token_emb = model.base_model.embedding.weight[inter_tok_id]
    inter_sim = float(torch.dot(h_inter_norm, F.normalize(token_emb, p=2, dim=-1)).item())

    # Hop 2: Intermediate state attends to second premise keys
    k2_sims = [float(torch.dot(h_inter_norm, h_norm[kp]).item()) for kp in premise_key_positions]
    pred_h2_k = premise_key_positions[k2_sims.index(max(k2_sims))] if k2_sims else 0
    h2_k_corr = (pred_h2_k == episode.hop2_key_pos)

    # Hop 2: Second key attends to final premise values
    h_mk2 = h_norm[pred_h2_k]
    v2_sims = [float(torch.dot(h_mk2, h_norm[vp]).item()) for vp in premise_val_positions]
    pred_h2_v = premise_val_positions[v2_sims.index(max(v2_sims))] if v2_sims else 0
    h2_v_corr = (pred_h2_v == episode.hop2_val_pos)

    # Final token emission via dynamic binding
    final_val_rep = h_ad[0, pred_h2_v : pred_h2_v + 1]
    
    # Candidate bundle extraction
    # Candidate values are all premise values in context
    cand_positions = []
    cand_tokens = []
    for v in premise_vals:
        v_enc = tok.encode(v)[0]
        pos_list = [i for i, t in enumerate(episode.prompt_tokens[:-1]) if t == v_enc]
        if pos_list and pos_list[0] not in cand_positions:
            cand_positions.append(pos_list[0])
            cand_tokens.append(v_enc)

    if not cand_positions:
        cand_positions = [0]
        cand_tokens = [episode.prompt_tokens[0]]

    cand_states = torch.stack([h_ad[0, p] for p in cand_positions], dim=0).unsqueeze(0)
    cand_mask = torch.ones((1, len(cand_positions)), dtype=torch.bool, device=h_ad.device)

    bind_logits, _ = model.compute_binding_scores(
        retrieved_value_rep=final_val_rep,
        candidate_states=cand_states,
        candidate_mask=cand_mask,
    )
    pred_cand_idx = bind_logits.argmax().item()
    pred_token = cand_tokens[pred_cand_idx]

    cand_corr = (pred_token == episode.target_token)
    final_corr = cand_corr

    return CompositionalHopTrace(
        episode_id=episode.episode_id,
        hop1_key_matched_correctly=h1_k_corr,
        hop1_val_routed_correctly=h1_v_corr,
        intermediate_state_similarity=inter_sim,
        hop2_key_matched_correctly=h2_k_corr,
        hop2_val_routed_correctly=h2_v_corr,
        final_candidate_selected_correctly=cand_corr,
        final_token_correct=final_corr,
        hop1_key_pos_pred=pred_h1_k,
        hop1_val_pos_pred=pred_h1_v,
        hop2_key_pos_pred=pred_h2_k,
        hop2_val_pos_pred=pred_h2_v,
        final_pred_token=pred_token,
        expected_intermediate_token=episode.intermediate_token,
        expected_target_token=episode.target_token,
    )
