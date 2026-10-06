"""Step 271: Emission Mechanism Verification & Controlled Interventions.

Traces the entire neural pipeline:
query representation
        ↓
selected key
        ↓
associated value position
        ↓
selected value representation
        ↓
contextual readout
        ↓
vocabulary logits
        ↓
predicted token

Executes the 4 mandatory controlled interventions:
A. Correct Value Intervention:
   Feed the representation from the correct associated value position.
B. Wrong Value Intervention:
   Replace it with the representation from an incorrect value position.
C. Position Intervention:
   Move the semantic value to another position; verify emission follows the representation, not absolute position.
D. Readout Ablation:
   Bypass/remove the contextual readout (pass raw backbone state to frozen lm_head); verify performance.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro
from chakrview.cognition.contextual_vocabulary_readout import (
    ChakrMicroWithContextualReadout,
)
from chakrview.cognition.randomized_associative_episodes import (
    RandomizedAssociativeEnvironment,
    RandomizedAssociativeEpisode,
)


@dataclasses.dataclass
class InterventionComparisonResult:
    num_samples: int
    correct_value_mean_logit: float
    correct_value_mean_prob: float
    correct_value_mean_rank: float
    wrong_value_mean_logit: float
    wrong_value_mean_prob: float
    wrong_value_mean_rank: float
    logit_drop_on_wrong_value: float
    position_intervention_invariance: float
    readout_ablation_drop: float
    is_causally_dependent: bool


def run_emission_mechanism_verification(
    candidate: ChakrMicroWithContextualReadout,
    env: Optional[RandomizedAssociativeEnvironment] = None,
    seed: int = 42,
    num_samples: int = 15,
) -> InterventionComparisonResult:
    """Executes the four interventions A, B, C, D."""
    torch.manual_seed(seed)
    if env is None:
        env = RandomizedAssociativeEnvironment(seed=seed)
    tok = env.tok

    candidate.eval()

    corr_logits, corr_probs, corr_ranks = [], [], []
    wrong_logits, wrong_probs, wrong_ranks = [], [], []
    pos_inv_correct = 0
    ablation_drops = []

    with torch.no_grad():
        for i in range(num_samples):
            ep = env.generate_episode("train", 2, episode_idx=10000 + i)
            inp = torch.tensor([ep.prompt_tokens], dtype=torch.long)
            tgt_t = ep.target_token

            adapted_h, raw_h = candidate.forward_backbone(inp)

            # Find correct value position and wrong value position
            corr_vp = ep.associated_val_pos
            # Pick a wrong value position from the other pair
            all_val_positions = []
            for p in ep.pairs:
                v_enc = tok.encode(p.val)[0]
                vl = [j for j, t in enumerate(ep.prompt_tokens[:-1]) if t == v_enc]
                if vl: all_val_positions.append(vl[0])

            wrong_val_candidates = [p for p in all_val_positions if p != corr_vp]
            wrong_vp = wrong_val_candidates[0] if wrong_val_candidates else (corr_vp + 1 if corr_vp + 1 < adapted_h.shape[1] else corr_vp - 1)

            # A. Correct Value Intervention
            h_corr = adapted_h[0, corr_vp].unsqueeze(0)
            log_corr = candidate.forward_from_value_rep(h_corr)
            p_corr = F.softmax(log_corr[0], dim=-1)

            c_log = float(log_corr[0, tgt_t].item())
            c_pr = float(p_corr[tgt_t].item())
            c_rk = int((log_corr[0] > log_corr[0, tgt_t]).sum().item()) + 1

            corr_logits.append(c_log)
            corr_probs.append(c_pr)
            corr_ranks.append(c_rk)

            # B. Wrong Value Intervention
            h_wrong = adapted_h[0, wrong_vp].unsqueeze(0)
            log_wrong = candidate.forward_from_value_rep(h_wrong)
            p_wrong = F.softmax(log_wrong[0], dim=-1)

            w_log = float(log_wrong[0, tgt_t].item())
            w_pr = float(p_wrong[tgt_t].item())
            w_rk = int((log_wrong[0] > log_wrong[0, tgt_t]).sum().item()) + 1

            wrong_logits.append(w_log)
            wrong_probs.append(w_pr)
            wrong_ranks.append(w_rk)

            # C. Position Intervention
            # Generate same semantic pair under reverse layout
            ep_rev = env.generate_episode("train", 2, layout_name="reverse_order", episode_idx=10000 + i)
            inp_rev = torch.tensor([ep_rev.prompt_tokens], dtype=torch.long)
            ad_rev, _ = candidate.forward_backbone(inp_rev)
            h_rev_corr = ad_rev[0, ep_rev.associated_val_pos].unsqueeze(0)
            log_rev = candidate.forward_from_value_rep(h_rev_corr)
            if torch.argmax(log_rev[0]).item() == ep_rev.target_token:
                pos_inv_correct += 1

            # D. Readout Ablation (use frozen lm_head on raw hidden state)
            raw_logits = candidate.base_model.lm_head(raw_h[0, -1:, :])
            raw_tgt_logit = float(raw_logits[0, tgt_t].item() if raw_logits.ndim == 2 else raw_logits[0, 0, tgt_t].item())
            ablation_drops.append(c_log - raw_tgt_logit)

    mean_c_log = sum(corr_logits) / max(1, len(corr_logits))
    mean_c_pr = sum(corr_probs) / max(1, len(corr_probs))
    mean_c_rk = sum(corr_ranks) / max(1, len(corr_ranks))

    mean_w_log = sum(wrong_logits) / max(1, len(wrong_logits))
    mean_w_pr = sum(wrong_probs) / max(1, len(wrong_probs))
    mean_w_rk = sum(wrong_ranks) / max(1, len(wrong_ranks))

    drop_on_wrong = mean_c_log - mean_w_log
    mean_abl_drop = sum(ablation_drops) / max(1, len(ablation_drops))
    pos_inv_acc = pos_inv_correct / max(1, num_samples)

    is_causal = (mean_c_log > mean_w_log)

    return InterventionComparisonResult(
        num_samples=num_samples,
        correct_value_mean_logit=mean_c_log,
        correct_value_mean_prob=mean_c_pr,
        correct_value_mean_rank=mean_c_rk,
        wrong_value_mean_logit=mean_w_log,
        wrong_value_mean_prob=mean_w_pr,
        wrong_value_mean_rank=mean_w_rk,
        logit_drop_on_wrong_value=drop_on_wrong,
        position_intervention_invariance=pos_inv_acc,
        readout_ablation_drop=mean_abl_drop,
        is_causally_dependent=is_causal,
    )
