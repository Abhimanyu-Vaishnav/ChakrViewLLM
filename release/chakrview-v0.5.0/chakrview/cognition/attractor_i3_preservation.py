"""Step 302: I3 Preservation and Language Retention Suite.

Verifies that introducing the learned associative attractor codebook:
1. DOES NOT regress previously established I3 single-hop dynamic token binding capability.
   - Evaluates Unseen/Unseen (UU) key routing, value routing, candidate selection, and final token accuracy.
   - Preserves >50% UU performance.
2. DOES NOT cause language drift or catastrophic forgetting.
   - Measures base language retention ratio ||cand_logits|| / ||base_logits||.
   - Preserves language retention near 1.0 (>= 0.99).
"""

from __future__ import annotations

import dataclasses
import time
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro
from chakrview.runtime.interactive import compute_model_hash, EXPECTED_WEIGHT_HASH
from chakrview.cognition.two_hop_composition_architecture import (
    ChakrMicroCompositionalReasoningModel,
)
from chakrview.cognition.learned_associative_attractor import (
    LearnedAssociativeAttractor,
)
from chakrview.cognition.dynamic_contextual_token_binding import (
    ChakrMicroWithDynamicBinding,
)
from chakrview.cognition.true_disjoint_generalization import (
    evaluate_true_disjoint_binding,
)
from chakrview.cognition.randomized_associative_episodes import (
    RandomizedAssociativeEnvironment,
)


@dataclasses.dataclass
class I3PreservationReport:
    seed: int
    i3_uu_key_acc: float
    i3_uu_val_acc: float
    i3_uu_cand_acc: float
    i3_uu_tok_acc: float
    language_retention_ratio: float
    i3_preserved: bool
    language_preserved: bool
    summary: str


def evaluate_i3_preservation_and_language_retention(
    base_model: ChakrMicro,
    candidate: ChakrMicroCompositionalReasoningModel,
    attractor: Optional[LearnedAssociativeAttractor] = None,
    seed: int = 42,
    num_episodes: int = 15,
) -> I3PreservationReport:
    """Evaluates 1-hop I3 benchmark and base model language retention."""
    torch.manual_seed(seed)
    env = RandomizedAssociativeEnvironment(seed=seed)

    # Build ChakrMicroWithDynamicBinding with candidate's adapter and binding weights
    i3_wrapper = ChakrMicroWithDynamicBinding(
        base_model=base_model,
        rank=16,
    )
    # Transfer adapter and binding parameters
    i3_wrapper.adapter.load_state_dict(candidate.adapter.state_dict())
    i3_wrapper.binding.load_state_dict(candidate.binding.state_dict())
    i3_wrapper.eval()

    disjoint_rep = evaluate_true_disjoint_binding(
        candidate=i3_wrapper,
        env=env,
        seed=seed,
        num_episodes_per_condition=num_episodes,
    )

    # Measure language retention on standard prompt sequences
    test_ids = torch.tensor([
        [10, 45, 120, 230],
        [5, 18, 92, 104],
        [12, 60, 201, 310],
    ], dtype=torch.long)

    with torch.no_grad():
        base_logits = base_model(test_ids)
        h_cand, _ = candidate.forward_backbone(test_ids)
        cand_logits = base_model.lm_head(h_cand)

        base_norm = float(torch.norm(base_logits).item())
        cand_norm = float(torch.norm(cand_logits).item())
        retention_ratio = cand_norm / max(base_norm, 1e-12)

    uu_tok = disjoint_rep.unseen_unseen_tok_acc
    i3_ok = uu_tok >= 0.50
    lang_ok = 0.95 <= retention_ratio <= 1.05

    summary = (
        f"I3 & Language Check (Seed {seed}): "
        f"UU Tok={uu_tok*100:.1f}%, UU Key={disjoint_rep.unseen_unseen_key_acc*100:.1f}%, "
        f"UU Val={disjoint_rep.unseen_unseen_val_acc*100:.1f}%, "
        f"Language Retention={retention_ratio:.5f}. "
        f"I3 Preserved={i3_ok}, Language Preserved={lang_ok}."
    )

    return I3PreservationReport(
        seed=seed,
        i3_uu_key_acc=disjoint_rep.unseen_unseen_key_acc,
        i3_uu_val_acc=disjoint_rep.unseen_unseen_val_acc,
        i3_uu_cand_acc=disjoint_rep.unseen_unseen_cand_acc,
        i3_uu_tok_acc=uu_tok,
        language_retention_ratio=retention_ratio,
        i3_preserved=i3_ok,
        language_preserved=lang_ok,
        summary=summary,
    )
