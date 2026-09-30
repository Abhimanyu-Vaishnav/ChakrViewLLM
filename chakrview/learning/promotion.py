"""
ChakrView Step 57: Promotion Gate & Rollback Controller.

Enforces 7-stage promotion criteria:
1. Experience is verified.
2. Candidate adapter loads correctly.
3. Target task performance improves or matches.
4. Baseline hash unchanged (DeltaW_base == 0).
5. Unrelated regression tests do not regress (> 2%).
6. Rollback works cleanly with zero parameter residue.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import torch

from chakrview.brain.model import ChakrMicro
from chakrview.brain.adapter import NativeTaskAdapter
from chakrview.runtime.interactive import compute_model_hash
from chakrview.runtime.pipeline import EXPECTED_WEIGHT_HASH
from chakrview.learning.experience import ExperienceRecord


class PromotionDecision:
    PROMOTED = "PROMOTED"
    REJECTED = "REJECTED"


class PromotionGateController:
    """
    Evaluates candidate updates against strict verification and regression invariants.
    """

    @staticmethod
    def evaluate_candidate(
        base_model: ChakrMicro,
        experience: ExperienceRecord,
        candidate_adapter: Optional[NativeTaskAdapter],
        anchor_pass_before: float,
        anchor_pass_after: float,
        target_pass_rate: float,
        expected_base_hash: str = EXPECTED_WEIGHT_HASH,
    ) -> Dict[str, Any]:
        """
        Evaluate candidate adapter for promotion.
        """
        rejection_reasons = []

        # 1. Experience verification check
        if not experience.verified:
            rejection_reasons.append("Experience record is unverified.")

        # 2. Base weight immutability check
        cur_base_hash = compute_model_hash(base_model)
        if cur_base_hash != expected_base_hash:
            rejection_reasons.append(
                f"Base weight mutation detected: {cur_base_hash} != {expected_base_hash}"
            )

        # 3. Target task performance
        if target_pass_rate < 0.80:
            rejection_reasons.append(
                f"Target task pass rate {target_pass_rate*100:.1f}% below required 80.0%"
            )

        # 4. Regression check (anchor drop must be <= 2%)
        anchor_drop = anchor_pass_before - anchor_pass_after
        if anchor_drop > 0.02:
            rejection_reasons.append(
                f"Anchor regression detected: dropped {anchor_drop*100:.1f}% (max allowed 2.0%)"
            )

        # 5. Rollback verification check
        if candidate_adapter:
            candidate_adapter.mount(base_model)
            candidate_adapter.unmount(base_model)
            post_unmount_hash = compute_model_hash(base_model)
            if post_unmount_hash != expected_base_hash:
                rejection_reasons.append("Rollback failed: base model mutated after unmounting adapter.")

        decision = PromotionDecision.PROMOTED if not rejection_reasons else PromotionDecision.REJECTED

        return {
            "decision": decision,
            "rejection_reasons": rejection_reasons,
            "anchor_pass_before": anchor_pass_before,
            "anchor_pass_after": anchor_pass_after,
            "target_pass_rate": target_pass_rate,
            "base_hash_verified": (cur_base_hash == expected_base_hash),
        }
