"""
Verification Loop for ChakrView Reasoning (Step 19).

Provides explicit verification of expected results vs actual observations,
diagnosing failure causes and recommending structured revisions.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
import time
from typing import Dict, List, Optional, Any
import uuid


class VerificationStatus(str, Enum):
    """Result of an explicit verification check."""
    PASS = "PASS"
    FAIL = "FAIL"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class VerificationCriteria:
    """Explicit condition required for verification success."""
    criterion_id: str = field(default_factory=lambda: f"crit_{uuid.uuid4().hex[:6]}")
    description: str = ""
    expected_value: Any = None
    tolerance: Optional[float] = None
    criterion_type: str = "exact_match"  # exact_match, numerical_tolerance, contains, truthy

    def to_dict(self) -> Dict[str, Any]:
        return {
            "criterion_id": self.criterion_id,
            "description": self.description,
            "expected_value": self.expected_value,
            "tolerance": self.tolerance,
            "criterion_type": self.criterion_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerificationCriteria":
        return cls(**data)


@dataclass
class VerificationResult:
    """
    Formally documented verification record.
    """
    verification_id: str = field(default_factory=lambda: f"vrf_{uuid.uuid4().hex[:8]}")
    decision_id: str = ""
    expected_result: Any = None
    actual_result: Any = None
    criteria_evaluated: List[Dict[str, Any]] = field(default_factory=list)
    status: VerificationStatus = VerificationStatus.PASS
    evidence_ids: List[str] = field(default_factory=list)
    failure_reason: Optional[str] = None
    revision_recommendation: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def is_passed(self) -> bool:
        return self.status == VerificationStatus.PASS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verification_id": self.verification_id,
            "decision_id": self.decision_id,
            "expected_result": self.expected_result,
            "actual_result": self.actual_result,
            "criteria_evaluated": list(self.criteria_evaluated),
            "status": self.status.value,
            "evidence_ids": list(self.evidence_ids),
            "failure_reason": self.failure_reason,
            "revision_recommendation": self.revision_recommendation,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerificationResult":
        d = dict(data)
        d["status"] = VerificationStatus(d["status"])
        return cls(**d)


class VerificationEngine:
    """
    Evaluates execution and reasoning outputs against expectations and explicit criteria.
    """

    def verify(
        self,
        decision_id: str,
        expected_result: Any,
        actual_result: Any,
        criteria: Optional[List[VerificationCriteria]] = None,
    ) -> VerificationResult:
        """
        Verify whether actual result satisfies expected result and all criteria.
        """
        if criteria is None:
            # Generate default criterion
            criteria = [
                VerificationCriteria(
                    description="Default matching check",
                    expected_value=expected_result,
                    criterion_type="exact_match" if not isinstance(expected_result, (int, float)) else "numerical_tolerance",
                    tolerance=1e-5 if isinstance(expected_result, (int, float)) else None,
                )
            ]

        eval_records: List[Dict[str, Any]] = []
        overall_pass = True
        failure_reasons = []

        for crit in criteria:
            passed = False
            crit_record = crit.to_dict()

            if crit.criterion_type == "numerical_tolerance":
                try:
                    exp_num = float(crit.expected_value)
                    act_num = float(actual_result)
                    tol = crit.tolerance if crit.tolerance is not None else 1e-5
                    diff = abs(exp_num - act_num)
                    passed = diff <= tol
                    crit_record["diff"] = diff
                except (ValueError, TypeError):
                    passed = False
                    crit_record["error"] = "Non-numeric values in numerical comparison"

            elif crit.criterion_type == "contains":
                passed = str(crit.expected_value).lower() in str(actual_result).lower()

            elif crit.criterion_type == "truthy":
                passed = bool(actual_result)

            else:  # exact_match
                passed = (crit.expected_value == actual_result)

            crit_record["passed"] = passed
            eval_records.append(crit_record)

            if not passed:
                overall_pass = False
                failure_reasons.append(
                    f"Criterion '{crit.description}' failed: expected {crit.expected_value}, got {actual_result}."
                )

        if overall_pass:
            return VerificationResult(
                decision_id=decision_id,
                expected_result=expected_result,
                actual_result=actual_result,
                criteria_evaluated=eval_records,
                status=VerificationStatus.PASS,
            )
        else:
            reason = "; ".join(failure_reasons)
            recommendation = "Revise hypothesis premises or select alternative capability candidate."
            return VerificationResult(
                decision_id=decision_id,
                expected_result=expected_result,
                actual_result=actual_result,
                criteria_evaluated=eval_records,
                status=VerificationStatus.FAIL,
                failure_reason=reason,
                revision_recommendation=recommendation,
            )
