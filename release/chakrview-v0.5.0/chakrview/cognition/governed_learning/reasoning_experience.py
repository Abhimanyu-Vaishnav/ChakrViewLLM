"""Step 488: Governed Reasoning Experience Schema for ChakrView RIL."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from typing import Dict, List, Optional, Any


@dataclasses.dataclass
class ReasoningExperience:
    experience_id: str
    task_id: str
    task_family: str
    input_representation: str
    trajectory_summary: List[Dict[str, Any]]
    intermediate_state_metadata: Dict[str, float]
    reasoning_depth: int
    outcome: str
    expected_outcome: str
    success: bool
    confidence: float
    uncertainty: float
    failure_category: str
    resource_summary: Dict[str, float]
    model_version: str
    candidate_version: str
    provenance: Dict[str, Any]
    evaluation_split: str

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def create(
        cls,
        task_id: str,
        task_family: str,
        input_representation: str,
        trajectory_summary: List[Dict[str, Any]],
        intermediate_state_metadata: Dict[str, float],
        reasoning_depth: int,
        outcome: str,
        expected_outcome: str,
        success: bool,
        confidence: float,
        uncertainty: float,
        failure_category: str,
        resource_summary: Dict[str, float],
        model_version: str,
        candidate_version: str,
        evaluation_split: str,
        parent_id: Optional[str] = None,
    ) -> ReasoningExperience:
        payload = f"{task_id}:{task_family}:{input_representation}:{model_version}:{evaluation_split}:{time.time()}"
        exp_id = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        provenance = {
            "source": "REASONING_TRAJECTORY_EVALUATION",
            "parent_id": parent_id,
            "timestamp": time.time(),
            "component": "RIL_governed_subsystem"
        }
        return cls(
            experience_id=exp_id,
            task_id=task_id,
            task_family=task_family,
            input_representation=input_representation,
            trajectory_summary=trajectory_summary,
            intermediate_state_metadata=intermediate_state_metadata,
            reasoning_depth=reasoning_depth,
            outcome=outcome,
            expected_outcome=expected_outcome,
            success=success,
            confidence=confidence,
            uncertainty=uncertainty,
            failure_category=failure_category,
            resource_summary=resource_summary,
            model_version=model_version,
            candidate_version=candidate_version,
            provenance=provenance,
            evaluation_split=evaluation_split
        )
