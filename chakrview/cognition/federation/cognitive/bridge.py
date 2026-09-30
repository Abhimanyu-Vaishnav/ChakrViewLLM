"""
Federated Reasoning Bridge for Step 42.

Translates CognitiveSteps from the cognitive DAG into Step 38 WorkUnits
and parses TaskResultEnvelopes back into step result dicts.

This is the translation layer between the cognitive orchestration layer and
the production federated task execution infrastructure.

AXIOM: ADVERTISEMENT != PERMISSION
  Converting a cognitive step to a WorkUnit does NOT grant execution rights.
  All WorkUnits still require sovereign ExecutionGrant authorization through
  the CapabilityGate before they execute.
"""

import logging
import uuid
from typing import Any, Dict, Optional

from chakrview.cognition.federation.cognitive.models import (
    CognitiveContextEnvelope,
    CognitiveStep,
)
from chakrview.cognition.federation.tasks.models import (
    ResourceRequirements,
    WorkUnit,
)

logger = logging.getLogger("chakrview.federation.cognitive.bridge")


class FederatedReasoningBridge:
    """
    Translates CognitiveSteps to WorkUnits and parses TaskResultEnvelopes
    back into cognitive step result dicts.
    """

    DEFAULT_CPU_CORES: float = 1.0
    DEFAULT_MEMORY_MB: int = 256

    def create_work_unit(
        self,
        step: CognitiveStep,
        task_id: str,
        sequence: int,
        envelope: Optional[CognitiveContextEnvelope] = None,
        tenant_id: str = "default",
    ) -> WorkUnit:
        """
        Create a WorkUnit representing a cognitive step for federated dispatch.

        The CognitiveContextEnvelope is serialised into input_payload so the
        worker receives bounded context without exceeding the wire frame ceiling.
        """
        payload: Dict[str, Any] = {
            "step_id": step.step_id,
            "role": step.role.value,
            "objective": step.input_payload.get("objective", ""),
            "input_payload": step.input_payload,
        }
        if envelope is not None:
            payload["context_envelope"] = envelope.to_dict()

        reqs = ResourceRequirements(
            capability_id=step.capability_id,
            min_cpu_cores=self.DEFAULT_CPU_CORES,
            min_memory_mb=self.DEFAULT_MEMORY_MB,
            tenant_id=tenant_id,
        )

        unit = WorkUnit(
            unit_id=f"cu_{step.step_id}_{uuid.uuid4().hex[:6]}",
            task_id=task_id,
            sequence=sequence,
            capability_id=step.capability_id,
            input_payload=payload,
            requirements=reqs,
            attempt=1,
        )

        logger.debug(
            "Bridge: WorkUnit '%s' for step '%s' (role=%s cap=%s)",
            unit.unit_id,
            step.step_id,
            step.role.value,
            step.capability_id,
        )
        return unit

    def parse_result_envelope(
        self,
        result_envelope: Any,
        step_id: str,
    ) -> Dict[str, Any]:
        """
        Parse a Step 38 TaskResultEnvelope into a cognitive step result dict.

        Expected TaskResultEnvelope fields used:
            result_data   dict   The capability output dict
            error_message str    Optional error if execution failed
            worker_id     str    Executing node ID
        """
        # Determine success from status
        from chakrview.cognition.federation.tasks.models import WorkUnitState
        status = getattr(result_envelope, "status", None)
        success = status == WorkUnitState.COMPLETED

        if not success:
            err = getattr(result_envelope, "error_message", None) or "Worker execution failed"
            return {
                "step_id": step_id,
                "success": False,
                "error": err,
                "node_id": getattr(result_envelope, "worker_id", "unknown"),
                "conclusion": "",
                "evidence": [],
                "hypotheses": [],
            }

        output = getattr(result_envelope, "result_data", {}) or {}
        return {
            "step_id": step_id,
            "success": True,
            "node_id": getattr(result_envelope, "worker_id", "unknown"),
            "conclusion": output.get("conclusion", ""),
            "evidence": output.get("evidence", []),
            "hypotheses": output.get("hypotheses", []),
            "output_tokens": output.get("output_tokens", []),
            "raw_output": output,
        }
