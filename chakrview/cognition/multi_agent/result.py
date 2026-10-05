"""
ChakrView Step 113: Multi-Agent Execution Results and Artifacts.

Defines schemas for worker execution output, reviewer critique,
and coordinator synthesis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from chakrview.cognition.multi_agent.contracts import WorkerRole, ReviewerVerdict


class WorkerExecutionStatus(str, Enum):
    """Status of worker execution."""
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    REJECTED = "REJECTED"
    MALFORMED_OUTPUT = "MALFORMED_OUTPUT"
    UNAUTHORIZED_TOOL = "UNAUTHORIZED_TOOL"


@dataclass
class WorkerResult:
    """
    Standard output produced by any specialized cognitive worker.
    """
    worker_id: str
    task_id: str
    role: WorkerRole
    status: WorkerExecutionStatus
    evidence: Dict[str, Any] = field(default_factory=dict)
    proposed_changes: Dict[str, str] = field(default_factory=dict)  # file_path -> content
    verification_output: Optional[Dict[str, Any]] = None
    failures: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 1.0
    context_tokens_used: int = 0
    error: Optional[str] = None

    def validate_against_schema(self, expected_schema: Dict[str, Any]) -> None:
        """Validate result against expected contract schema."""
        if not expected_schema:
            return
        for req_field in expected_schema.get("required_fields", []):
            if hasattr(self, req_field) and getattr(self, req_field) is not None:
                continue
            if req_field in self.evidence:
                continue
            raise ValueError(f"Missing required field '{req_field}' in worker output schema")



@dataclass
class ReviewerCritique:
    """
    Independent critique produced by the REVIEWER worker.
    """
    verdict: ReviewerVerdict
    assumptions_checked: List[str]
    identified_risks: List[str]
    evidence_verified: bool
    confidence: float
    reasoning_summary: str
    rejection_reasons: List[str] = field(default_factory=list)


@dataclass
class SynthesisResult:
    """
    Final synthesis produced by the COORDINATOR / SYNTHESIZER.
    """
    objective: str
    completed_tasks: List[str]
    rejected_tasks: List[str]
    changed_files: List[str]
    verification_results: Dict[str, Any]
    failures_encountered: List[Dict[str, Any]]
    recovery_actions: List[str]
    reviewer_verdict: ReviewerVerdict
    remaining_risks: List[str]
    confidence: float
    final_status: str  # "SUCCESS" or "FAILED"
