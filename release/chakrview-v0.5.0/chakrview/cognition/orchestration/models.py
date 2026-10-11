"""
Strongly Typed Data Models for Adaptive Cognitive Orchestration (Step 28).

Defines foundational contracts for resource-aware, minimum-sufficient cognitive orchestration:
- WorkloadClass: Deterministic task workload categorization
- TaskPlan: Bounded, role-driven cognitive execution plan
- ResourceAllocationDecision: Resource-aware compute and agent budget assignment
- OrchestrationState: Lifecycle tracking of multi-round deliberation
- SafePublicOrchestrationTrace: Sanitized telemetry audit trace without CoT leakage

CRITICAL ARCHITECTURAL AXIOMS:
1. ORCHESTRATOR != AUTHORITY, AGENT != AUTHORITY, NODE != AUTHORITY,
   REMOTE_AGENT != AUTHORITY, MESSAGE != AUTHORITY, CONSENSUS != AUTHORITY.
2. MINIMUM SUFFICIENT BOUNDED COGNITION:
   More agents != better intelligence. Allocate only the cognitive resources
   necessary for the specific workload class.
3. ZERO RUNTIME WEIGHT MUTATION:
   Model weights are immutable (Delta W == 0). Pre- and post-cycle SHA-256 weight
   fingerprints must match identically or fail closed.
4. STRICT BOUNDS:
   All agent counts, node counts, deliberation rounds, and retries are hard-ceiling protected.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import time
from typing import Dict, List, Optional, Any, Set
import uuid

from chakrview.cognition.unified.models import DecisionState, CognitiveTaskType
from chakrview.cognition.federated.models import AgentRole, ConflictState
from chakrview.cognition.adaptation.profiles import ResourceProfile


class WorkloadClass(str, Enum):
    """
    Formal workload categorization for minimum-sufficient cognitive allocation.
    """
    SIMPLE = "SIMPLE"                                     # Direct/declarative recall; minimal single-agent allocation
    STANDARD = "STANDARD"                                 # Balanced factual or analytical task; standard multi-agent set
    COMPLEX = "COMPLEX"                                   # Multi-step reasoning/planning; expanded agent set & deliberation
    AMBIGUOUS = "AMBIGUOUS"                               # Underspecified/uncertain; evidence gathering & critique
    CONFLICTED = "CONFLICTED"                             # Contradictions present; contradiction analysis & minority preservation
    VERIFICATION_REQUIRED = "VERIFICATION_REQUIRED"       # High-risk claim/action; mandatory verification stage
    RESOURCE_CONSTRAINED = "RESOURCE_CONSTRAINED"         # Hardware or network constrained; reduced federation, safety preserved


# Hard architectural limits for Step 28
MAX_ORCHESTRATION_AGENTS = 8
MAX_ORCHESTRATION_NODES = 8
MAX_DELIBERATION_ROUNDS = 3
MAX_ORCHESTRATION_RETRIES = 2
MAX_ORCHESTRATION_TASKS = 16


@dataclass
class TaskPlan:
    """
    Bounded, role-driven execution plan produced by AdaptiveTaskPlanner.
    """
    plan_id: str
    task_id: str
    workload_class: WorkloadClass
    primary_objective: str
    required_roles: List[AgentRole]
    optional_roles: List[AgentRole] = field(default_factory=list)
    role_dependencies: Dict[str, List[str]] = field(default_factory=dict)
    execution_budget_ms: int = 5000
    max_deliberation_rounds: int = 1
    max_agent_count: int = 2
    max_node_count: int = 1
    verification_required: bool = False
    termination_conditions: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.max_agent_count > MAX_ORCHESTRATION_AGENTS:
            raise ValueError(f"max_agent_count exceeds hard limit of {MAX_ORCHESTRATION_AGENTS}")
        if self.max_node_count > MAX_ORCHESTRATION_NODES:
            raise ValueError(f"max_node_count exceeds hard limit of {MAX_ORCHESTRATION_NODES}")
        if self.max_deliberation_rounds > MAX_DELIBERATION_ROUNDS:
            raise ValueError(f"max_deliberation_rounds exceeds hard limit of {MAX_DELIBERATION_ROUNDS}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "task_id": self.task_id,
            "workload_class": self.workload_class.value,
            "primary_objective": self.primary_objective,
            "required_roles": [r.value for r in self.required_roles],
            "optional_roles": [r.value for r in self.optional_roles],
            "role_dependencies": self.role_dependencies,
            "execution_budget_ms": self.execution_budget_ms,
            "max_deliberation_rounds": self.max_deliberation_rounds,
            "max_agent_count": self.max_agent_count,
            "max_node_count": self.max_node_count,
            "verification_required": self.verification_required,
            "termination_conditions": self.termination_conditions,
            "created_at": self.created_at,
        }


@dataclass
class ResourceAllocationDecision:
    """
    Deterministic assignment of computational and agent resources.
    """
    decision_id: str
    task_id: str
    workload_class: WorkloadClass
    resource_profile: ResourceProfile
    allocated_agent_count: int
    allocated_node_count: int
    role_distribution: Dict[str, int]
    selected_node_ids: List[str]
    max_rounds: int
    execution_budget_ms: int
    retry_budget: int
    verification_budget: int
    deliberation_budget: int
    explanation: str
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.allocated_agent_count > MAX_ORCHESTRATION_AGENTS:
            raise ValueError(f"allocated_agent_count exceeds hard limit of {MAX_ORCHESTRATION_AGENTS}")
        if self.allocated_node_count > MAX_ORCHESTRATION_NODES:
            raise ValueError(f"allocated_node_count exceeds hard limit of {MAX_ORCHESTRATION_NODES}")
        if self.max_rounds > MAX_DELIBERATION_ROUNDS:
            raise ValueError(f"max_rounds exceeds hard limit of {MAX_DELIBERATION_ROUNDS}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "task_id": self.task_id,
            "workload_class": self.workload_class.value,
            "resource_profile": self.resource_profile.value,
            "allocated_agent_count": self.allocated_agent_count,
            "allocated_node_count": self.allocated_node_count,
            "role_distribution": self.role_distribution,
            "selected_node_ids": self.selected_node_ids,
            "max_rounds": self.max_rounds,
            "execution_budget_ms": self.execution_budget_ms,
            "retry_budget": self.retry_budget,
            "verification_budget": self.verification_budget,
            "deliberation_budget": self.deliberation_budget,
            "explanation": self.explanation,
            "timestamp": self.timestamp,
        }


@dataclass
class OrchestrationState:
    """
    Dynamic state of the orchestration lifecycle across deliberation rounds.
    """
    orchestration_id: str
    task_id: str
    tenant_id: str
    session_id: str
    workload_class: WorkloadClass
    current_round: int = 0
    max_rounds: int = 1
    participating_agents: Set[str] = field(default_factory=set)
    participating_nodes: Set[str] = field(default_factory=set)
    evidence_count: int = 0
    conflicts_detected: int = 0
    minority_perspectives_count: int = 0
    verification_passed: Optional[bool] = None
    decision_state: DecisionState = DecisionState.INSUFFICIENT_INFORMATION
    confidence: float = 0.5
    termination_reason: str = "initialized"
    is_terminal: bool = False
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

    def advance_round(self, reason: str) -> None:
        self.current_round += 1
        self.termination_reason = reason

    def mark_terminal(self, decision: DecisionState, confidence: float, reason: str) -> None:
        self.decision_state = decision
        self.confidence = max(0.0, min(1.0, confidence))
        self.termination_reason = reason
        self.is_terminal = True
        self.completed_at = time.time()


@dataclass
class SafePublicOrchestrationTrace:
    """
    Sanitized audit trace for public telemetry.
    Strictly conceals private chain-of-thought, hidden weights, and credentials.
    """
    trace_id: str
    task_id: str
    tenant_id: str
    session_id: str
    workload_class: str
    resource_profile: str
    allocated_agents: int
    allocated_nodes: int
    rounds_executed: int
    conflicts_detected: int
    minority_evidence_preserved: int
    verification_invoked: bool
    verification_passed: Optional[bool]
    final_decision_state: str
    confidence: float
    total_latency_ms: float
    termination_reason: str
    weights_modified: bool = False  # Permanent core invariant: ALWAYS False
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
