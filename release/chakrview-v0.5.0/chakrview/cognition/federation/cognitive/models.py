"""
Core data models for the Federated Cognitive Orchestration layer (Step 42).

Defines:
- CognitiveRole: Typed agent roles for cognitive DAG steps
- CognitiveStepState: Deterministic per-step lifecycle
- CognitiveStep: A typed node in the cognitive DAG
- CognitiveTaskGraph: Dependency-ordered DAG of cognitive steps
- CognitiveContextEnvelope: Bounded, secret-scanned inter-node context carrier
- CognitiveEpisodeState: Episode lifecycle state machine
- CognitiveEpisode: Multi-turn federated collaborative reasoning session
- ConflictRecord: Anti-majority minority evidence record
- SynthesisResult: Multi-node reasoning aggregation result

AXIOMS:
- DATA != AUTHORITY: Context envelopes are passive data, never authority.
- LOCAL_POLICY > CONSENSUS_DECISION: Local gates govern all executions.
- ZERO NEURAL WEIGHT MUTATION: ΔW = 0 at all times.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Any

from chakrview.cognition.federation.cognitive.errors import (
    CognitiveContextOverflowError,
    CognitiveContextTenantViolationError,
    CognitiveStepDependencyError,
    CognitiveGraphCycleError,
    CognitiveEpisodeStateError,
    SecretLeakageInContextError,
)

# ─────────────────────────────────────────────────────────────────────────────
# Token Budget Constants
# ─────────────────────────────────────────────────────────────────────────────

MAX_CONTEXT_TOKENS: int = 512          # Hard ceiling from ChakrMicro architecture
DEFAULT_OUTPUT_TOKEN_BUDGET: int = 64  # Reserved for generated output tokens
MAX_ENVELOPE_CONTEXT_TOKENS: int = MAX_CONTEXT_TOKENS - DEFAULT_OUTPUT_TOKEN_BUDGET  # 448

# ─────────────────────────────────────────────────────────────────────────────
# Registered Cognitive Capability IDs
# ─────────────────────────────────────────────────────────────────────────────

CAPABILITY_NEURAL_INFERENCE = "cognitive_neural_inference"
CAPABILITY_ANALYST = "cognitive_analyst_reasoning"
CAPABILITY_RESEARCHER = "cognitive_researcher_retrieval"
CAPABILITY_CRITIC = "cognitive_critic_evaluation"
CAPABILITY_SYNTHESIZER = "cognitive_synthesizer"
CAPABILITY_VERIFIER = "cognitive_verifier"

# ─────────────────────────────────────────────────────────────────────────────
# Secret Scanning Constants (CT-07 Mitigation)
# ─────────────────────────────────────────────────────────────────────────────

PROHIBITED_CONTEXT_KEYWORDS = (
    "private_key",
    "secret_key",
    "BEGIN PRIVATE KEY",
    "token_secret",
    "auth_token",
    "password",
)


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class CognitiveRole(str, Enum):
    """Typed agent roles for federated reasoning steps."""
    ANALYST = "analyst"
    RESEARCHER = "researcher"
    CRITIC = "critic"
    SYNTHESIZER = "synthesizer"
    VERIFIER = "verifier"
    NEURAL_INFERENCE = "neural_inference"


class CognitiveStepState(str, Enum):
    """Deterministic lifecycle state of a single cognitive DAG step."""
    PENDING = "pending"
    DISPATCHED = "dispatched"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class CognitiveEpisodeState(str, Enum):
    """Lifecycle states of a full cognitive episode."""
    UNINITIALIZED = "uninitialized"
    PLANNING = "planning"
    EXECUTING = "executing"
    SYNTHESIZING = "synthesizing"
    FINALIZING = "finalizing"
    COMMITTED = "committed"
    FAILED = "failed"
    REJECTED = "rejected"


# Fail-closed transition table
VALID_EPISODE_TRANSITIONS: Dict[CognitiveEpisodeState, Set[CognitiveEpisodeState]] = {
    CognitiveEpisodeState.UNINITIALIZED: {
        CognitiveEpisodeState.PLANNING,
        CognitiveEpisodeState.FAILED,
    },
    CognitiveEpisodeState.PLANNING: {
        CognitiveEpisodeState.EXECUTING,
        CognitiveEpisodeState.FAILED,
        CognitiveEpisodeState.REJECTED,
    },
    CognitiveEpisodeState.EXECUTING: {
        CognitiveEpisodeState.SYNTHESIZING,
        CognitiveEpisodeState.FAILED,
        CognitiveEpisodeState.REJECTED,
    },
    CognitiveEpisodeState.SYNTHESIZING: {
        CognitiveEpisodeState.FINALIZING,
        CognitiveEpisodeState.EXECUTING,   # Revision loop
        CognitiveEpisodeState.FAILED,
    },
    CognitiveEpisodeState.FINALIZING: {
        CognitiveEpisodeState.COMMITTED,
        CognitiveEpisodeState.FAILED,
    },
    CognitiveEpisodeState.COMMITTED: set(),   # Terminal
    CognitiveEpisodeState.FAILED:    set(),   # Terminal
    CognitiveEpisodeState.REJECTED:  set(),   # Terminal
}


# ─────────────────────────────────────────────────────────────────────────────
# CognitiveStep
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CognitiveStep:
    """
    A typed node in the cognitive reasoning DAG.

    Each step maps to a single federated capability invocation.
    Dependencies must be COMPLETED before this step becomes dispatchable.
    """
    step_id: str
    role: CognitiveRole
    capability_id: str
    input_payload: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    state: CognitiveStepState = CognitiveStepState.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    dispatched_to_node: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

    def mark_dispatched(self, node_id: str = "local") -> None:
        self.state = CognitiveStepState.DISPATCHED
        self.dispatched_to_node = node_id

    def mark_completed(self, result: Dict[str, Any]) -> None:
        self.state = CognitiveStepState.COMPLETED
        self.result = result
        self.completed_at = time.time()

    def mark_failed(self, error: str) -> None:
        self.state = CognitiveStepState.FAILED
        self.error = error
        self.completed_at = time.time()

    def is_terminal(self) -> bool:
        return self.state in (
            CognitiveStepState.COMPLETED,
            CognitiveStepState.FAILED,
            CognitiveStepState.SKIPPED,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "role": self.role.value,
            "capability_id": self.capability_id,
            "input_payload": self.input_payload,
            "dependencies": self.dependencies,
            "state": self.state.value,
            "result": self.result,
            "error": self.error,
            "dispatched_to_node": self.dispatched_to_node,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


# ─────────────────────────────────────────────────────────────────────────────
# CognitiveTaskGraph
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CognitiveTaskGraph:
    """
    Directed Acyclic Graph (DAG) of CognitiveSteps representing a structured
    collaborative reasoning plan.

    Invariants:
    - All step_ids referenced in dependencies must exist in the graph.
    - No cycles are permitted (DFS-verified on validate()).
    - Steps are dispatched only when all dependencies are COMPLETED.
    """
    graph_id: str
    episode_id: str
    steps: Dict[str, CognitiveStep] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def add_step(self, step: CognitiveStep) -> None:
        """Add a step; raises ValueError on duplicate step_id."""
        if step.step_id in self.steps:
            raise ValueError(
                f"Duplicate step_id '{step.step_id}' in graph '{self.graph_id}'"
            )
        self.steps[step.step_id] = step

    def validate(self) -> None:
        """
        Validate graph integrity:
        1. All dependency step_ids must reference existing steps.
        2. No directed cycles (DFS).
        """
        for step_id, step in self.steps.items():
            for dep in step.dependencies:
                if dep not in self.steps:
                    raise CognitiveStepDependencyError(
                        f"Step '{step_id}' depends on unknown step '{dep}'"
                    )
        self._detect_cycles()

    def _detect_cycles(self) -> None:
        """DFS-based cycle detection. Raises CognitiveGraphCycleError on first cycle."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[str, int] = {sid: WHITE for sid in self.steps}

        def dfs(node: str) -> None:
            color[node] = GRAY
            for dep in self.steps[node].dependencies:
                if color[dep] == GRAY:
                    raise CognitiveGraphCycleError(
                        f"Cycle detected involving steps '{node}' and '{dep}'"
                    )
                if color[dep] == WHITE:
                    dfs(dep)
            color[node] = BLACK

        for step_id in self.steps:
            if color[step_id] == WHITE:
                dfs(step_id)

    def get_ready_steps(self) -> List[CognitiveStep]:
        """
        Return all PENDING steps whose every dependency is in COMPLETED state.
        """
        ready = []
        for step in self.steps.values():
            if step.state != CognitiveStepState.PENDING:
                continue
            deps_done = all(
                self.steps[dep].state == CognitiveStepState.COMPLETED
                for dep in step.dependencies
                if dep in self.steps
            )
            if deps_done:
                ready.append(step)
        return ready

    def mark_step_completed(self, step_id: str, result: Dict[str, Any]) -> None:
        if step_id not in self.steps:
            raise KeyError(f"Step '{step_id}' not in graph '{self.graph_id}'")
        self.steps[step_id].mark_completed(result)

    def mark_step_failed(self, step_id: str, error: str) -> None:
        if step_id not in self.steps:
            raise KeyError(f"Step '{step_id}' not in graph '{self.graph_id}'")
        self.steps[step_id].mark_failed(error)

    def is_complete(self) -> bool:
        """True when every step has reached a terminal state."""
        return all(step.is_terminal() for step in self.steps.values())

    def has_failures(self) -> bool:
        """True if any step is in FAILED state."""
        return any(
            step.state == CognitiveStepState.FAILED
            for step in self.steps.values()
        )

    def completed_results(self) -> Dict[str, Dict[str, Any]]:
        """Return {step_id: result} for all COMPLETED steps."""
        return {
            sid: step.result
            for sid, step in self.steps.items()
            if step.state == CognitiveStepState.COMPLETED and step.result is not None
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "episode_id": self.episode_id,
            "steps": {sid: s.to_dict() for sid, s in self.steps.items()},
            "created_at": self.created_at,
        }


# ─────────────────────────────────────────────────────────────────────────────
# CognitiveContextEnvelope
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CognitiveContextEnvelope:
    """
    Bounded, sanitized inter-node cognitive context carrier.

    Carries evidence, hypotheses, and context across federated workers
    while enforcing:
    1. Token ceiling: total word-count estimate <= MAX_ENVELOPE_CONTEXT_TOKENS (448).
    2. Secret scan: prohibited keywords (private_key, etc.) cause immediate rejection.
    3. Tenant isolation: tenant_id mismatch raises CognitiveContextTenantViolationError.
    4. Bounded collection sizes via FIFO eviction.
    """
    envelope_id: str
    episode_id: str
    tenant_id: str
    session_id: str
    token_count: int = 0
    context_items: List[str] = field(default_factory=list)
    hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    evidence_items: List[str] = field(default_factory=list)
    provenance_chain: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    MAX_CONTEXT_ITEMS: int = 15
    MAX_HYPOTHESES: int = 5
    MAX_EVIDENCE_ITEMS: int = 10
    MAX_TOKEN_BUDGET: int = MAX_ENVELOPE_CONTEXT_TOKENS  # 448

    def validate(self) -> None:
        """Enforce all envelope invariants: token ceiling + secret scan."""
        if self.token_count > self.MAX_TOKEN_BUDGET:
            raise CognitiveContextOverflowError(
                f"Envelope '{self.envelope_id}' token_count {self.token_count} "
                f"exceeds ceiling {self.MAX_TOKEN_BUDGET}"
            )
        self._scan_for_secrets()

    def validate_for_tenant(self, expected_tenant_id: str) -> None:
        """Enforce cross-tenant isolation. Raises on mismatch."""
        if self.tenant_id != expected_tenant_id:
            raise CognitiveContextTenantViolationError(
                f"Envelope tenant '{self.tenant_id}' != expected '{expected_tenant_id}'"
            )

    def _scan_for_secrets(self) -> None:
        """Scan all text content for prohibited credential keywords (CT-07)."""
        all_text = (
            " ".join(self.context_items)
            + " "
            + " ".join(self.evidence_items)
            + " "
            + " ".join(str(h) for h in self.hypotheses)
        )
        for keyword in PROHIBITED_CONTEXT_KEYWORDS:
            if keyword in all_text:
                raise SecretLeakageInContextError(
                    f"Prohibited keyword '{keyword}' in envelope '{self.envelope_id}'"
                )

    def add_context_item(self, text: str) -> None:
        """Add context item with bounded FIFO eviction at MAX_CONTEXT_ITEMS."""
        if len(self.context_items) >= self.MAX_CONTEXT_ITEMS:
            self.context_items.pop(0)
        self.context_items.append(text)
        self._update_token_count()

    def add_hypothesis(self, hypothesis: Dict[str, Any]) -> None:
        """Add hypothesis with bounded FIFO eviction at MAX_HYPOTHESES."""
        if len(self.hypotheses) >= self.MAX_HYPOTHESES:
            self.hypotheses.pop(0)
        self.hypotheses.append(hypothesis)

    def add_evidence(self, evidence: str) -> None:
        """Add evidence item with bounded FIFO eviction at MAX_EVIDENCE_ITEMS."""
        if len(self.evidence_items) >= self.MAX_EVIDENCE_ITEMS:
            self.evidence_items.pop(0)
        self.evidence_items.append(evidence)
        self._update_token_count()

    def add_provenance(self, node_id: str) -> None:
        """Record a contributing node_id in the provenance chain (deduped)."""
        if node_id not in self.provenance_chain:
            self.provenance_chain.append(node_id)

    def _update_token_count(self) -> None:
        """
        Approximate token count as whitespace-split words across all text fields.
        CPU-safe: no tokenizer required.
        """
        combined = " ".join(self.context_items) + " " + " ".join(self.evidence_items)
        self.token_count = min(len(combined.split()), self.MAX_TOKEN_BUDGET)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "envelope_id": self.envelope_id,
            "episode_id": self.episode_id,
            "tenant_id": self.tenant_id,
            "session_id": self.session_id,
            "token_count": self.token_count,
            "context_items": list(self.context_items),
            "hypotheses": list(self.hypotheses),
            "evidence_items": list(self.evidence_items),
            "provenance_chain": list(self.provenance_chain),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CognitiveContextEnvelope":
        return cls(
            envelope_id=data["envelope_id"],
            episode_id=data["episode_id"],
            tenant_id=data["tenant_id"],
            session_id=data["session_id"],
            token_count=data.get("token_count", 0),
            context_items=list(data.get("context_items", [])),
            hypotheses=list(data.get("hypotheses", [])),
            evidence_items=list(data.get("evidence_items", [])),
            provenance_chain=list(data.get("provenance_chain", [])),
            created_at=data.get("created_at", time.time()),
        )


# ─────────────────────────────────────────────────────────────────────────────
# ConflictRecord & SynthesisResult
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConflictRecord:
    """
    Preserved record of minority-opinion or contradicted evidence.

    Implements the Anti-Majority Evidence Preservation principle (CT-12).
    Disagreements are documented, never silently discarded.
    """
    record_id: str
    episode_id: str
    step_id: str
    dissenting_node_id: str
    majority_conclusion: str
    minority_conclusion: str
    contradiction_description: str
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "episode_id": self.episode_id,
            "step_id": self.step_id,
            "dissenting_node_id": self.dissenting_node_id,
            "majority_conclusion": self.majority_conclusion,
            "minority_conclusion": self.minority_conclusion,
            "contradiction_description": self.contradiction_description,
            "created_at": self.created_at,
        }


@dataclass
class SynthesisResult:
    """Aggregated output from CognitiveSynthesisEngine across distributed workers."""
    synthesis_id: str
    episode_id: str
    synthesized_conclusion: str
    evidence_count: int
    hypothesis_count: int
    conflict_records: List[ConflictRecord] = field(default_factory=list)
    participating_nodes: List[str] = field(default_factory=list)
    has_minority_evidence: bool = False
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "synthesis_id": self.synthesis_id,
            "episode_id": self.episode_id,
            "synthesized_conclusion": self.synthesized_conclusion,
            "evidence_count": self.evidence_count,
            "hypothesis_count": self.hypothesis_count,
            "conflict_records": [cr.to_dict() for cr in self.conflict_records],
            "participating_nodes": list(self.participating_nodes),
            "has_minority_evidence": self.has_minority_evidence,
            "created_at": self.created_at,
        }


# ─────────────────────────────────────────────────────────────────────────────
# CognitiveEpisode
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CognitiveEpisode:
    """
    Multi-step, multi-node collaborative cognitive reasoning session.

    State transitions are strictly enforced through VALID_EPISODE_TRANSITIONS.
    Terminal states (COMMITTED, FAILED, REJECTED) are absorbing.
    """
    episode_id: str
    tenant_id: str
    session_id: str
    objective: str
    task_graph: Optional[CognitiveTaskGraph] = None
    context: Optional[CognitiveContextEnvelope] = None
    synthesis_result: Optional[SynthesisResult] = None
    state: CognitiveEpisodeState = CognitiveEpisodeState.UNINITIALIZED
    error: Optional[str] = None
    consensus_proposal_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def transition_to(self, new_state: CognitiveEpisodeState) -> None:
        """Enforce fail-closed state machine. Raises CognitiveEpisodeStateError on violation."""
        allowed = VALID_EPISODE_TRANSITIONS.get(self.state, set())
        if new_state not in allowed:
            raise CognitiveEpisodeStateError(
                f"Invalid episode transition: '{self.state.value}' -> '{new_state.value}' "
                f"(episode '{self.episode_id}')"
            )
        self.state = new_state
        self.updated_at = time.time()

    def is_terminal(self) -> bool:
        return self.state in (
            CognitiveEpisodeState.COMMITTED,
            CognitiveEpisodeState.FAILED,
            CognitiveEpisodeState.REJECTED,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "tenant_id": self.tenant_id,
            "session_id": self.session_id,
            "objective": self.objective,
            "state": self.state.value,
            "error": self.error,
            "consensus_proposal_id": self.consensus_proposal_id,
            "task_graph": self.task_graph.to_dict() if self.task_graph else None,
            "context": self.context.to_dict() if self.context else None,
            "synthesis_result": (
                self.synthesis_result.to_dict() if self.synthesis_result else None
            ),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
