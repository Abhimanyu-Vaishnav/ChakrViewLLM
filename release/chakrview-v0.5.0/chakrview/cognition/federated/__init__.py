"""
Multi-Agent Federated Cognition & Cooperative Intelligence Subsystem (Step 26).

Provides:
- Agent Identity, Roles, Capabilities, Status & Contracts
- Typed Message Protocol & Tamper-Evident Hashing
- Tenant-Scoped Agent Registry with Hard Capacity Bounds
- Hardware-Adaptive Federated Execution Policies & Ceilings
- Deterministic Task Decomposition
- Specialized Logical Cognitive Agents (Analyst, Researcher, Critic, Planner, Synthesizer, Verifier)
- Epistemic Evidence Aggregation & Provenance Tracking
- Conflict Resolution & Minority Evidence Preservation
- Consensus Synthesis & Decision Mapping
- Fault-Isolated Federated Cognition Engine
- Safe Public Federated Audit Traces
"""

from chakrview.cognition.federated.models import (
    AgentRole,
    AgentStatus,
    AgentCapability,
    AgentIdentity,
    AgentContract,
    MessageType,
    MessagePriority,
    AgentMessage,
    MessageEnvelope,
    AgentTask,
    ConflictState,
    FederatedConflictRecord,
    FederatedSynthesisCandidate,
    SafePublicFederatedTrace,
)
from chakrview.cognition.federated.protocol import (
    FederatedProtocolValidator,
    MessageValidationError,
    MessageTamperingError,
    TenantRoutingError,
)
from chakrview.cognition.federated.registry import (
    AgentRegistry,
    DuplicateAgentIdentityError,
    RegistryCapacityExceededError,
    AgentNotFoundError,
)
from chakrview.cognition.federated.policy import (
    FederatedExecutionPolicy,
    HARD_CEILING_FEDERATED_AGENTS,
    HARD_CEILING_FEDERATED_ROUNDS,
    HARD_CEILING_FEDERATED_MESSAGES,
    HARD_CEILING_DELEGATION_DEPTH,
)
from chakrview.cognition.federated.decomposition import (
    FederatedTaskDecomposer,
)
from chakrview.cognition.federated.agents import (
    FederatedAgent,
    AnalystAgent,
    ResearcherAgent,
    CriticAgent,
    PlannerAgent,
    SynthesizerAgent,
    VerifierAgent,
    TenantIsolationError,
    AgentExecutionError,
)
from chakrview.cognition.federated.evidence import (
    FederatedEvidenceAggregator,
    EvidenceCategory,
)
from chakrview.cognition.federated.conflict import (
    FederatedConflictResolver,
)
from chakrview.cognition.federated.synthesis import (
    FederatedSynthesizer,
)
from chakrview.cognition.federated.engine import (
    FederatedCognitionEngine,
)

__all__ = [
    "AgentRole",
    "AgentStatus",
    "AgentCapability",
    "AgentIdentity",
    "AgentContract",
    "MessageType",
    "MessagePriority",
    "AgentMessage",
    "MessageEnvelope",
    "AgentTask",
    "ConflictState",
    "FederatedConflictRecord",
    "FederatedSynthesisCandidate",
    "SafePublicFederatedTrace",
    "FederatedProtocolValidator",
    "MessageValidationError",
    "MessageTamperingError",
    "TenantRoutingError",
    "AgentRegistry",
    "DuplicateAgentIdentityError",
    "RegistryCapacityExceededError",
    "AgentNotFoundError",
    "FederatedExecutionPolicy",
    "HARD_CEILING_FEDERATED_AGENTS",
    "HARD_CEILING_FEDERATED_ROUNDS",
    "HARD_CEILING_FEDERATED_MESSAGES",
    "HARD_CEILING_DELEGATION_DEPTH",
    "FederatedTaskDecomposer",
    "FederatedAgent",
    "AnalystAgent",
    "ResearcherAgent",
    "CriticAgent",
    "PlannerAgent",
    "SynthesizerAgent",
    "VerifierAgent",
    "TenantIsolationError",
    "AgentExecutionError",
    "FederatedEvidenceAggregator",
    "EvidenceCategory",
    "FederatedConflictResolver",
    "FederatedSynthesizer",
    "FederatedCognitionEngine",
]
