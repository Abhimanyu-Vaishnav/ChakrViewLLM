"""
Federated Consensus, Byzantine Fault Tolerance & Multi-Node State Agreement Subsystem (Step 40).
"""

from chakrview.cognition.federation.consensus.models import (
    ConsensusPhase,
    ConsensusTransitionType,
    VoteType,
    FaultToleranceMode,
    ConsensusProposal,
    ConsensusVote,
    QuorumCertificate,
    EquivocationEvidence,
    ConsensusLogEntry,
    ConsensusConfig,
    GENESIS_PARENT_HASH,
    MAX_CONSENSUS_PAYLOAD_BYTES,
)
from chakrview.cognition.federation.consensus.errors import (
    ConsensusError,
    InvalidProposalError,
    StaleConsensusMessageError,
    DoubleVotingError,
    EquivocationError,
    UnauthorizedValidatorError,
    InvalidQuorumCertificateError,
    QuorumNotReachedError,
    ViewChangeTimeoutError,
    TenantConsensusIsolationError,
    SovereignPolicyViolationError,
    StateDivergenceError,
)
from chakrview.cognition.federation.consensus.validator import ConsensusValidator
from chakrview.cognition.federation.consensus.state_machine import ReplicatedStateMachine
from chakrview.cognition.federation.consensus.engine import FederatedConsensusEngine

__all__ = [
    # Models
    "ConsensusPhase",
    "ConsensusTransitionType",
    "VoteType",
    "FaultToleranceMode",
    "ConsensusProposal",
    "ConsensusVote",
    "QuorumCertificate",
    "EquivocationEvidence",
    "ConsensusLogEntry",
    "ConsensusConfig",
    "GENESIS_PARENT_HASH",
    "MAX_CONSENSUS_PAYLOAD_BYTES",
    # Errors
    "ConsensusError",
    "InvalidProposalError",
    "StaleConsensusMessageError",
    "DoubleVotingError",
    "EquivocationError",
    "UnauthorizedValidatorError",
    "InvalidQuorumCertificateError",
    "QuorumNotReachedError",
    "ViewChangeTimeoutError",
    "TenantConsensusIsolationError",
    "SovereignPolicyViolationError",
    "StateDivergenceError",
    # Core Components
    "ConsensusValidator",
    "ReplicatedStateMachine",
    "FederatedConsensusEngine",
]
