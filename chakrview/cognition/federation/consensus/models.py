"""
Strongly Typed Consensus, Quorum Certificate, and State Machine Replication Models (Step 40).
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import json
import time
from typing import Dict, List, Optional, Any, Set, Tuple


GENESIS_PARENT_HASH = "0" * 64
MAX_CONSENSUS_PAYLOAD_BYTES = 1048576  # 1 MB


class ConsensusPhase(str, Enum):
    """Lifecycle phases for a consensus slot/round."""
    IDLE = "IDLE"
    PROPOSE = "PROPOSE"
    PREVOTE = "PREVOTE"
    PRECOMMIT = "PRECOMMIT"
    COMMITTED = "COMMITTED"
    VIEW_CHANGE = "VIEW_CHANGE"
    FAILED = "FAILED"


class ConsensusTransitionType(str, Enum):
    """Taxonomy of cluster-wide state mutations governed by federated consensus."""
    MEMBERSHIP_TOPOLOGY_UPDATE = "MEMBERSHIP_TOPOLOGY_UPDATE"
    PEER_REVOCATION_AGREEMENT = "PEER_REVOCATION_AGREEMENT"
    TASK_COMMIT_FINALIZATION = "TASK_COMMIT_FINALIZATION"
    CHECKPOINT_AUTHORIZATION = "CHECKPOINT_AUTHORIZATION"
    EPOCH_ADVANCEMENT = "EPOCH_ADVANCEMENT"
    CUSTOM_STATE_TRANSITION = "CUSTOM_STATE_TRANSITION"


class VoteType(str, Enum):
    """Taxonomy of votes cast by consensus validators."""
    PREVOTE = "PREVOTE"
    PRECOMMIT = "PRECOMMIT"
    VIEW_CHANGE = "VIEW_CHANGE"


class FaultToleranceMode(str, Enum):
    """Operating mode determining quorum thresholds."""
    BFT = "BFT"  # Byzantine Fault Tolerance: Q >= floor(2N/3) + 1 (resilient to f < N/3 Byzantine nodes)
    CFT = "CFT"  # Crash Fault Tolerance: Q >= floor(N/2) + 1 (resilient to f < N/2 crash failures)


@dataclass
class ConsensusProposal:
    """
    Candidate state machine mutation proposed by the designated round leader.
    Binds monotonic slot identifiers, canonical payload, parent hash, and cryptographic digest.
    """
    proposal_id: str
    epoch: int
    round: int
    height: int
    proposer_id: str
    transition_type: ConsensusTransitionType
    payload: Dict[str, Any]
    parent_hash: str = GENESIS_PARENT_HASH
    tenant_id: str = "default"
    proposal_digest: str = ""
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.proposal_digest:
            self.proposal_digest = self.compute_digest()

    def compute_digest(self) -> str:
        """Deterministic SHA-256 digest over canonical JSON representation."""
        payload_repr = json.dumps(self.payload, sort_keys=True, separators=(',', ':'))
        canonical_str = (
            f"{self.epoch}:{self.round}:{self.height}:{self.proposer_id}:"
            f"{self.transition_type.value if hasattr(self.transition_type, 'value') else self.transition_type}:"
            f"{self.parent_hash}:{self.tenant_id}:{payload_repr}"
        )
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify that stored digest matches canonical digest."""
        return self.compute_digest() == self.proposal_digest

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "epoch": self.epoch,
            "round": self.round,
            "height": self.height,
            "proposer_id": self.proposer_id,
            "transition_type": self.transition_type.value if hasattr(self.transition_type, "value") else str(self.transition_type),
            "payload": self.payload,
            "parent_hash": self.parent_hash,
            "tenant_id": self.tenant_id,
            "proposal_digest": self.proposal_digest,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConsensusProposal":
        d = dict(data)
        d["transition_type"] = ConsensusTransitionType(d["transition_type"])
        return cls(**d)


@dataclass
class ConsensusVote:
    """
    Authenticated vote cast by a consensus validator for a specific proposal digest or NIL.
    """
    vote_id: str
    epoch: int
    round: int
    height: int
    voter_id: str
    vote_type: VoteType
    proposal_digest: str  # Digest of the proposal voted for, or "NIL"
    signature: str = ""   # Cryptographic Ed25519 or HMAC signature
    timestamp: float = field(default_factory=time.time)

    def compute_digest(self) -> str:
        canonical_str = f"{self.epoch}:{self.round}:{self.height}:{self.voter_id}:{self.vote_type.value}:{self.proposal_digest}"
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vote_id": self.vote_id,
            "epoch": self.epoch,
            "round": self.round,
            "height": self.height,
            "voter_id": self.voter_id,
            "vote_type": self.vote_type.value,
            "proposal_digest": self.proposal_digest,
            "signature": self.signature,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConsensusVote":
        d = dict(data)
        d["vote_type"] = VoteType(d["vote_type"])
        return cls(**d)


@dataclass
class QuorumCertificate:
    """
    Cryptographic proof that a supermajority (or majority) of authorized validators
    signed the same proposal digest for the specified slot.
    """
    qc_id: str
    epoch: int
    round: int
    height: int
    proposal_digest: str
    vote_type: VoteType
    voter_ids: List[str]
    signatures: Dict[str, str] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def compute_digest(self) -> str:
        sorted_voters = sorted(self.voter_ids)
        canonical_str = f"{self.epoch}:{self.round}:{self.height}:{self.proposal_digest}:{self.vote_type.value}:{','.join(sorted_voters)}"
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "qc_id": self.qc_id,
            "epoch": self.epoch,
            "round": self.round,
            "height": self.height,
            "proposal_digest": self.proposal_digest,
            "vote_type": self.vote_type.value,
            "voter_ids": self.voter_ids,
            "signatures": self.signatures,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuorumCertificate":
        d = dict(data)
        d["vote_type"] = VoteType(d["vote_type"])
        return cls(**d)


@dataclass
class EquivocationEvidence:
    """
    Cryptographic proof of a Byzantine fault: a proposer emitted multiple distinct
    proposals for the exact same slot (epoch, round, height).
    """
    evidence_id: str
    epoch: int
    round: int
    height: int
    offender_id: str
    proposal_digest_1: str
    proposal_digest_2: str
    evidence_digest: str = ""
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.evidence_digest:
            canonical_str = f"{self.epoch}:{self.round}:{self.height}:{self.offender_id}:{self.proposal_digest_1}:{self.proposal_digest_2}"
            self.evidence_digest = hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConsensusLogEntry:
    """
    Durably committed entry in the Replicated State Machine log.
    Guarantees total order and strict replayability.
    """
    height: int
    epoch: int
    round: int
    proposal: ConsensusProposal
    quorum_certificate: QuorumCertificate
    state_root_hash: str
    committed_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "height": self.height,
            "epoch": self.epoch,
            "round": self.round,
            "proposal": self.proposal.to_dict(),
            "quorum_certificate": self.quorum_certificate.to_dict(),
            "state_root_hash": self.state_root_hash,
            "committed_at": self.committed_at,
        }


@dataclass
class ConsensusConfig:
    """Operational parameters and safety thresholds for consensus execution."""
    min_quorum_ratio: float = 2.0 / 3.0
    proposal_timeout_sec: float = 3.0
    vote_timeout_sec: float = 2.0
    max_payload_size_bytes: int = MAX_CONSENSUS_PAYLOAD_BYTES
    fault_tolerance_mode: FaultToleranceMode = FaultToleranceMode.BFT
    tenant_id: str = "default"

    def compute_quorum_threshold(self, validator_count: int) -> int:
        """
        Calculate required vote count for quorum.
        BFT mode: Q = floor(2N/3) + 1
        CFT mode: Q = floor(N/2) + 1
        """
        if validator_count <= 0:
            return 1
        if self.fault_tolerance_mode == FaultToleranceMode.BFT:
            return (2 * validator_count) // 3 + 1
        return (validator_count // 2) + 1
