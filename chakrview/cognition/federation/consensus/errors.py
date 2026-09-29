"""
Consensus and Byzantine Fault Tolerance Error Hierarchy (Step 40).
"""

from typing import Optional


class ConsensusError(Exception):
    """Base exception for all federated consensus and state agreement errors."""
    def __init__(self, message: str, node_id: Optional[str] = None, height: Optional[int] = None) -> None:
        super().__init__(message)
        self.message = message
        self.node_id = node_id
        self.height = height

    def __str__(self) -> str:
        ctx = []
        if self.node_id:
            ctx.append(f"node={self.node_id}")
        if self.height is not None:
            ctx.append(f"height={self.height}")
        ctx_str = f" [{', '.join(ctx)}]" if ctx else ""
        return f"{self.message}{ctx_str}"


class InvalidProposalError(ConsensusError):
    """Raised when a consensus proposal is malformed, oversized, or has an invalid digest."""
    pass


class StaleConsensusMessageError(ConsensusError):
    """Raised when a consensus message belongs to a past epoch, round, or height."""
    pass


class DoubleVotingError(ConsensusError):
    """Raised when a validator casts contradictory votes within the same round (Byzantine fault)."""
    pass


class EquivocationError(ConsensusError):
    """Raised when a proposer emits multiple conflicting proposals for the same slot (Byzantine fault)."""
    pass


class UnauthorizedValidatorError(ConsensusError):
    """Raised when a node not in the active consensus validator set attempts to propose or vote."""
    pass


class InvalidQuorumCertificateError(ConsensusError):
    """Raised when a Quorum Certificate fails cryptographic multi-signature or threshold verification."""
    pass


class QuorumNotReachedError(ConsensusError):
    """Raised when a consensus round concludes without reaching supermajority or majority quorum."""
    pass


class ViewChangeTimeoutError(ConsensusError):
    """Raised when a consensus round times out waiting for a proposal or valid votes."""
    pass


class TenantConsensusIsolationError(ConsensusError):
    """Raised when a consensus proposal crosses unauthorized tenant boundaries."""
    pass


class SovereignPolicyViolationError(ConsensusError):
    """
    Raised when a quorum-agreed consensus transition violates local sovereign security policy.
    Axiom: LOCAL_POLICY > CONSENSUS_DECISION.
    """
    pass


class StateDivergenceError(ConsensusError):
    """Raised when parent hash linking fails or local state machine log diverges."""
    pass
