"""
Federated Consensus Engine Coordinating BFT Rounds, Quorum Certification, and State Replication (Step 40).
"""

import hashlib
import json
import logging
import threading
import time
from typing import Dict, List, Optional, Set, Any, Tuple

from chakrview.cognition.federation.consensus.models import (
    ConsensusPhase,
    ConsensusTransitionType,
    VoteType,
    FaultToleranceMode,
    ConsensusProposal,
    ConsensusVote,
    QuorumCertificate,
    ConsensusConfig,
    GENESIS_PARENT_HASH,
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
)
from chakrview.cognition.federation.consensus.validator import ConsensusValidator
from chakrview.cognition.federation.consensus.state_machine import ReplicatedStateMachine

logger = logging.getLogger("chakrview.consensus.engine")


class FederatedConsensusEngine:
    """
    Coordinates distributed multi-node consensus.
    Implements a 3-phase BFT consensus protocol (PROPOSE -> PREVOTE -> PRECOMMIT -> COMMIT)
    with Quorum Certificates, deterministic round-robin leaders, and view-change timeouts.
    """

    def __init__(
        self,
        local_node_id: str,
        validators: List[str],
        config: Optional[ConsensusConfig] = None,
        capability_gate: Optional[Any] = None,
        journal: Optional[Any] = None,
        audit_logger: Optional[Any] = None,
    ) -> None:
        self.local_node_id = local_node_id
        self.validators: List[str] = sorted(list(validators))
        self.config = config or ConsensusConfig()
        self.journal = journal
        self.audit_logger = audit_logger

        self._lock = threading.RLock()
        self.validator = ConsensusValidator(
            config=self.config,
            authorized_validators=set(self.validators),
        )
        self.state_machine = ReplicatedStateMachine(
            local_node_id=local_node_id,
            capability_gate=capability_gate,
            journal=journal,
        )

        # Slot Tracking
        self._epoch = 1
        self._current_round = 0
        self._current_height = 0
        self._phase = ConsensusPhase.IDLE

        # Round State: (epoch, round, height) -> ConsensusProposal
        self._proposals: Dict[Tuple[int, int, int], ConsensusProposal] = {}
        # Prevotes: (epoch, round, height, proposal_digest) -> Dict[voter_id, ConsensusVote]
        self._prevotes: Dict[Tuple[int, int, int, str], Dict[str, ConsensusVote]] = {}
        # Precommits: (epoch, round, height, proposal_digest) -> Dict[voter_id, ConsensusVote]
        self._precommits: Dict[Tuple[int, int, int, str], Dict[str, ConsensusVote]] = {}
        # View change votes: (epoch, round, height) -> Dict[voter_id, ConsensusVote]
        self._view_change_votes: Dict[Tuple[int, int, int], Dict[str, ConsensusVote]] = {}

        # Quorum Certificates: (epoch, round, height, vote_type) -> QuorumCertificate
        self._quorum_certificates: Dict[Tuple[int, int, int, VoteType], QuorumCertificate] = {}

    @property
    def current_height(self) -> int:
        with self._lock:
            return self.state_machine.current_height

    @property
    def current_round(self) -> int:
        with self._lock:
            return self._current_round

    @property
    def epoch(self) -> int:
        with self._lock:
            return self._epoch

    @property
    def phase(self) -> ConsensusPhase:
        with self._lock:
            return self._phase

    def get_leader_for_round(self, height: int, round_num: int) -> str:
        """
        Deterministic leader schedule: round-robin over sorted validator set.
        """
        with self._lock:
            if not self.validators:
                return self.local_node_id
            idx = (height + round_num) % len(self.validators)
            return self.validators[idx]

    def is_leader_for_current_round(self) -> bool:
        with self._lock:
            expected_height = self.state_machine.current_height + 1
            return self.get_leader_for_round(expected_height, self._current_round) == self.local_node_id

    def update_validator_set(self, new_validators: List[str]) -> None:
        """Update active consensus validator list."""
        with self._lock:
            self.validators = sorted(list(new_validators))
            self.validator.set_authorized_validators(set(self.validators))
            logger.info("Updated consensus validator set to %d nodes", len(self.validators))

    def create_proposal(
        self,
        transition_type: ConsensusTransitionType,
        payload: Dict[str, Any],
        tenant_id: Optional[str] = None,
    ) -> ConsensusProposal:
        """
        Create and record a new proposal for the current slot.
        Only the designated leader may create proposals.
        """
        with self._lock:
            expected_height = self.state_machine.current_height + 1
            leader = self.get_leader_for_round(expected_height, self._current_round)
            if leader != self.local_node_id:
                raise UnauthorizedValidatorError(
                    f"Local node {self.local_node_id} is not the designated leader ({leader}) for round {self._current_round}",
                    node_id=self.local_node_id,
                    height=expected_height,
                )

            proposal_id = f"prop_{self._epoch}_{self._current_round}_{expected_height}_{int(time.time() * 1000)}"
            proposal = ConsensusProposal(
                proposal_id=proposal_id,
                epoch=self._epoch,
                round=self._current_round,
                height=expected_height,
                proposer_id=self.local_node_id,
                transition_type=transition_type,
                payload=payload,
                parent_hash=self.state_machine.state_root_hash,
                tenant_id=tenant_id or self.config.tenant_id,
            )

            # Validate against self
            self.validator.validate_proposal(
                proposal,
                current_epoch=self._epoch,
                current_height=self.state_machine.current_height,
            )

            slot_key = (self._epoch, self._current_round, expected_height)
            self._proposals[slot_key] = proposal
            self._phase = ConsensusPhase.PREVOTE

            self._log_audit("CONSENSUS_PROPOSAL_CREATED", {"proposal_id": proposal.proposal_id, "height": expected_height})
            self._wal_log("CONSENSUS_PROPOSAL_BROADCAST", {"proposal_id": proposal.proposal_id, "height": expected_height})
            return proposal

    def receive_proposal(self, proposal: ConsensusProposal) -> bool:
        """
        Receive and validate a proposal broadcast from a peer leader.
        """
        with self._lock:
            # Verify sender matches designated leader
            expected_leader = self.get_leader_for_round(proposal.height, proposal.round)
            if proposal.proposer_id != expected_leader:
                raise UnauthorizedValidatorError(
                    f"Proposal proposer {proposal.proposer_id} does not match designated leader {expected_leader}",
                    node_id=proposal.proposer_id,
                    height=proposal.height,
                )

            # Validate proposal syntax, size, digest, and check for equivocation
            self.validator.validate_proposal(
                proposal,
                current_epoch=self._epoch,
                current_height=self.state_machine.current_height,
            )

            slot_key = (proposal.epoch, proposal.round, proposal.height)
            self._proposals[slot_key] = proposal
            self._phase = ConsensusPhase.PREVOTE

            self._log_audit("CONSENSUS_PROPOSAL_VALIDATED", {"proposal_id": proposal.proposal_id, "height": proposal.height})
            return True

    def record_prevote(self, vote: ConsensusVote) -> Optional[QuorumCertificate]:
        """
        Record a PREVOTE from a validator.
        If a quorum is reached for a proposal digest, returns a PREVOTE QuorumCertificate.
        """
        with self._lock:
            if vote.vote_type != VoteType.PREVOTE:
                raise ConsensusError(f"Expected PREVOTE, received {vote.vote_type.value}")

            self.validator.validate_vote(
                vote,
                current_epoch=self._epoch,
                current_height=self.state_machine.current_height,
            )

            group_key = (vote.epoch, vote.round, vote.height, vote.proposal_digest)
            if group_key not in self._prevotes:
                self._prevotes[group_key] = {}
            self._prevotes[group_key][vote.voter_id] = vote

            self._log_audit("CONSENSUS_VOTE_RECEIVED", {"voter_id": vote.voter_id, "type": "PREVOTE", "digest": vote.proposal_digest})

            # Check if quorum reached
            collected = self._prevotes[group_key]
            required_quorum = self.config.compute_quorum_threshold(len(self.validators))
            if len(collected) >= required_quorum:
                qc_key = (vote.epoch, vote.round, vote.height, VoteType.PREVOTE)
                if qc_key not in self._quorum_certificates:
                    qc = QuorumCertificate(
                        qc_id=f"qc_prevote_{vote.epoch}_{vote.round}_{vote.height}_{int(time.time())}",
                        epoch=vote.epoch,
                        round=vote.round,
                        height=vote.height,
                        proposal_digest=vote.proposal_digest,
                        vote_type=VoteType.PREVOTE,
                        voter_ids=list(collected.keys()),
                        signatures={v_id: v.signature for v_id, v in collected.items()},
                    )
                    self._quorum_certificates[qc_key] = qc
                    self._phase = ConsensusPhase.PRECOMMIT
                    self._log_audit("CONSENSUS_QUORUM_CERTIFIED", {"qc_id": qc.qc_id, "type": "PREVOTE", "votes": len(collected)})
                    return qc
                return self._quorum_certificates[qc_key]

            return None

    def record_precommit(self, vote: ConsensusVote) -> Optional[QuorumCertificate]:
        """
        Record a PRECOMMIT from a validator.
        If a quorum is reached for a proposal digest, returns a PRECOMMIT QuorumCertificate.
        """
        with self._lock:
            if vote.vote_type != VoteType.PRECOMMIT:
                raise ConsensusError(f"Expected PRECOMMIT, received {vote.vote_type.value}")

            self.validator.validate_vote(
                vote,
                current_epoch=self._epoch,
                current_height=self.state_machine.current_height,
            )

            group_key = (vote.epoch, vote.round, vote.height, vote.proposal_digest)
            if group_key not in self._precommits:
                self._precommits[group_key] = {}
            self._precommits[group_key][vote.voter_id] = vote

            self._log_audit("CONSENSUS_VOTE_RECEIVED", {"voter_id": vote.voter_id, "type": "PRECOMMIT", "digest": vote.proposal_digest})

            # Check if quorum reached
            collected = self._precommits[group_key]
            required_quorum = self.config.compute_quorum_threshold(len(self.validators))
            if len(collected) >= required_quorum:
                qc_key = (vote.epoch, vote.round, vote.height, VoteType.PRECOMMIT)
                if qc_key not in self._quorum_certificates:
                    qc = QuorumCertificate(
                        qc_id=f"qc_precommit_{vote.epoch}_{vote.round}_{vote.height}_{int(time.time())}",
                        epoch=vote.epoch,
                        round=vote.round,
                        height=vote.height,
                        proposal_digest=vote.proposal_digest,
                        vote_type=VoteType.PRECOMMIT,
                        voter_ids=list(collected.keys()),
                        signatures={v_id: v.signature for v_id, v in collected.items()},
                    )
                    self._quorum_certificates[qc_key] = qc
                    self._phase = ConsensusPhase.COMMITTED
                    self._log_audit("CONSENSUS_QUORUM_CERTIFIED", {"qc_id": qc.qc_id, "type": "PRECOMMIT", "votes": len(collected)})
                    return qc
                return self._quorum_certificates[qc_key]

            return None

    def commit_block(
        self,
        proposal: ConsensusProposal,
        precommit_qc: QuorumCertificate,
    ) -> Any:
        """
        Commit and execute proposal against the Replicated State Machine.
        Validates QC threshold before committing.
        """
        with self._lock:
            # 1. Validate Quorum Certificate
            self.validator.validate_quorum_certificate(precommit_qc, validator_count=len(self.validators))

            if precommit_qc.proposal_digest != proposal.proposal_digest:
                raise InvalidQuorumCertificateError(
                    f"QC digest {precommit_qc.proposal_digest} != proposal digest {proposal.proposal_digest}",
                    height=proposal.height,
                )

            # 2. Apply to Replicated State Machine
            entry = self.state_machine.apply_committed_proposal(proposal, precommit_qc)

            # 3. Advance consensus slot state
            self._current_round = 0
            self._phase = ConsensusPhase.IDLE

            self._log_audit("CONSENSUS_COMMIT_EXECUTED", {"height": entry.height, "state_root": entry.state_root_hash})
            return entry

    def trigger_view_change(self, reason: str = "Timeout") -> int:
        """
        Advance round on leader failure or proposal timeout.
        """
        with self._lock:
            self._current_round += 1
            self._phase = ConsensusPhase.VIEW_CHANGE
            expected_height = self.state_machine.current_height + 1
            new_leader = self.get_leader_for_round(expected_height, self._current_round)

            logger.warning(
                "Triggered view change to round %d at height %d (reason: %s, new leader: %s)",
                self._current_round, expected_height, reason, new_leader,
            )
            self._log_audit("CONSENSUS_VIEW_TIMEOUT", {"round": self._current_round, "reason": reason, "leader": new_leader})
            self._wal_log("CONSENSUS_VIEW_CHANGED", {"round": self._current_round, "leader": new_leader})
            return self._current_round

    def _log_audit(self, event_type: str, details: Dict[str, Any]) -> None:
        if self.audit_logger:
            if hasattr(self.audit_logger, "log_event"):
                try:
                    self.audit_logger.log_event(event_type=event_type, details=details)
                except Exception:
                    pass
            elif hasattr(self.audit_logger, "log"):
                try:
                    from chakrview.cognition.peering.models import AuditEventType
                    ev = getattr(AuditEventType, "CONSENSUS_EVENT", AuditEventType.STATE_MUTATED)
                    self.audit_logger.log(
                        event_type=ev,
                        epoch=self._epoch,
                        peer_id=self.local_node_id,
                        details={"consensus_event": event_type, **details},
                    )
                except Exception:
                    pass

    def _wal_log(self, entry_type: str, payload: Dict[str, Any]) -> None:
        if self.journal and hasattr(self.journal, "append_entry"):
            try:
                self.journal.append_entry(entry_type, payload)
            except Exception:
                pass
