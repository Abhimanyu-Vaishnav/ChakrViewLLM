"""
Consensus Validator, Equivocation Detector, and Quorum Verification (Step 40).
"""

import hashlib
import json
import logging
import re
import threading
from typing import Dict, List, Optional, Set, Tuple, Any

from chakrview.cognition.federation.consensus.models import (
    ConsensusProposal,
    ConsensusVote,
    QuorumCertificate,
    EquivocationEvidence,
    ConsensusConfig,
    VoteType,
    MAX_CONSENSUS_PAYLOAD_BYTES,
)
from chakrview.cognition.federation.consensus.errors import (
    InvalidProposalError,
    StaleConsensusMessageError,
    DoubleVotingError,
    EquivocationError,
    UnauthorizedValidatorError,
    InvalidQuorumCertificateError,
    TenantConsensusIsolationError,
)

logger = logging.getLogger("chakrview.consensus.validator")

# Prohibited keyword patterns for secret and neural weight leakage prevention
PROHIBITED_KEYWORD_PATTERNS = [
    re.compile(r"private[_-]?key", re.IGNORECASE),
    re.compile(r"session[_-]?secret", re.IGNORECASE),
    re.compile(r"api[_-]?secret", re.IGNORECASE),
    re.compile(r"auth[_-]?token", re.IGNORECASE),
    re.compile(r"model[_-]?weight", re.IGNORECASE),
    re.compile(r"raw[_-]?weights", re.IGNORECASE),
    re.compile(r"neural[_-]?tensor", re.IGNORECASE),
    re.compile(r"secret[_-]?key", re.IGNORECASE),
]


class ConsensusValidator:
    """
    Validates proposals, votes, and quorum certificates.
    Detects Byzantine faults including proposer equivocation and double voting.
    """

    def __init__(
        self,
        config: Optional[ConsensusConfig] = None,
        authorized_validators: Optional[Set[str]] = None,
    ) -> None:
        self.config = config or ConsensusConfig()
        self._authorized_validators: Set[str] = set(authorized_validators) if authorized_validators else set()
        self._lock = threading.RLock()

        # Equivocation tracking: (epoch, round, height, proposer_id) -> first seen ConsensusProposal
        self._seen_proposals: Dict[Tuple[int, int, int, str], ConsensusProposal] = {}
        # Double voting tracking: (epoch, round, height, voter_id, vote_type) -> first seen ConsensusVote
        self._seen_votes: Dict[Tuple[int, int, int, str, VoteType], ConsensusVote] = {}
        # Slashing / Equivocation evidence registry
        self._equivocation_records: List[EquivocationEvidence] = []

    def set_authorized_validators(self, validators: Set[str]) -> None:
        """Update active consensus validator set."""
        with self._lock:
            self._authorized_validators = set(validators)

    def get_authorized_validators(self) -> Set[str]:
        with self._lock:
            return set(self._authorized_validators)

    def validate_proposal(
        self,
        proposal: ConsensusProposal,
        current_epoch: int,
        current_height: int,
    ) -> bool:
        """
        Validate candidate proposal syntax, size, freshness, tenant isolation,
        prohibited content, and check for proposer equivocation.
        """
        with self._lock:
            # 1. Proposer authorization
            if self._authorized_validators and proposal.proposer_id not in self._authorized_validators:
                raise UnauthorizedValidatorError(
                    f"Proposer {proposal.proposer_id} is not an authorized validator",
                    node_id=proposal.proposer_id,
                    height=proposal.height,
                )

            # 2. Payload size bounds
            payload_str = json.dumps(proposal.payload, sort_keys=True)
            if len(payload_str.encode("utf-8")) > self.config.max_payload_size_bytes:
                raise InvalidProposalError(
                    f"Proposal {proposal.proposal_id} payload exceeds maximum size "
                    f"({len(payload_str.encode('utf-8'))} > {self.config.max_payload_size_bytes})",
                    node_id=proposal.proposer_id,
                    height=proposal.height,
                )

            # 3. Prohibited content scan (Zero Secret / Zero Weight leakage)
            self._scan_prohibited_content(payload_str)

            # 4. Cryptographic digest integrity
            if not proposal.verify_integrity():
                raise InvalidProposalError(
                    f"Proposal {proposal.proposal_id} failed integrity check: digest mismatch",
                    node_id=proposal.proposer_id,
                    height=proposal.height,
                )

            # 5. Monotonic slot progression (Freshness)
            if proposal.epoch < current_epoch:
                raise StaleConsensusMessageError(
                    f"Stale proposal epoch {proposal.epoch} < current {current_epoch}",
                    node_id=proposal.proposer_id,
                    height=proposal.height,
                )
            if proposal.height <= current_height:
                raise StaleConsensusMessageError(
                    f"Stale proposal height {proposal.height} <= current committed {current_height}",
                    node_id=proposal.proposer_id,
                    height=proposal.height,
                )

            # 6. Tenant isolation
            if (
                self.config.tenant_id != "*"
                and proposal.tenant_id != "*"
                and proposal.tenant_id != self.config.tenant_id
            ):
                raise TenantConsensusIsolationError(
                    f"Proposal tenant {proposal.tenant_id} does not match node tenant {self.config.tenant_id}",
                    node_id=proposal.proposer_id,
                    height=proposal.height,
                )

            # 7. Byzantine Equivocation Check
            slot_key = (proposal.epoch, proposal.round, proposal.height, proposal.proposer_id)
            existing = self._seen_proposals.get(slot_key)
            if existing is not None:
                if existing.proposal_digest != proposal.proposal_digest:
                    # Byzantine double-proposal detected!
                    evidence = EquivocationEvidence(
                        evidence_id=f"eqv_{proposal.epoch}_{proposal.round}_{proposal.height}_{proposal.proposer_id}",
                        epoch=proposal.epoch,
                        round=proposal.round,
                        height=proposal.height,
                        offender_id=proposal.proposer_id,
                        proposal_digest_1=existing.proposal_digest,
                        proposal_digest_2=proposal.proposal_digest,
                    )
                    self._equivocation_records.append(evidence)
                    logger.critical(
                        "BYZANTINE EQUIVOCATION DETECTED: Proposer %s sent conflicting proposals for slot %s",
                        proposal.proposer_id, slot_key,
                    )
                    raise EquivocationError(
                        f"Proposer {proposal.proposer_id} equivocated at height {proposal.height}, round {proposal.round}",
                        node_id=proposal.proposer_id,
                        height=proposal.height,
                    )
                return True

            self._seen_proposals[slot_key] = proposal
            return True

    def validate_vote(
        self,
        vote: ConsensusVote,
        current_epoch: int,
        current_height: int,
    ) -> bool:
        """
        Validate validator vote integrity, authorization, freshness, and detect double-voting.
        """
        with self._lock:
            # 1. Voter authorization
            if self._authorized_validators and vote.voter_id not in self._authorized_validators:
                raise UnauthorizedValidatorError(
                    f"Voter {vote.voter_id} is not an authorized consensus validator",
                    node_id=vote.voter_id,
                    height=vote.height,
                )

            # 2. Freshness check
            if vote.epoch < current_epoch:
                raise StaleConsensusMessageError(
                    f"Stale vote epoch {vote.epoch} < current {current_epoch}",
                    node_id=vote.voter_id,
                    height=vote.height,
                )
            if vote.height <= current_height:
                raise StaleConsensusMessageError(
                    f"Stale vote height {vote.height} <= current {current_height}",
                    node_id=vote.voter_id,
                    height=vote.height,
                )

            # 3. Byzantine Double-Voting Check
            vote_key = (vote.epoch, vote.round, vote.height, vote.voter_id, vote.vote_type)
            existing = self._seen_votes.get(vote_key)
            if existing is not None:
                if existing.proposal_digest != vote.proposal_digest:
                    logger.critical(
                        "BYZANTINE DOUBLE-VOTING DETECTED: Validator %s cast contradictory %s votes: %s vs %s",
                        vote.voter_id, vote.vote_type.value, existing.proposal_digest, vote.proposal_digest,
                    )
                    raise DoubleVotingError(
                        f"Validator {vote.voter_id} double-voted in round {vote.round}, height {vote.height}",
                        node_id=vote.voter_id,
                        height=vote.height,
                    )
                return True

            self._seen_votes[vote_key] = vote
            return True

    def validate_quorum_certificate(
        self,
        qc: QuorumCertificate,
        validator_count: Optional[int] = None,
    ) -> bool:
        """
        Validate that a Quorum Certificate satisfies the required quorum threshold
        and contains non-duplicate, authorized votes.
        """
        with self._lock:
            val_count = validator_count if validator_count is not None else len(self._authorized_validators)
            required_quorum = self.config.compute_quorum_threshold(val_count)

            # Check unique voters
            unique_voters = set(qc.voter_ids)
            if len(unique_voters) < len(qc.voter_ids):
                raise InvalidQuorumCertificateError(
                    f"QC {qc.qc_id} contains duplicate voter IDs",
                    height=qc.height,
                )

            # Verify all voters are authorized
            if self._authorized_validators:
                unauthorized = unique_voters - self._authorized_validators
                if unauthorized:
                    raise UnauthorizedValidatorError(
                        f"QC {qc.qc_id} contains unauthorized voters: {unauthorized}",
                        height=qc.height,
                    )

            # Verify quorum threshold
            if len(unique_voters) < required_quorum:
                raise InvalidQuorumCertificateError(
                    f"QC {qc.qc_id} voter count {len(unique_voters)} is below required quorum {required_quorum} "
                    f"(total validators: {val_count}, mode: {self.config.fault_tolerance_mode.value})",
                    height=qc.height,
                )

            return True

    def get_equivocation_evidence(self) -> List[EquivocationEvidence]:
        """Retrieve all recorded Byzantine equivocation records."""
        with self._lock:
            return list(self._equivocation_records)

    def _scan_prohibited_content(self, payload_text: str) -> None:
        """Enforces Zero Secret and Zero Model Weight leakage in consensus payloads."""
        for pattern in PROHIBITED_KEYWORD_PATTERNS:
            if pattern.search(payload_text):
                raise InvalidProposalError(
                    f"Consensus proposal contains prohibited keyword matching: {pattern.pattern}"
                )
