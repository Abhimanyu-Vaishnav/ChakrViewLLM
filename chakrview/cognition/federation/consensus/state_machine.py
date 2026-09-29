"""
Deterministic Replicated State Machine (RSM) with Hash Chaining and Sovereign Local Defense (Step 40).
"""

import hashlib
import json
import logging
import threading
import time
from typing import Dict, List, Optional, Set, Any, Tuple

from chakrview.cognition.federation.consensus.models import (
    ConsensusProposal,
    QuorumCertificate,
    ConsensusLogEntry,
    ConsensusTransitionType,
    GENESIS_PARENT_HASH,
)
from chakrview.cognition.federation.consensus.errors import (
    StateDivergenceError,
    SovereignPolicyViolationError,
)

logger = logging.getLogger("chakrview.consensus.state_machine")


class ReplicatedStateMachine:
    """
    Deterministic State Machine Replication engine.
    Applies consensus proposals committed via Quorum Certificates into a verifiable,
    monotonic state root hash chain.
    
    Axiom:
    LOCAL_POLICY > CONSENSUS_DECISION
    CONSENSUS != AUTHORITY
    A quorum agreement among remote peers can NEVER override sovereign local policy.
    """

    def __init__(
        self,
        local_node_id: str,
        capability_gate: Optional[Any] = None,
        journal: Optional[Any] = None,
    ) -> None:
        self.local_node_id = local_node_id
        self.capability_gate = capability_gate
        self.journal = journal
        self._lock = threading.RLock()

        # Log & State
        self._log: List[ConsensusLogEntry] = []
        self._current_height = 0
        self._state_root_hash = GENESIS_PARENT_HASH

        # State tables updated via consensus
        self._active_membership: Set[str] = set()
        self._revoked_nodes: Set[str] = set()
        self._quarantined_nodes: Set[str] = set()
        self._committed_tasks: Dict[str, Dict[str, Any]] = {}
        self._authorized_checkpoints: Dict[str, str] = {}
        self._epoch = 1

    @property
    def current_height(self) -> int:
        with self._lock:
            return self._current_height

    @property
    def state_root_hash(self) -> str:
        with self._lock:
            return self._state_root_hash

    @property
    def epoch(self) -> int:
        with self._lock:
            return self._epoch

    def get_active_membership(self) -> Set[str]:
        with self._lock:
            return set(self._active_membership)

    def get_revoked_nodes(self) -> Set[str]:
        with self._lock:
            return set(self._revoked_nodes)

    def is_node_revoked(self, node_id: str) -> bool:
        with self._lock:
            return node_id in self._revoked_nodes

    def apply_committed_proposal(
        self,
        proposal: ConsensusProposal,
        quorum_certificate: QuorumCertificate,
    ) -> ConsensusLogEntry:
        """
        Atomically commit a proposal backed by a valid Quorum Certificate.
        Validates parent hash linkage, evaluates local sovereign policy, and advances the state root.
        """
        with self._lock:
            expected_height = self._current_height + 1
            if proposal.height != expected_height:
                raise StateDivergenceError(
                    f"Proposal height {proposal.height} does not match expected log height {expected_height}",
                    node_id=self.local_node_id,
                    height=proposal.height,
                )

            # Validate parent hash linkage
            if proposal.parent_hash != self._state_root_hash:
                raise StateDivergenceError(
                    f"Proposal parent hash {proposal.parent_hash} does not match current state root {self._state_root_hash}",
                    node_id=self.local_node_id,
                    height=proposal.height,
                )

            # Sovereign Local Policy Check: CONSENSUS != AUTHORITY
            self._verify_sovereign_policy(proposal)

            # Apply state mutation
            self._execute_transition(proposal)

            # Compute new monotonic state root hash
            canonical_entry = (
                f"{self._state_root_hash}:{proposal.height}:{proposal.proposal_digest}:{quorum_certificate.compute_digest()}"
            )
            new_state_root = hashlib.sha256(canonical_entry.encode("utf-8")).hexdigest()

            entry = ConsensusLogEntry(
                height=proposal.height,
                epoch=proposal.epoch,
                round=proposal.round,
                proposal=proposal,
                quorum_certificate=quorum_certificate,
                state_root_hash=new_state_root,
            )

            self._log.append(entry)
            self._current_height = proposal.height
            self._state_root_hash = new_state_root

            # Append to WAL if configured
            if self.journal and hasattr(self.journal, "append_entry"):
                self.journal.append_entry(
                    "CONSENSUS_BLOCK_COMMITTED",
                    {
                        "height": entry.height,
                        "epoch": entry.epoch,
                        "state_root": new_state_root,
                        "proposal_digest": proposal.proposal_digest,
                    },
                )

            logger.info(
                "Committed consensus entry height=%d, transition=%s, new_state_root=%s",
                entry.height, proposal.transition_type.value, new_state_root[:16],
            )
            return entry

    def _verify_sovereign_policy(self, proposal: ConsensusProposal) -> None:
        """
        Enforce: LOCAL_POLICY > CONSENSUS_DECISION.
        Even if a cluster majority agreed, local sovereignty can deny local actuation.
        """
        # If capability gate is attached, evaluate action
        if self.capability_gate and hasattr(self.capability_gate, "authorize"):
            # Check if transition commands an action requiring capability authorization
            cap_id = proposal.payload.get("capability_id")
            if cap_id:
                authorized = self.capability_gate.authorize(cap_id, context={"proposal": proposal.to_dict()})
                if not authorized:
                    raise SovereignPolicyViolationError(
                        f"Consensus proposal {proposal.proposal_id} commanded capability {cap_id} "
                        f"which was DENIED by local sovereign policy",
                        node_id=self.local_node_id,
                        height=proposal.height,
                    )

        # Prohibit self-revocation or self-quarantine by remote quorum without local consent
        target_revoked = proposal.payload.get("target_node_id")
        if (
            proposal.transition_type == ConsensusTransitionType.PEER_REVOCATION_AGREEMENT
            and target_revoked == self.local_node_id
            and proposal.payload.get("unilateral_remote_force", False)
        ):
            raise SovereignPolicyViolationError(
                "Remote consensus quorum cannot unilaterally force local self-revocation",
                node_id=self.local_node_id,
                height=proposal.height,
            )

    def _execute_transition(self, proposal: ConsensusProposal) -> None:
        """Execute in-memory state transition according to transition taxonomy."""
        payload = proposal.payload

        if proposal.transition_type == ConsensusTransitionType.MEMBERSHIP_TOPOLOGY_UPDATE:
            add_nodes = set(payload.get("added_nodes", []))
            remove_nodes = set(payload.get("removed_nodes", []))
            self._active_membership = (self._active_membership | add_nodes) - remove_nodes

        elif proposal.transition_type == ConsensusTransitionType.PEER_REVOCATION_AGREEMENT:
            target = payload.get("target_node_id")
            if target:
                self._revoked_nodes.add(target)
                self._active_membership.discard(target)

        elif proposal.transition_type == ConsensusTransitionType.TASK_COMMIT_FINALIZATION:
            task_id = payload.get("task_id")
            if task_id:
                self._committed_tasks[task_id] = payload.get("result", {})

        elif proposal.transition_type == ConsensusTransitionType.CHECKPOINT_AUTHORIZATION:
            cp_id = payload.get("checkpoint_id")
            unit_id = payload.get("unit_id")
            if cp_id and unit_id:
                self._authorized_checkpoints[unit_id] = cp_id

        elif proposal.transition_type == ConsensusTransitionType.EPOCH_ADVANCEMENT:
            new_epoch = payload.get("new_epoch", self._epoch + 1)
            self._epoch = new_epoch

    def get_log_entries(self, from_height: int = 1, to_height: Optional[int] = None) -> List[ConsensusLogEntry]:
        """Retrieve slice of committed consensus entries."""
        with self._lock:
            end = to_height if to_height is not None else self._current_height
            return [e for e in self._log if from_height <= e.height <= end]
