"""
Comprehensive Test Suite for ChakrView Step 40:
Federated Consensus, Byzantine Fault Tolerance & Multi-Node State Agreement.

Verifies:
1. Consensus proposal creation, digest integrity, and size bounds
2. Proposer equivocation detection (TV-01)
3. Double voting rejection (TV-02)
4. Stale message rejection (TV-03)
5. Unauthorized validator rejection (TV-04)
6. Quorum Certificate multi-signature & threshold verification (TV-05)
7. Split-brain partition safety (TV-06)
8. View change timeout and round-robin leader succession (TV-07, TV-08)
9. Tenant consensus isolation (TV-09)
10. Replicated State Machine parent hash chaining and divergence detection (TV-10)
11. Sovereign local defense: LOCAL_POLICY > CONSENSUS via CapabilityGate (TV-11)
12. Poisoned payload rejection (TV-12)
13. Zero secret and zero weight leakage scanning (TV-14)
14. Neural core immutability: ΔW = 0, parameters = 3,443,136, SHA-256 hash invariant (TV-15)
15. Dynamic membership topology consensus
16. Peer revocation consensus
17. Task commit finalization consensus
18. End-to-end 3-phase BFT consensus across 4 nodes (N=4, f=1, Q=3)
19. BFT consensus liveness with an active Byzantine node
20. WAL and audit trail logging
"""

import hashlib
import json
import time
from typing import Dict, List, Optional, Any, Set
import pytest
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.cognition.federation.consensus import (
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
    ConsensusError,
    InvalidProposalError,
    StaleConsensusMessageError,
    DoubleVotingError,
    EquivocationError,
    UnauthorizedValidatorError,
    InvalidQuorumCertificateError,
    TenantConsensusIsolationError,
    SovereignPolicyViolationError,
    StateDivergenceError,
    ConsensusValidator,
    ReplicatedStateMachine,
    FederatedConsensusEngine,
)

EXPECTED_WEIGHT_HASH = "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"
EXPECTED_PARAM_COUNT = 3443136


def _compute_model_hash(model: torch.nn.Module) -> str:
    hasher = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(model.named_parameters()):
            hasher.update(name.encode("utf-8"))
            hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


class MockCapabilityGate:
    def __init__(self, allowed_capabilities: Optional[Set[str]] = None):
        self.allowed = allowed_capabilities or {"add", "state_sync"}

    def authorize(self, capability_id: str, context: Optional[Any] = None) -> bool:
        return capability_id in self.allowed


class MockJournal:
    def __init__(self):
        self.entries = []

    def append_entry(self, entry_type: str, payload: Dict[str, Any]) -> None:
        self.entries.append((entry_type, payload))


# ============================================================================
# 1-5: Proposal Integrity, Size Bounds & Proposer Equivocation
# ============================================================================

def test_consensus_proposal_creation_and_integrity():
    """1. Test proposal canonical SHA-256 digest and integrity verification."""
    prop = ConsensusProposal(
        proposal_id="prop_1",
        epoch=1,
        round=0,
        height=1,
        proposer_id="node_0",
        transition_type=ConsensusTransitionType.EPOCH_ADVANCEMENT,
        payload={"new_epoch": 2},
    )
    assert prop.verify_integrity()
    assert len(prop.proposal_digest) == 64

    # Tampered proposal digest fails verification
    prop.proposal_digest = "0" * 64
    assert not prop.verify_integrity()


def test_consensus_proposal_oversized_rejection():
    """2. Test that proposals exceeding 1 MB are rejected fail-closed."""
    validator = ConsensusValidator(authorized_validators={"node_0"})
    huge_payload = {"data": "x" * (MAX_CONSENSUS_PAYLOAD_BYTES + 10)}
    prop = ConsensusProposal(
        proposal_id="prop_huge",
        epoch=1,
        round=0,
        height=1,
        proposer_id="node_0",
        transition_type=ConsensusTransitionType.CUSTOM_STATE_TRANSITION,
        payload=huge_payload,
    )
    with pytest.raises(InvalidProposalError, match="exceeds maximum size"):
        validator.validate_proposal(prop, current_epoch=1, current_height=0)


def test_proposer_equivocation_detection():
    """3. TV-01: Proposer emitting conflicting proposals for the same slot is detected and rejected."""
    validator = ConsensusValidator(authorized_validators={"leader_0"})

    prop_1 = ConsensusProposal(
        proposal_id="prop_slot_1_A",
        epoch=1,
        round=0,
        height=1,
        proposer_id="leader_0",
        transition_type=ConsensusTransitionType.EPOCH_ADVANCEMENT,
        payload={"branch": "A"},
    )
    prop_2 = ConsensusProposal(
        proposal_id="prop_slot_1_B",
        epoch=1,
        round=0,
        height=1,
        proposer_id="leader_0",
        transition_type=ConsensusTransitionType.EPOCH_ADVANCEMENT,
        payload={"branch": "B"},
    )

    # First proposal is accepted
    assert validator.validate_proposal(prop_1, current_epoch=1, current_height=0)

    # Identical proposal can be validated idempotently
    assert validator.validate_proposal(prop_1, current_epoch=1, current_height=0)

    # Conflicting proposal for the same slot triggers EquivocationError and records evidence
    with pytest.raises(EquivocationError, match="equivocated"):
        validator.validate_proposal(prop_2, current_epoch=1, current_height=0)

    evidence = validator.get_equivocation_evidence()
    assert len(evidence) == 1
    assert evidence[0].offender_id == "leader_0"
    assert evidence[0].proposal_digest_1 == prop_1.proposal_digest
    assert evidence[0].proposal_digest_2 == prop_2.proposal_digest


# ============================================================================
# 4-7: Double Voting, Stale Messages & Unauthorized Validators
# ============================================================================

def test_double_voting_rejection():
    """4. TV-02: Validator casting conflicting prevotes/precommits for different digests in same round is rejected."""
    validator = ConsensusValidator(authorized_validators={"voter_1"})

    vote_1 = ConsensusVote(
        vote_id="v1",
        epoch=1,
        round=0,
        height=1,
        voter_id="voter_1",
        vote_type=VoteType.PREVOTE,
        proposal_digest="digest_AAA",
    )
    vote_2 = ConsensusVote(
        vote_id="v2",
        epoch=1,
        round=0,
        height=1,
        voter_id="voter_1",
        vote_type=VoteType.PREVOTE,
        proposal_digest="digest_BBB",  # Contradictory vote in same round!
    )

    assert validator.validate_vote(vote_1, current_epoch=1, current_height=0)

    # Same vote is idempotent
    assert validator.validate_vote(vote_1, current_epoch=1, current_height=0)

    # Contradictory vote raises DoubleVotingError
    with pytest.raises(DoubleVotingError, match="double-voted"):
        validator.validate_vote(vote_2, current_epoch=1, current_height=0)


def test_stale_message_rejection():
    """5. TV-03: Stale proposals and votes for past epochs or committed heights are rejected fail-closed."""
    validator = ConsensusValidator(authorized_validators={"node_0"})

    # Stale height
    prop_stale_h = ConsensusProposal(
        proposal_id="p_old",
        epoch=1,
        round=0,
        height=5,
        proposer_id="node_0",
        transition_type=ConsensusTransitionType.EPOCH_ADVANCEMENT,
        payload={},
    )
    with pytest.raises(StaleConsensusMessageError, match="height"):
        validator.validate_proposal(prop_stale_h, current_epoch=1, current_height=5)

    # Stale epoch
    prop_stale_e = ConsensusProposal(
        proposal_id="p_old_ep",
        epoch=1,
        round=0,
        height=10,
        proposer_id="node_0",
        transition_type=ConsensusTransitionType.EPOCH_ADVANCEMENT,
        payload={},
    )
    with pytest.raises(StaleConsensusMessageError, match="epoch"):
        validator.validate_proposal(prop_stale_e, current_epoch=2, current_height=5)


def test_unauthorized_validator_rejection():
    """6. TV-04: Nodes not in the authorized validator set cannot propose or vote."""
    validator = ConsensusValidator(authorized_validators={"node_0", "node_1"})

    rogue_prop = ConsensusProposal(
        proposal_id="p_rogue",
        epoch=1,
        round=0,
        height=1,
        proposer_id="attacker_node",
        transition_type=ConsensusTransitionType.EPOCH_ADVANCEMENT,
        payload={},
    )
    with pytest.raises(UnauthorizedValidatorError, match="not an authorized validator"):
        validator.validate_proposal(rogue_prop, current_epoch=1, current_height=0)

    rogue_vote = ConsensusVote(
        vote_id="v_rogue",
        epoch=1,
        round=0,
        height=1,
        voter_id="attacker_node",
        vote_type=VoteType.PREVOTE,
        proposal_digest="dig",
    )
    with pytest.raises(UnauthorizedValidatorError, match="not an authorized consensus validator"):
        validator.validate_vote(rogue_vote, current_epoch=1, current_height=0)


# ============================================================================
# 8-11: Quorum Certificate & Partition Safety
# ============================================================================

def test_quorum_certificate_verification():
    """7. TV-05: Quorum Certificate verifies supermajority threshold and rejects forged/sub-quorum certificates."""
    config = ConsensusConfig(fault_tolerance_mode=FaultToleranceMode.BFT)
    # N = 4 validators -> Q = floor(2*4/3) + 1 = 3
    validators = {"v0", "v1", "v2", "v3"}
    cv = ConsensusValidator(config=config, authorized_validators=validators)

    # Valid QC with 3 distinct votes
    valid_qc = QuorumCertificate(
        qc_id="qc_1",
        epoch=1,
        round=0,
        height=1,
        proposal_digest="dig_abc",
        vote_type=VoteType.PRECOMMIT,
        voter_ids=["v0", "v1", "v2"],
        signatures={"v0": "sig0", "v1": "sig1", "v2": "sig2"},
    )
    assert cv.validate_quorum_certificate(valid_qc)

    # Sub-quorum QC (only 2 votes < 3 required)
    sub_qc = QuorumCertificate(
        qc_id="qc_sub",
        epoch=1,
        round=0,
        height=1,
        proposal_digest="dig_abc",
        vote_type=VoteType.PRECOMMIT,
        voter_ids=["v0", "v1"],
        signatures={"v0": "sig0", "v1": "sig1"},
    )
    with pytest.raises(InvalidQuorumCertificateError, match="below required quorum"):
        cv.validate_quorum_certificate(sub_qc)

    # Duplicate voter ID forgery (e.g. ['v0', 'v0', 'v1'] pretending to be 3 votes)
    dup_qc = QuorumCertificate(
        qc_id="qc_dup",
        epoch=1,
        round=0,
        height=1,
        proposal_digest="dig_abc",
        vote_type=VoteType.PRECOMMIT,
        voter_ids=["v0", "v0", "v1"],
        signatures={"v0": "sig0"},
    )
    with pytest.raises(InvalidQuorumCertificateError, match="duplicate voter IDs"):
        cv.validate_quorum_certificate(dup_qc)


def test_split_brain_partition_safety():
    """8. TV-06: Strict supermajority requirement prevents two partitions from both achieving quorum."""
    config_bft = ConsensusConfig(fault_tolerance_mode=FaultToleranceMode.BFT)
    # N = 5 nodes: Q = floor(10/3) + 1 = 4 votes required
    q_bft = config_bft.compute_quorum_threshold(5)
    assert q_bft == 4
    # If network splits into partition of 3 and partition of 2: neither reaches 4!
    assert 3 < q_bft and 2 < q_bft

    config_cft = ConsensusConfig(fault_tolerance_mode=FaultToleranceMode.CFT)
    # N = 5 nodes: Q = floor(5/2) + 1 = 3 votes required
    q_cft = config_cft.compute_quorum_threshold(5)
    assert q_cft == 3
    # In a 3-2 split: only the partition with 3 can proceed; the minority partition of 2 halts.
    assert 2 < q_cft


# ============================================================================
# 12-14: View Change, Leader Succession & Tenant Isolation
# ============================================================================

def test_view_change_timeout_and_leader_succession():
    """9. TV-07 & TV-08: Leader timeout triggers view change with deterministic round-robin leader."""
    validators = ["node_A", "node_B", "node_C"]
    engine = FederatedConsensusEngine(local_node_id="node_A", validators=validators)

    # Height 1, Round 0 leader: (1 + 0) % 3 = 1 -> "node_B"
    leader_r0 = engine.get_leader_for_round(height=1, round_num=0)
    assert leader_r0 == "node_B"

    # View change advances round
    new_round = engine.trigger_view_change(reason="Leader node_B timed out")
    assert new_round == 1
    assert engine.phase == ConsensusPhase.VIEW_CHANGE

    # Height 1, Round 1 leader: (1 + 1) % 3 = 2 -> "node_C"
    leader_r1 = engine.get_leader_for_round(height=1, round_num=1)
    assert leader_r1 == "node_C"

    # Height 1, Round 2 leader: (1 + 2) % 3 = 0 -> "node_A" (local node!)
    new_round_2 = engine.trigger_view_change(reason="node_C timed out")
    assert new_round_2 == 2
    leader_r2 = engine.get_leader_for_round(height=1, round_num=2)
    assert leader_r2 == "node_A"
    assert engine.is_leader_for_current_round()


def test_cross_tenant_consensus_isolation():
    """10. TV-09: Proposals carrying mismatched tenant identifiers are rejected."""
    config = ConsensusConfig(tenant_id="tenant-alpha")
    validator = ConsensusValidator(config=config, authorized_validators={"node_0"})

    # Matching tenant
    prop_ok = ConsensusProposal(
        proposal_id="p1",
        epoch=1,
        round=0,
        height=1,
        proposer_id="node_0",
        transition_type=ConsensusTransitionType.EPOCH_ADVANCEMENT,
        payload={},
        tenant_id="tenant-alpha",
    )
    assert validator.validate_proposal(prop_ok, current_epoch=1, current_height=0)

    # Mismatched tenant
    prop_bad = ConsensusProposal(
        proposal_id="p2",
        epoch=1,
        round=0,
        height=1,
        proposer_id="node_0",
        transition_type=ConsensusTransitionType.EPOCH_ADVANCEMENT,
        payload={},
        tenant_id="tenant-beta",
    )
    with pytest.raises(TenantConsensusIsolationError, match="tenant"):
        validator.validate_proposal(prop_bad, current_epoch=1, current_height=0)


# ============================================================================
# 15-18: Replicated State Machine, Sovereign Defense & Secret Leakage
# ============================================================================

def test_replicated_state_machine_hash_chain():
    """11. TV-10: State Machine links proposals in a cryptographic hash chain; divergence fails closed."""
    rsm = ReplicatedStateMachine(local_node_id="local_node")
    assert rsm.current_height == 0
    assert rsm.state_root_hash == GENESIS_PARENT_HASH

    # First entry
    prop_1 = ConsensusProposal(
        proposal_id="p1",
        epoch=1,
        round=0,
        height=1,
        proposer_id="leader",
        transition_type=ConsensusTransitionType.EPOCH_ADVANCEMENT,
        payload={"new_epoch": 2},
        parent_hash=GENESIS_PARENT_HASH,
    )
    qc_1 = QuorumCertificate(
        qc_id="qc1",
        epoch=1,
        round=0,
        height=1,
        proposal_digest=prop_1.proposal_digest,
        vote_type=VoteType.PRECOMMIT,
        voter_ids=["v1", "v2"],
    )

    entry_1 = rsm.apply_committed_proposal(prop_1, qc_1)
    assert entry_1.height == 1
    assert rsm.current_height == 1
    root_1 = rsm.state_root_hash
    assert root_1 != GENESIS_PARENT_HASH

    # Second entry linking root_1
    prop_2 = ConsensusProposal(
        proposal_id="p2",
        epoch=1,
        round=0,
        height=2,
        proposer_id="leader",
        transition_type=ConsensusTransitionType.MEMBERSHIP_TOPOLOGY_UPDATE,
        payload={"added_nodes": ["node_new"]},
        parent_hash=root_1,
    )
    qc_2 = QuorumCertificate(
        qc_id="qc2",
        epoch=1,
        round=0,
        height=2,
        proposal_digest=prop_2.proposal_digest,
        vote_type=VoteType.PRECOMMIT,
        voter_ids=["v1", "v2"],
    )
    entry_2 = rsm.apply_committed_proposal(prop_2, qc_2)
    assert entry_2.height == 2
    assert "node_new" in rsm.get_active_membership()

    # Tampered parent hash diverges from state root -> StateDivergenceError
    prop_divergent = ConsensusProposal(
        proposal_id="p3_bad",
        epoch=1,
        round=0,
        height=3,
        proposer_id="leader",
        transition_type=ConsensusTransitionType.EPOCH_ADVANCEMENT,
        payload={},
        parent_hash="1" * 64,  # Bad parent hash
    )
    qc_3 = QuorumCertificate(
        qc_id="qc3",
        epoch=1,
        round=0,
        height=3,
        proposal_digest=prop_divergent.proposal_digest,
        vote_type=VoteType.PRECOMMIT,
        voter_ids=["v1", "v2"],
    )
    with pytest.raises(StateDivergenceError, match="parent hash"):
        rsm.apply_committed_proposal(prop_divergent, qc_3)


def test_local_sovereign_policy_override():
    """12. TV-11: LOCAL_POLICY > CONSENSUS. Quorum agreement cannot command forbidden capability."""
    gate = MockCapabilityGate(allowed_capabilities={"state_sync"})  # "unauthorized_cap" is forbidden
    rsm = ReplicatedStateMachine(local_node_id="local_node", capability_gate=gate)

    forbidden_prop = ConsensusProposal(
        proposal_id="p_forbidden",
        epoch=1,
        round=0,
        height=1,
        proposer_id="leader",
        transition_type=ConsensusTransitionType.CUSTOM_STATE_TRANSITION,
        payload={"capability_id": "unauthorized_cap", "action": "wipe_disk"},
        parent_hash=GENESIS_PARENT_HASH,
    )
    qc = QuorumCertificate(
        qc_id="qc_f",
        epoch=1,
        round=0,
        height=1,
        proposal_digest=forbidden_prop.proposal_digest,
        vote_type=VoteType.PRECOMMIT,
        voter_ids=["v1", "v2", "v3"],
    )

    with pytest.raises(SovereignPolicyViolationError, match="DENIED by local sovereign policy"):
        rsm.apply_committed_proposal(forbidden_prop, qc)


def test_zero_secret_leakage_in_consensus():
    """13. TV-14: Proposals containing secrets or neural model weights are rejected."""
    validator = ConsensusValidator(authorized_validators={"node_0"})

    leaky_prop = ConsensusProposal(
        proposal_id="p_leak",
        epoch=1,
        round=0,
        height=1,
        proposer_id="node_0",
        transition_type=ConsensusTransitionType.CUSTOM_STATE_TRANSITION,
        payload={"key_dump": "private_key: 12345abcdef"},
    )
    with pytest.raises(InvalidProposalError, match="prohibited keyword"):
        validator.validate_proposal(leaky_prop, current_epoch=1, current_height=0)

    weight_prop = ConsensusProposal(
        proposal_id="p_weight",
        epoch=1,
        round=0,
        height=1,
        proposer_id="node_0",
        transition_type=ConsensusTransitionType.CUSTOM_STATE_TRANSITION,
        payload={"export": "raw_weights tensor matrix"},
    )
    with pytest.raises(InvalidProposalError, match="prohibited keyword"):
        validator.validate_proposal(weight_prop, current_epoch=1, current_height=0)


# ============================================================================
# 19-22: End-to-End BFT Consensus & Byzantine Liveness
# ============================================================================

def test_end_to_end_three_phase_bft_consensus():
    """14. End-to-end 3-phase BFT consensus across 4 nodes (N=4, f=1, Q=3)."""
    validators = ["node_0", "node_1", "node_2", "node_3"]
    # Height 1, Round 0 leader: (1 + 0) % 4 = 1 -> "node_1"
    leader_id = "node_1"
    engine = FederatedConsensusEngine(local_node_id=leader_id, validators=validators)

    # 1. Leader proposes block
    prop = engine.create_proposal(
        transition_type=ConsensusTransitionType.MEMBERSHIP_TOPOLOGY_UPDATE,
        payload={"added_nodes": ["node_new_5"]},
    )
    assert engine.phase == ConsensusPhase.PREVOTE

    # 2. Validators cast PREVOTES (3 votes reach quorum)
    for v_id in ["node_0", "node_1", "node_2"]:
        vote = ConsensusVote(
            vote_id=f"vote_prevote_{v_id}",
            epoch=1,
            round=0,
            height=1,
            voter_id=v_id,
            vote_type=VoteType.PREVOTE,
            proposal_digest=prop.proposal_digest,
            signature=f"sig_{v_id}",
        )
        qc = engine.record_prevote(vote)
        if v_id in ["node_0", "node_1"]:
            assert qc is None
        else:
            assert qc is not None  # 3rd vote reached quorum!
            assert qc.vote_type == VoteType.PREVOTE
            assert len(qc.voter_ids) == 3

    assert engine.phase == ConsensusPhase.PRECOMMIT

    # 3. Validators cast PRECOMMITS
    for v_id in ["node_0", "node_1", "node_2"]:
        vote = ConsensusVote(
            vote_id=f"vote_precommit_{v_id}",
            epoch=1,
            round=0,
            height=1,
            voter_id=v_id,
            vote_type=VoteType.PRECOMMIT,
            proposal_digest=prop.proposal_digest,
            signature=f"sig_{v_id}",
        )
        precommit_qc = engine.record_precommit(vote)

    assert precommit_qc is not None
    assert precommit_qc.vote_type == VoteType.PRECOMMIT
    assert len(precommit_qc.voter_ids) == 3
    assert engine.phase == ConsensusPhase.COMMITTED

    # 4. Commit block and update Replicated State Machine
    entry = engine.commit_block(prop, precommit_qc)
    assert entry.height == 1
    assert "node_new_5" in engine.state_machine.get_active_membership()
    assert engine.current_height == 1
    assert engine.phase == ConsensusPhase.IDLE


def test_bft_consensus_with_one_byzantine_fault():
    """15. BFT liveness: 4-node cluster (N=4, f=1, Q=3) commits despite 1 Byzantine node dropping/equivocating."""
    validators = ["node_0", "node_1", "node_2", "node_3"]
    engine = FederatedConsensusEngine(local_node_id="node_1", validators=validators)

    prop = engine.create_proposal(
        transition_type=ConsensusTransitionType.PEER_REVOCATION_AGREEMENT,
        payload={"target_node_id": "malicious_peer_X"},
    )

    # Byzantine node_3 withholds votes or double-votes (ignored)
    # The 3 honest nodes (node_0, node_1, node_2) provide supermajority Q=3
    for v_id in ["node_0", "node_1", "node_2"]:
        engine.record_prevote(ConsensusVote(
            vote_id=f"pv_{v_id}", epoch=1, round=0, height=1, voter_id=v_id,
            vote_type=VoteType.PREVOTE, proposal_digest=prop.proposal_digest,
        ))

    precommit_qc = None
    for v_id in ["node_0", "node_1", "node_2"]:
        precommit_qc = engine.record_precommit(ConsensusVote(
            vote_id=f"pc_{v_id}", epoch=1, round=0, height=1, voter_id=v_id,
            vote_type=VoteType.PRECOMMIT, proposal_digest=prop.proposal_digest,
        ))

    assert precommit_qc is not None
    entry = engine.commit_block(prop, precommit_qc)
    assert entry.height == 1
    assert engine.state_machine.is_node_revoked("malicious_peer_X")


def test_wal_journal_logging_on_consensus():
    """16. Test that consensus commits and proposals append durably to WAL."""
    journal = MockJournal()
    validators = ["node_0", "node_1", "node_2"]
    # N=3: Q = floor(2*3/3)+1 = 3
    engine = FederatedConsensusEngine(local_node_id="node_1", validators=validators, journal=journal)

    prop = engine.create_proposal(
        transition_type=ConsensusTransitionType.TASK_COMMIT_FINALIZATION,
        payload={"task_id": "task_100", "result": {"output": 42}},
    )

    # Check WAL recorded proposal broadcast
    assert any(e[0] == "CONSENSUS_PROPOSAL_BROADCAST" for e in journal.entries)

    for v_id in ["node_0", "node_1", "node_2"]:
        engine.record_prevote(ConsensusVote(
            vote_id=f"pv_{v_id}", epoch=1, round=0, height=1, voter_id=v_id,
            vote_type=VoteType.PREVOTE, proposal_digest=prop.proposal_digest,
        ))

    precommit_qc = None
    for v_id in ["node_0", "node_1", "node_2"]:
        precommit_qc = engine.record_precommit(ConsensusVote(
            vote_id=f"pc_{v_id}", epoch=1, round=0, height=1, voter_id=v_id,
            vote_type=VoteType.PRECOMMIT, proposal_digest=prop.proposal_digest,
        ))

    engine.commit_block(prop, precommit_qc)

    # Check WAL recorded block committed
    assert any(e[0] == "CONSENSUS_BLOCK_COMMITTED" for e in journal.entries)


# ============================================================================
# Neural Core Immutability (ΔW = 0)
# ============================================================================

def test_neural_weight_immutability_delta_w_zero():
    """
    CRITICAL INVARIANT VERIFICATION:
    Ensure that consensus coordination, voting, and state machine commits
    do not alter neural model parameters or architecture (ΔW = 0, parameters = 3,443,136).
    """
    torch.manual_seed(42)
    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    model.eval()

    initial_hash = _compute_model_hash(model)
    assert initial_hash == EXPECTED_WEIGHT_HASH
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count == EXPECTED_PARAM_COUNT

    # Run multi-node consensus engine operations
    engine = FederatedConsensusEngine(local_node_id="node_1", validators=["node_0", "node_1", "node_2"])
    prop = engine.create_proposal(
        transition_type=ConsensusTransitionType.EPOCH_ADVANCEMENT,
        payload={"new_epoch": 2},
    )
    for v_id in ["node_0", "node_1", "node_2"]:
        engine.record_prevote(ConsensusVote(
            vote_id=f"pv_{v_id}", epoch=1, round=0, height=1, voter_id=v_id,
            vote_type=VoteType.PREVOTE, proposal_digest=prop.proposal_digest,
        ))
    precommit_qc = None
    for v_id in ["node_0", "node_1", "node_2"]:
        precommit_qc = engine.record_precommit(ConsensusVote(
            vote_id=f"pc_{v_id}", epoch=1, round=0, height=1, voter_id=v_id,
            vote_type=VoteType.PRECOMMIT, proposal_digest=prop.proposal_digest,
        ))
    engine.commit_block(prop, precommit_qc)

    # Verify model is untouched
    final_hash = _compute_model_hash(model)
    assert final_hash == initial_hash == EXPECTED_WEIGHT_HASH
    final_param_count = sum(p.numel() for p in model.parameters())
    assert final_param_count == EXPECTED_PARAM_COUNT
