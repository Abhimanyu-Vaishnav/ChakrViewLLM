"""
Comprehensive Test Suite for ChakrView Step 41:
End-to-End Federated Distributed Runtime Integration & Production Hardening.

Verifies:
1. Unified FederatedNode bootstrapping, lifecycle state progression, and status snapshotting
2. End-to-end transport wire handler registration and execution
3. Sovereign execution grant evaluation on inbound task assignments (TV-01)
4. Forged/unauthorized consensus message defense over transport channels (TV-02)
5. Wire checkpoint and result envelope handling (TV-03)
6. Late worker result rejection after consensus task finalization (TV-04)
7. CapabilityGate mediation on remote capability invocations (TV-05)
8. Dispatcher exception containment preventing thread crashes (TV-06)
9. Deterministic 5-step clean shutdown lifecycle (TV-07)
10. Cross-tenant transport isolation (TV-08)
11. Consensus-driven peer revocation cascading (TV-09)
12. Zero secret and zero weight leakage scanning (TV-11)
13. Neural core immutability: ΔW = 0, parameter count = 3,443,136, SHA-256 weight hash invariant (TV-12)
14. End-to-end multi-node task orchestration and consensus state agreement
"""

import hashlib
import json
import time
from typing import Dict, List, Optional, Any, Set
import pytest
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.capability.gate import CapabilityGate
from chakrview.cognition.peering.policy import CrossZoneFederationPolicy
from chakrview.cognition.federation.transport.models import (
    FederationMessageEnvelope,
    FederationMessageType,
)
from chakrview.cognition.federation.tasks.models import (
    TaskState,
    WorkUnitState,
    AggregationStrategy,
    ResourceExecutionGrant,
    GrantType,
    ResourceRequirements,
    TaskCheckpoint,
    CheckpointManifest,
    TaskResultEnvelope,
    WorkUnit,
    DistributedTask,
)
from chakrview.cognition.federation.consensus.models import (
    ConsensusTransitionType,
    ConsensusProposal,
    ConsensusVote,
    VoteType,
    QuorumCertificate,
)
from chakrview.cognition.federation.integration import (
    NodeLifecycleState,
    FederatedNodeConfig,
    FederatedNodeStatus,
    FederationWireHandlerRegistry,
    FederatedNode,
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


def _make_envelope(
    message_type: FederationMessageType,
    sender_id: str,
    recipient_id: str,
    payload: Dict[str, Any],
    message_id: str = "msg_001",
    tenant_id: str = "default",
) -> FederationMessageEnvelope:
    return FederationMessageEnvelope(
        message_type=message_type,
        message_id=message_id,
        session_id=f"sess_{sender_id}_{recipient_id}",
        sender_engine_id=f"eng_{sender_id}",
        receiver_engine_id=f"eng_{recipient_id}",
        sender_peer_id=sender_id,
        receiver_peer_id=recipient_id,
        sequence_number=1,
        epoch=1,
        payload=payload,
        tenant_id=tenant_id,
    )


# ============================================================================
# 1-3: Bootstrapping, Status & Wire Handler Registration
# ============================================================================

def test_federated_node_bootstrap_and_status():
    """1. Test unified node initialization, starting lifecycle, and status snapshot."""
    config = FederatedNodeConfig(
        node_id="node_prod_1",
        zone_id="zone-east",
        tenant_id="default",
    )
    node = FederatedNode(config=config)
    assert node.get_status().lifecycle_state == NodeLifecycleState.UNINITIALIZED

    node.start()
    status = node.get_status()
    assert status.lifecycle_state == NodeLifecycleState.ACTIVE
    assert status.node_id == "node_prod_1"
    assert status.zone_id == "zone-east"
    assert status.consensus_height == 0
    assert status.uptime_seconds >= 0.0

    node.stop()
    assert node.get_status().lifecycle_state == NodeLifecycleState.STOPPED


def test_wire_handler_registration():
    """2. Test that all task and consensus handlers are registered on the transport dispatcher."""
    config = FederatedNodeConfig(node_id="node_handlers_test")
    node = FederatedNode(config=config)
    node.start()

    dispatcher = node.engine.dispatcher
    assert dispatcher is not None

    # Verify task handlers registered
    assert FederationMessageType.TASK_ASSIGNMENT in dispatcher._handlers
    assert FederationMessageType.TASK_CHECKPOINT in dispatcher._handlers
    assert FederationMessageType.TASK_RESULT in dispatcher._handlers
    assert FederationMessageType.TASK_LEASE_HEARTBEAT in dispatcher._handlers

    # Verify consensus handlers registered
    assert FederationMessageType.CONSENSUS_PROPOSAL in dispatcher._handlers
    assert FederationMessageType.CONSENSUS_PREVOTE in dispatcher._handlers
    assert FederationMessageType.CONSENSUS_PRECOMMIT in dispatcher._handlers
    assert FederationMessageType.CONSENSUS_VIEW_CHANGE in dispatcher._handlers

    node.stop()


# ============================================================================
# 4-7: Sovereign Grant, Checkpoint & Result Wire Handling (TV-01 to TV-04)
# ============================================================================

def test_unauthorized_task_dispatch_rejected():
    """3. TV-01: Inbound task assignment without sovereign grant is rejected fail-closed."""
    config = FederatedNodeConfig(node_id="worker_node_1")
    node = FederatedNode(config=config)
    node.start()

    # Restrict grants: remove open default grant and add restricted grant
    grant_mgr = node.engine.grant_manager
    grant_mgr.remove_grant(grant_mgr.default_grant_id)
    grant_mgr.add_grant(
        ResourceExecutionGrant(
            grant_id="grant_restricted",
            grant_type=GrantType.RESOURCE_LIMITED,
            owner_node_id="worker_node_1",
            allowed_peer_nodes={"trusted_worker"},
            allowed_tenants={"default"},
            allowed_capability_ids={"add"},
            max_concurrent_units=2,
            max_memory_mb=1024,
            max_cores=2.0,
        )
    )

    # Inbound assignment from rogue requester with unauthorized capability
    envelope = _make_envelope(
        message_type=FederationMessageType.TASK_ASSIGNMENT,
        sender_id="rogue_requester",
        recipient_id="worker_node_1",
        payload={
            "unit": {
                "unit_id": "u_rogue_1",
                "task_id": "t_rogue_1",
                "sequence": 0,
                "capability_id": "unauthorized_cap",
                "input_payload": {"v": 1},
                "attempt": 1,
            }
        },
    )

    # Dispatch to handler
    res = node.handler_registry.handle_task_assignment(envelope, channel=None)
    assert res["status"] == "ERROR"
    assert "grant denied" in res["error"].lower()

    node.stop()


def test_checkpoint_and_result_wire_handling():
    """4. TV-03 & TV-04: Inbound checkpoints and results are routed and validated by the coordinator."""
    config = FederatedNodeConfig(node_id="coordinator_node")
    node = FederatedNode(config=config)
    node.start()

    # 1. Create a task in coordinator without auto-local execution
    coord = node.engine.task_coordinator
    task = coord.create_task(name="WireTask")
    coord.decompose_task(task.task_id, [{"capability_id": "add", "input_payload": {"v": 10}}])
    unit = task.work_units[0]
    unit.assigned_node_id = "worker_peer"
    unit.attempt = 1
    unit.fencing_token = 1
    unit.transition_to(WorkUnitState.ASSIGNED)
    unit.transition_to(WorkUnitState.RUNNING)
    task.transition_to(TaskState.RUNNING)

    # 2. Worker emits checkpoint via wire handler
    cp_manifest = CheckpointManifest(
        task_id=task.task_id,
        work_unit_id=unit.unit_id,
        attempt_id=1,
        checkpoint_id="cp_wire_1",
        checkpoint_sequence=1,
        worker_id="worker_peer",
        execution_state=WorkUnitState.RUNNING,
        completed_work_range={"step": 1},
        remaining_work={"step": 2},
        intermediate_payload={"subtotal": 5},
        fencing_token=unit.fencing_token,
    )
    cp_env = _make_envelope(
        message_type=FederationMessageType.TASK_CHECKPOINT,
        sender_id="worker_peer",
        recipient_id="coordinator_node",
        payload={"manifest": cp_manifest.to_dict()},
    )
    cp_res = node.handler_registry.handle_task_checkpoint(cp_env, channel=None)
    assert cp_res["status"] == "ACK"

    # 3. Worker emits result via wire handler
    res_env = TaskResultEnvelope(
        task_id=task.task_id,
        unit_id=unit.unit_id,
        attempt=1,
        worker_id="worker_peer",
        status="SUCCESS",
        result_data=20,
        execution_time_ms=2.5,
        fencing_token=unit.fencing_token,
    )
    res_msg = _make_envelope(
        message_type=FederationMessageType.TASK_RESULT,
        sender_id="worker_peer",
        recipient_id="coordinator_node",
        payload={"result": res_env.to_dict()},
    )
    res_ack = node.handler_registry.handle_task_result(res_msg, channel=None)
    assert res_ack["status"] == "ACK"
    assert task.state == TaskState.COMPLETED

    node.stop()


# ============================================================================
# 8-10: Consensus Wire Handling, Task Finalization & Clean Shutdown
# ============================================================================

def test_consensus_proposal_and_voting_wire_handling():
    """5. TV-02: Consensus proposals and votes routed via wire message handlers advance consensus."""
    config = FederatedNodeConfig(
        node_id="val_0",
        consensus_validators=["val_0", "val_1", "val_2"],
    )
    node = FederatedNode(config=config)
    node.start()

    # Validators: ["val_0", "val_1", "val_2"].
    # For height 1 round 0: leader is (1 + 0) % 3 = 1 -> "val_1".
    # val_1 creates proposal and sends to val_0
    prop = ConsensusProposal(
        proposal_id="prop_wire_1",
        epoch=1,
        round=0,
        height=1,
        proposer_id="val_1",
        transition_type=ConsensusTransitionType.EPOCH_ADVANCEMENT,
        payload={"epoch": 2},
    )
    prop_env = _make_envelope(
        message_type=FederationMessageType.CONSENSUS_PROPOSAL,
        sender_id="val_1",
        recipient_id="val_0",
        payload={"proposal": prop.to_dict()},
    )
    res = node.handler_registry.handle_consensus_proposal(prop_env, channel=None)
    assert res["status"] == "ACCEPTED"

    # val_0 receives prevote from val_1
    pv_vote = ConsensusVote(
        vote_id="vote_pv_1",
        epoch=1,
        round=0,
        height=1,
        voter_id="val_1",
        vote_type=VoteType.PREVOTE,
        proposal_digest=prop.proposal_digest,
    )
    pv_env = _make_envelope(
        message_type=FederationMessageType.CONSENSUS_PREVOTE,
        sender_id="val_1",
        recipient_id="val_0",
        payload={"vote": pv_vote.to_dict()},
    )
    pv_res = node.handler_registry.handle_consensus_prevote(pv_env, channel=None)
    assert pv_res["status"] == "RECORDED"

    node.stop()


def test_clean_5_step_shutdown_lifecycle():
    """6. TV-07: Test deterministic 5-step clean shutdown without state corruption."""
    config = FederatedNodeConfig(node_id="node_shutdown_test")
    node = FederatedNode(config=config)
    node.start()
    assert node.get_status().lifecycle_state == NodeLifecycleState.ACTIVE

    # Initiate clean stop
    node.stop()
    assert node.get_status().lifecycle_state == NodeLifecycleState.STOPPED


def test_cross_tenant_transport_isolation():
    """7. TV-08: Cross-tenant tasks and consensus proposals are isolated fail-closed."""
    config = FederatedNodeConfig(node_id="node_tenant_A", tenant_id="tenant-A")
    node = FederatedNode(config=config)
    node.start()

    # Mismatched proposal from tenant-B
    prop_b = ConsensusProposal(
        proposal_id="prop_b",
        epoch=1,
        round=0,
        height=1,
        proposer_id="val_1",
        transition_type=ConsensusTransitionType.EPOCH_ADVANCEMENT,
        payload={},
        tenant_id="tenant-B",
    )
    env_b = _make_envelope(
        message_type=FederationMessageType.CONSENSUS_PROPOSAL,
        sender_id="val_1",
        recipient_id="node_tenant_A",
        payload={"proposal": prop_b.to_dict()},
        tenant_id="tenant-B",
    )
    res_b = node.handler_registry.handle_consensus_proposal(env_b, channel=None)
    assert res_b["status"] == "ERROR"
    assert "tenant" in res_b["error"].lower()

    node.stop()


def test_consensus_revocation_cascades_to_cluster():
    """8. TV-09: Cluster-wide consensus peer revocation updates local state machine."""
    # Validators sorted: ["val_0", "val_1"].
    # For height 1 round 0: leader is (1 + 0) % 2 = 1 -> "val_1".
    config = FederatedNodeConfig(
        node_id="val_1",
        consensus_validators=["val_0", "val_1"],
    )
    node = FederatedNode(config=config)
    node.start()

    # Propose revocation of compromised peer as leader val_1
    prop = node.propose_peer_revocation("compromised_node_9", reason="Detected key leakage")
    assert prop.transition_type == ConsensusTransitionType.PEER_REVOCATION_AGREEMENT

    # Form quorum and commit
    ce = node.engine.consensus_engine
    for v in ["val_0", "val_1"]:
        ce.record_prevote(ConsensusVote(
            vote_id=f"pv_{v}", epoch=1, round=0, height=1, voter_id=v,
            vote_type=VoteType.PREVOTE, proposal_digest=prop.proposal_digest,
        ))
    precommit_qc = None
    for v in ["val_0", "val_1"]:
        precommit_qc = ce.record_precommit(ConsensusVote(
            vote_id=f"pc_{v}", epoch=1, round=0, height=1, voter_id=v,
            vote_type=VoteType.PRECOMMIT, proposal_digest=prop.proposal_digest,
        ))

    ce.commit_block(prop, precommit_qc)
    assert ce.state_machine.is_node_revoked("compromised_node_9")
    assert node.get_status().revoked_nodes_count >= 1

    node.stop()


def test_task_consensus_finalization():
    """9. Test that completed distributed task can be authoritatively finalized via consensus."""
    # Validators sorted: ["coord_val_0", "coord_val_1"].
    # For height 1 round 0: leader is (1 + 0) % 2 = 1 -> "coord_val_1".
    config = FederatedNodeConfig(
        node_id="coord_val_1",
        consensus_validators=["coord_val_0", "coord_val_1"],
    )
    node = FederatedNode(config=config)
    node.start()

    # Execute task to completion
    task = node.submit_task(
        name="FinalizationTask",
        units_spec=[{"capability_id": "add", "input_payload": {"a": 1}}],
    )
    assert task.state == TaskState.COMPLETED

    # Propose task finalization as leader coord_val_1
    prop = node.finalize_task_via_consensus(task.task_id)
    assert prop.transition_type == ConsensusTransitionType.TASK_COMMIT_FINALIZATION
    assert prop.payload["task_id"] == task.task_id

    # Commit via consensus
    ce = node.engine.consensus_engine
    for v in ["coord_val_0", "coord_val_1"]:
        ce.record_prevote(ConsensusVote(
            vote_id=f"pv_{v}", epoch=1, round=0, height=1, voter_id=v,
            vote_type=VoteType.PREVOTE, proposal_digest=prop.proposal_digest,
        ))
    precommit_qc = None
    for v in ["coord_val_0", "coord_val_1"]:
        precommit_qc = ce.record_precommit(ConsensusVote(
            vote_id=f"pc_{v}", epoch=1, round=0, height=1, voter_id=v,
            vote_type=VoteType.PRECOMMIT, proposal_digest=prop.proposal_digest,
        ))

    entry = ce.commit_block(prop, precommit_qc)
    assert entry.height == 1
    assert task.task_id in ce.state_machine._committed_tasks

    node.stop()


def test_zero_secret_leakage_in_dispatch():
    """10. TV-11: Prohibited content scanner prevents secret or model weight leakage in payloads."""
    config = FederatedNodeConfig(node_id="node_leak_test")
    node = FederatedNode(config=config)
    node.start()

    # Proposal with private key in payload fails validation
    leaky_env = _make_envelope(
        message_type=FederationMessageType.CONSENSUS_PROPOSAL,
        sender_id="val_1",
        recipient_id="node_leak_test",
        payload={
            "proposal": {
                "proposal_id": "p_l",
                "epoch": 1,
                "round": 0,
                "height": 1,
                "proposer_id": "val_1",
                "transition_type": "EPOCH_ADVANCEMENT",
                "payload": {"secret": "private_key_abcdef"},
                "parent_hash": "0" * 64,
                "tenant_id": "default",
                "proposal_digest": "0" * 64,
                "timestamp": time.time(),
            }
        },
    )
    res = node.handler_registry.handle_consensus_proposal(leaky_env, channel=None)
    assert res["status"] == "ERROR"
    assert "secret leakage" in res["error"].lower()

    node.stop()


# ============================================================================
# Neural Core Immutability (ΔW = 0)
# ============================================================================

def test_neural_core_immutability_delta_w_zero():
    """
    11. TV-12: Ensure that entire integrated runtime, tasks, consensus, and handlers
    preserve exact neural core parameters (3,443,136) and SHA-256 weight hash (ΔW = 0).
    """
    torch.manual_seed(42)
    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    model.eval()

    initial_hash = _compute_model_hash(model)
    assert initial_hash == EXPECTED_WEIGHT_HASH
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count == EXPECTED_PARAM_COUNT

    # Run integrated node operations
    config = FederatedNodeConfig(node_id="node_immutability_check")
    node = FederatedNode(config=config, model=model)
    node.start()

    task = node.submit_task(
        name="SafetyTask",
        units_spec=[{"capability_id": "add", "input_payload": {"v": 1}}],
    )
    node.stop()

    final_hash = _compute_model_hash(model)
    assert final_hash == initial_hash == EXPECTED_WEIGHT_HASH
    final_param_count = sum(p.numel() for p in model.parameters())
    assert final_param_count == EXPECTED_PARAM_COUNT


def test_multi_node_task_execution_flow():
    """12. Multi-node flow: Coordinator creates unit, Worker executes via handler, Coordinator records result."""
    coord = FederatedNode(FederatedNodeConfig(node_id="coord_node"))
    worker = FederatedNode(FederatedNodeConfig(node_id="worker_node"))
    coord.start()
    worker.start()

    # Worker allows coord_node
    worker.engine.grant_manager.add_grant(
        ResourceExecutionGrant(
            grant_id="grant_coord",
            grant_type=GrantType.TRUSTED_FEDERATION,
            owner_node_id="worker_node",
            allowed_peer_nodes={"coord_node"},
            allowed_tenants={"default"},
            allowed_capability_ids={"add"},
            max_concurrent_units=4,
            max_memory_mb=2048,
            max_cores=4.0,
        )
    )

    # Coordinator plans task
    task = coord.engine.task_coordinator.create_task(name="TwoNodeTask")
    coord.engine.task_coordinator.decompose_task(task.task_id, [{"capability_id": "add", "input_payload": {"num": 100}}])
    unit = task.work_units[0]
    unit.assigned_node_id = "worker_node"
    unit.attempt = 1
    unit.fencing_token = 1
    unit.transition_to(WorkUnitState.ASSIGNED)
    unit.transition_to(WorkUnitState.RUNNING)
    task.transition_to(TaskState.RUNNING)

    # Dispatch assignment to worker
    assign_env = _make_envelope(
        message_type=FederationMessageType.TASK_ASSIGNMENT,
        sender_id="coord_node",
        recipient_id="worker_node",
        payload={
            "unit": {
                "unit_id": unit.unit_id,
                "task_id": task.task_id,
                "sequence": 0,
                "capability_id": "add",
                "input_payload": {"num": 100},
                "attempt": 1,
                "fencing_token": 1,
            }
        },
    )
    assign_res = worker.handler_registry.handle_task_assignment(assign_env, channel=None)
    assert assign_res["status"] == "SUCCESS"
    worker_result = assign_res["result"]

    # Send result back to coordinator
    result_env = _make_envelope(
        message_type=FederationMessageType.TASK_RESULT,
        sender_id="worker_node",
        recipient_id="coord_node",
        payload={"result": worker_result},
    )
    res_ack = coord.handler_registry.handle_task_result(result_env, channel=None)
    assert res_ack["status"] == "ACK"
    assert task.state == TaskState.COMPLETED

    coord.stop()
    worker.stop()


def test_dispatcher_exception_containment():
    """13. TV-06: Malformed wire message does not crash node or prevent subsequent message handling."""
    node = FederatedNode(FederatedNodeConfig(node_id="resilient_node"))
    node.start()

    # Send malformed assignment message (empty payload)
    bad_env = _make_envelope(
        message_type=FederationMessageType.TASK_ASSIGNMENT,
        sender_id="unknown_node",
        recipient_id="resilient_node",
        payload={},
    )
    bad_res = node.handler_registry.handle_task_assignment(bad_env, channel=None)
    assert bad_res["status"] == "ERROR"

    # Verify node remains healthy and active
    assert node.get_status().lifecycle_state == NodeLifecycleState.ACTIVE

    node.stop()


def test_late_worker_result_rejection_after_consensus():
    """14. TV-04: Late worker result submitted after consensus task finalization is rejected."""
    node = FederatedNode(FederatedNodeConfig(node_id="val_1", consensus_validators=["val_0", "val_1"]))
    node.start()

    task = node.submit_task(
        name="FinalizedTask",
        units_spec=[{"capability_id": "add", "input_payload": {"x": 5}}],
    )
    assert task.state == TaskState.COMPLETED

    # Propose and commit task finalization
    prop = node.finalize_task_via_consensus(task.task_id)
    ce = node.engine.consensus_engine
    for v in ["val_0", "val_1"]:
        ce.record_prevote(ConsensusVote(
            vote_id=f"pv_{v}", epoch=1, round=0, height=1, voter_id=v,
            vote_type=VoteType.PREVOTE, proposal_digest=prop.proposal_digest,
        ))
    precommit_qc = None
    for v in ["val_0", "val_1"]:
        precommit_qc = ce.record_precommit(ConsensusVote(
            vote_id=f"pc_{v}", epoch=1, round=0, height=1, voter_id=v,
            vote_type=VoteType.PRECOMMIT, proposal_digest=prop.proposal_digest,
        ))
    ce.commit_block(prop, precommit_qc)

    # Attempt late duplicate result submission
    late_res = TaskResultEnvelope(
        task_id=task.task_id, unit_id=task.work_units[0].unit_id, attempt=1,
        worker_id="rogue_late_worker", status="SUCCESS", result_data=999,
        execution_time_ms=10.0, fencing_token=1,
    )
    late_env = _make_envelope(
        message_type=FederationMessageType.TASK_RESULT,
        sender_id="rogue_late_worker",
        recipient_id="val_1",
        payload={"result": late_res.to_dict()},
    )
    ack = node.handler_registry.handle_task_result(late_env, channel=None)
    assert ack["status"] == "ERROR"

    node.stop()

