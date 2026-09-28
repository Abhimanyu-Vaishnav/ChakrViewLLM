"""
Comprehensive Test Suite for Cognitive Identity, Self-Model & System State (Step 18).

Verifies:
1. SystemIdentity creation, immutability, parameter invariants.
2. Deterministic state updates and monotonic version increments.
3. Snapshot creation, serialization, deserialization, and deep immutability.
4. Non-destructive rollback semantics with audit history preservation.
5. Snapshot comparison and structured delta generation.
6. Strict semantic distinction: UNKNOWN != FALSE.
7. Strict semantic distinction: UNAVAILABLE != UNKNOWN.
8. Strict semantic distinction: DISABLED != UNAVAILABLE.
9. Knowledge assertion provenance and temporal validity.
10. Explicit uncertainty tracking and uncalibrated flags.
11. Task lifecycle transitions and observation logging.
12. EnvironmentState integration with EnvironmentProfile.
13. CapabilityState synchronization with CapabilityRegistry.
14. ConstraintState and policy restriction evaluation.
15. Multi-tenant privacy isolation (cross-owner access rejection).
16. Corruption-safe JSON loading and injection defense.
17. DATA != AUTHORITY principle enforcement.
18. CapabilityGate remains strictly authoritative over CapabilityState observations.
19. Frozen neural core model invariants.
"""

import json
import pytest
import time

from chakrview.state.identity import (
    SystemIdentity,
    get_current_system_identity,
)
from chakrview.state.epistemic import (
    EpistemicStatus,
    KnowledgeAssertion,
    KnowledgeState,
)
from chakrview.state.uncertainty import (
    Uncertainty,
    UncertaintyState,
)
from chakrview.state.task_state import (
    TaskPhase,
    TaskState,
)
from chakrview.state.environment_state import (
    OperationalMode,
    DeviceConnectionStatus,
    EnvironmentState,
)
from chakrview.state.capability_state import (
    ObservedCapabilityStatus,
    CapabilityObservation,
    CapabilityState,
)
from chakrview.state.constraints import (
    PolicyRestriction,
    ConstraintState,
)
from chakrview.state.snapshot import (
    SnapshotMetadata,
    CognitiveStateSnapshot,
    SnapshotDiff,
    compare_snapshots,
)
from chakrview.state.manager import (
    CognitiveStateManager,
    StateIsolationError,
    SnapshotNotFoundError,
)
from chakrview.capability.environment import get_edge_environment, get_desktop_environment
from chakrview.capability.bridge import get_standard_capability_registry
from chakrview.capability.gate import CapabilityGate, CapabilityAuthorizationError
from chakrview.capability.contract import CapabilityRequest


# =====================================================================
# 1. System Identity Tests
# =====================================================================

def test_system_identity_invariants_and_immutability():
    ident = get_current_system_identity()
    assert ident.model_parameters == 3443136
    assert ident.tokenizer_vocab == 4096
    assert ident.context_length == 512
    assert ident.architecture_version == "Step 18"
    assert "calculator" in ident.supported_capabilities

    # Verify frozen immutability
    with pytest.raises(Exception):
        ident.model_parameters = 10000000  # type: ignore

    # Verify boundary checks
    with pytest.raises(ValueError):
        SystemIdentity(system_id="bad", model_parameters=5000000)


def test_system_identity_serialization():
    ident = get_current_system_identity(system_id="node_test_01")
    d = ident.to_dict()
    assert d["system_id"] == "node_test_01"
    assert d["model_parameters"] == 3443136

    restored = SystemIdentity.from_dict(d)
    assert restored.system_id == ident.system_id
    assert restored.model_parameters == ident.model_parameters


# =====================================================================
# 2. Semantic Distinctions: UNKNOWN vs FALSE vs UNAVAILABLE vs DISABLED
# =====================================================================

def test_unknown_vs_false_distinction():
    """
    CRITICAL RULE: UNKNOWN != FALSE.
    Acknowledging that a fact is unknown must not be conflated with the assertion being false.
    """
    state = KnowledgeState()

    # Fact explicitly known to be false
    assert_false = state.assert_fact(
        subject="client_x",
        predicate="is_registered_for_gst",
        value=False,
        status=EpistemicStatus.KNOWN,
    )
    assert assert_false.status == EpistemicStatus.KNOWN
    assert assert_false.value is False

    # Fact explicitly unknown
    assert_unknown = state.assert_fact(
        subject="client_x",
        predicate="has_overseas_subsidiary",
        value=None,
        status=EpistemicStatus.UNKNOWN,
    )
    assert assert_unknown.status == EpistemicStatus.UNKNOWN
    assert assert_unknown.value is not False
    assert assert_unknown.status != assert_false.status


def test_unavailable_vs_unknown_distinction():
    """
    CRITICAL RULE: UNAVAILABLE != UNKNOWN.
    A sensor or subsystem being unreachable is distinct from having zero knowledge of the concept.
    """
    state = KnowledgeState()

    # Telemetry unavailable because hardware sensor is disconnected
    assert_unavail = state.assert_fact(
        subject="environment",
        predicate="ambient_temperature_celsius",
        value=None,
        status=EpistemicStatus.UNAVAILABLE,
        source="sensor_probe_01",
    )
    assert assert_unavail.status == EpistemicStatus.UNAVAILABLE
    assert assert_unavail.status != EpistemicStatus.UNKNOWN


def test_disabled_vs_unavailable_distinction():
    """
    CRITICAL RULE: DISABLED != UNAVAILABLE.
    Policy/administrative shutdown is distinct from physical failure/unavailability.
    """
    cap_state = CapabilityState()

    # Capability administratively shut down by policy
    cap_state.record_status(
        capability_id="mock_motor_actuator",
        status=ObservedCapabilityStatus.DISABLED,
        health_notes="Disabled by safety administrator",
    )
    # Capability whose hardware interface is missing
    cap_state.record_status(
        capability_id="thermal_camera",
        status=ObservedCapabilityStatus.UNAVAILABLE,
        health_notes="Hardware device not found on bus",
    )

    st_disabled = cap_state.get_status("mock_motor_actuator")
    st_unavail = cap_state.get_status("thermal_camera")
    st_unknown = cap_state.get_status("unprobed_capability")

    assert st_disabled == ObservedCapabilityStatus.DISABLED
    assert st_unavail == ObservedCapabilityStatus.UNAVAILABLE
    assert st_unknown == ObservedCapabilityStatus.UNKNOWN
    assert st_disabled != st_unavail
    assert st_unavail != st_unknown


# =====================================================================
# 3. Knowledge State & Epistemic Qualification Tests
# =====================================================================

def test_knowledge_assertion_validity_and_temporal_tracking():
    state = KnowledgeState()
    now = time.time()

    # Valid fact with future expiration
    a1 = state.assert_fact(
        subject="tax_rate",
        predicate="standard_gst",
        value=0.18,
        status=EpistemicStatus.KNOWN,
        valid_until=now + 1000.0,
    )
    assert a1.is_valid_at(now) is True

    # Expired fact
    a2 = state.assert_fact(
        subject="temporary_permit",
        predicate="status",
        value="ACTIVE",
        status=EpistemicStatus.KNOWN,
        valid_until=now - 50.0,
    )
    assert a2.is_valid_at(now) is False

    # Mark conflicting
    state.mark_conflicting([a1.assertion_id])
    assert state.get_assertion(a1.assertion_id).status == EpistemicStatus.CONFLICTING

    # Mark stale
    state.mark_stale(a1.assertion_id)
    assert state.get_assertion(a1.assertion_id).status == EpistemicStatus.STALE
    assert a1.is_valid_at(now) is False


def test_knowledge_querying_by_subject_and_predicate():
    state = KnowledgeState()
    state.assert_fact("company_a", "revenue_lakh", 50.0)
    state.assert_fact("company_a", "employee_count", 12)
    state.assert_fact("company_b", "revenue_lakh", 120.0)

    res_a = state.query(subject="company_a")
    assert len(res_a) == 2

    res_rev = state.query(predicate="revenue_lakh")
    assert len(res_rev) == 2


# =====================================================================
# 4. Uncertainty State Tests
# =====================================================================

def test_uncertainty_state_recording():
    unc_state = UncertaintyState()

    unc = unc_state.record_uncertainty(
        key="company_valuation_estimate",
        confidence=0.65,
        reason="Limited historical data provided in documents",
        source="retrieval_step_3",
        evidence_refs=["doc_financials_2025"],
        is_uncalibrated=True,
    )
    assert unc.confidence == 0.65
    assert unc.is_uncalibrated is True
    assert unc_state.has_uncertainty("company_valuation_estimate") is True

    cleared = unc_state.clear_uncertainty("company_valuation_estimate")
    assert cleared is unc
    assert unc_state.has_uncertainty("company_valuation_estimate") is False


# =====================================================================
# 5. Task State Lifecycle Tests
# =====================================================================

def test_task_state_lifecycle():
    task = TaskState(
        task_id="task_audit_101",
        goal="Audit client GST records for Q3",
        status="PLANNING",
    )
    assert task.phase == TaskPhase.IDLE

    task.transition_phase(TaskPhase.PLANNING, note="Rule planner selected")
    assert task.phase == TaskPhase.PLANNING

    task.record_decision(
        rationale="Client operating in multiple states requires interstate rule validation",
        chosen_action="dispatch_gst_skill",
    )
    assert len(task.decisions) == 1

    task.record_observation(
        step_id="step_1",
        output={"records_found": 45},
        success=True,
        elapsed_ms=12.5,
    )
    assert len(task.observations) == 1

    task.mark_completed(
        summary="All records reconciled without discrepancy",
        output_references={"reconciled_count": 45},
    )
    assert task.status == "COMPLETED"
    assert task.phase == TaskPhase.COMPLETED
    assert task.completion_state["outputs"]["reconciled_count"] == 45


# =====================================================================
# 6. Environment & Capability State Synchronization
# =====================================================================

def test_environment_state_sync_with_profile():
    env_state = EnvironmentState()
    edge_profile = get_edge_environment()

    env_state.sync_from_profile(edge_profile)
    assert env_state.environment_id == "env_edge_arm64"
    assert env_state.architecture == "ARM64"
    assert env_state.available_resources["memory_limit_mb"] == 256
    assert env_state.network_status == DeviceConnectionStatus.DISCONNECTED

    # Operational mode and safety interlocks
    env_state.set_operational_mode(OperationalMode.EMERGENCY_STOP, reason="Interlock tripped")
    assert env_state.operational_mode == OperationalMode.EMERGENCY_STOP

    env_state.set_safety_interlock("e_stop_switch", True)
    assert env_state.safety_interlock_status["e_stop_switch"] is True


def test_capability_state_sync_with_registry():
    cap_state = CapabilityState()
    reg = get_standard_capability_registry()
    desktop_env = get_desktop_environment()

    cap_state.sync_with_registry(registry=reg, policy=desktop_env)

    assert cap_state.get_status("calculator") == ObservedCapabilityStatus.AVAILABLE
    assert cap_state.get_status("system_clock") == ObservedCapabilityStatus.AVAILABLE
    assert cap_state.is_available("calculator") is True
    assert cap_state.capabilities["calculator"].is_authorized is True

    # Check unprobed capability
    assert cap_state.get_status("non_existent_cap") == ObservedCapabilityStatus.UNKNOWN


# =====================================================================
# 7. Constraint State & Policy Restriction Tests
# =====================================================================

def test_constraint_state_enforcement():
    constraints = ConstraintState(
        max_memory_mb=512,
        allowed_capabilities={"calculator", "system_clock"},
        blocked_capabilities={"mock_motor_actuator"},
        allowed_risk_levels={"READ_ONLY", "COMPUTE"},
    )

    # Allowed capability
    ok, err = constraints.is_capability_permitted("calculator", "COMPUTE")
    assert ok is True
    assert err is None

    # Blocked capability
    ok, err = constraints.is_capability_permitted("mock_motor_actuator", "PHYSICAL_ACTION")
    assert ok is False
    assert "blocked_capabilities" in err

    # Capability not in allowed whitelist
    ok, err = constraints.is_capability_permitted("text_transform", "COMPUTE")
    assert ok is False
    assert "not in allowed_capabilities" in err

    # Risk level disallowed
    ok, err = constraints.is_capability_permitted("calculator", "HIGH_IMPACT")
    assert ok is False
    assert "not in permitted risk levels" in err


# =====================================================================
# 8. Snapshot, Serialization & Immutability Tests
# =====================================================================

def test_snapshot_creation_serialization_and_immutability():
    mgr = CognitiveStateManager(owner_id="user_alpha", session_id="sess_01")
    mgr.assert_knowledge("entity_x", "type", "organization")
    mgr.update_task(TaskState(task_id="t1", goal="Initial task"))

    snap = mgr.snapshot(description="Baseline snapshot")
    assert snap.state_version == mgr.state_version
    assert snap.owner_id == "user_alpha"
    assert "t1" in snap.tasks

    # Verify JSON serialization round-trip
    json_repr = snap.to_json()
    assert isinstance(json_repr, str)

    restored = CognitiveStateSnapshot.from_json(json_repr)
    assert restored.snapshot_id == snap.snapshot_id
    assert restored.state_version == snap.state_version
    assert restored.owner_id == snap.owner_id
    assert restored.identity.model_parameters == 3443136
    assert "t1" in restored.tasks

    # Verify Snapshot Immutability: Mutating manager state must NOT mutate snap
    mgr.assert_knowledge("entity_y", "type", "corporation")
    mgr.update_task(TaskState(task_id="t2", goal="Second task"))
    assert "t2" not in snap.tasks
    assert len(snap.knowledge.assertions) == 1
    assert len(mgr.knowledge.assertions) == 2


def test_snapshot_comparison():
    mgr = CognitiveStateManager(owner_id="user_alpha")
    mgr.update_task(TaskState(task_id="t1", goal="Analyze data", phase=TaskPhase.PLANNING))
    mgr.assert_knowledge("client", "status", "active")
    snap1 = mgr.snapshot(description="Step 1")

    # Mutate state
    mgr.tasks["t1"].transition_phase(TaskPhase.COMPLETED)
    mgr.assert_knowledge("client", "status", "pending_review")
    mgr.record_uncertainty("client_credit_score", confidence=0.7, reason="Pending update", source="credit_bureau")
    snap2 = mgr.snapshot(description="Step 2")

    diff = compare_snapshots(snap1, snap2)
    assert diff.has_differences is True
    assert diff.base_version == snap1.state_version
    assert diff.target_version == snap2.state_version
    assert "t1" in diff.task_changes["modified_tasks"]
    assert "client_credit_score" in diff.uncertainty_changes["added_uncertainties"]


# =====================================================================
# 9. Rollback & Audit History Preservation Tests
# =====================================================================

def test_rollback_semantics_and_audit_preservation():
    mgr = CognitiveStateManager(owner_id="user_alpha")
    mgr.assert_knowledge("target_val", "score", 100)
    snap_good = mgr.snapshot(description="Good verified state")

    # Introduce corrupted/bad state
    mgr.assert_knowledge("target_val", "score", 9999)
    mgr.update_task(TaskState(task_id="err_task", goal="Bad task", status="FAILED"))
    snap_bad = mgr.snapshot(description="Faulty state")

    version_before_rollback = mgr.state_version
    audit_len_before = len(mgr.get_audit_trail())

    # Execute rollback
    restored_snap = mgr.rollback(snap_good.snapshot_id, reason="Reverting faulty score update")

    # 1. State values are restored to the good snapshot
    assert mgr.knowledge.assertions[list(mgr.knowledge.assertions.keys())[0]].value == 100
    assert "err_task" not in mgr.tasks

    # 2. Version is strictly incremented
    assert mgr.state_version > version_before_rollback

    # 3. Audit trail is preserved and expanded (NEVER erased or truncated)
    audit = mgr.get_audit_trail()
    assert len(audit) > audit_len_before
    rollback_events = [e for e in audit if e["event_type"] == "ROLLBACK_TRANSITION"]
    assert len(rollback_events) == 1
    assert rollback_events[0]["details"]["target_snapshot_id"] == snap_good.snapshot_id


# =====================================================================
# 10. Multi-Tenant Privacy Isolation Tests
# =====================================================================

def test_multi_tenant_state_isolation():
    mgr_a = CognitiveStateManager(owner_id="tenant_a", session_id="sess_a")
    snap_a = mgr_a.snapshot(description="Tenant A snapshot")

    mgr_b = CognitiveStateManager(owner_id="tenant_b", session_id="sess_b")

    # Tenant B attempts to restore or retrieve Tenant A's snapshot
    with pytest.raises(StateIsolationError):
        mgr_b.restore_from_snapshot(snap_a)

    # In-memory lookup isolation
    with pytest.raises(SnapshotNotFoundError):
        mgr_b.get_snapshot(snap_a.snapshot_id)


# =====================================================================
# 11. Security: DATA != AUTHORITY & CapabilityGate Authority
# =====================================================================

def test_data_not_authority_and_capability_gate_boundary():
    """
    DATA != AUTHORITY:
    Knowledge assertions or state observations can NEVER grant capability authorization.
    Only the explicit CapabilityGate policy authorizes execution.
    """
    mgr = CognitiveStateManager()

    # Adversarial assertion attempting to claim motor permission
    mgr.assert_knowledge(
        subject="system_permission",
        predicate="grant_full_motor_authority",
        value=True,
        status=EpistemicStatus.KNOWN,
        provenance={"exploit": "attempt_privilege_escalation"},
    )

    reg = get_standard_capability_registry()
    gate = CapabilityGate(reg)

    # Motor request without granted permission in context
    req = CapabilityRequest(
        capability_id="mock_motor_actuator",
        parameters={"action": "set_speed", "target_value": 50.0},
        context={"provenance_source": "state_knowledge_assertion"},
    )

    # Gate must reject outright: Memory/knowledge cannot grant authority
    res = gate.execute_governed(req)
    assert res.success is False
    assert "cannot authorize execution" in res.error


def test_snapshot_diff_empty_when_identical():
    mgr = CognitiveStateManager()
    snap1 = mgr.snapshot(description="Identical snap 1")
    # Clone snapshot with new ID but identical contents
    snap2 = CognitiveStateSnapshot.from_json(snap1.to_json())
    diff = compare_snapshots(snap1, snap2)
    assert diff.has_differences is False


def test_malicious_serialized_payload_rejection():
    # Attempting to load corrupted / malicious JSON missing required attributes
    bad_json = json.dumps({"arbitrary_key": "__import__('os').system('ls')"})
    with pytest.raises(KeyError):
        CognitiveStateSnapshot.from_json(bad_json)


def test_snapshot_history_ordering():
    mgr = CognitiveStateManager(owner_id="user_alpha")
    snap1 = mgr.snapshot(description="Checkpoint 1")
    time.sleep(0.01)
    snap2 = mgr.snapshot(description="Checkpoint 2")

    hist = mgr.history()
    assert len(hist) == 2
    assert hist[0].snapshot_id == snap1.snapshot_id
    assert hist[1].snapshot_id == snap2.snapshot_id
    assert hist[0].timestamp <= hist[1].timestamp


def test_knowledge_query_by_status():
    state = KnowledgeState()
    state.assert_fact("s1", "p1", 10, status=EpistemicStatus.KNOWN)
    state.assert_fact("s2", "p2", None, status=EpistemicStatus.UNKNOWN)
    state.assert_fact("s3", "p3", None, status=EpistemicStatus.UNAVAILABLE)

    knowns = state.query(status=EpistemicStatus.KNOWN)
    unknowns = state.query(status=EpistemicStatus.UNKNOWN)
    unavails = state.query(status=EpistemicStatus.UNAVAILABLE)

    assert len(knowns) == 1
    assert len(unknowns) == 1
    assert len(unavails) == 1
    assert knowns[0].subject == "s1"
    assert unknowns[0].subject == "s2"
    assert unavails[0].subject == "s3"


# =====================================================================
# 12. Frozen Neural Core Invariant Checks
# =====================================================================

def test_frozen_neural_core_invariants():
    from chakrview.brain.model import ChakrMicro
    from chakrview.brain.config import ModelConfig
    from chakrview.tokenizer import BOS_ID, EOS_ID, PAD_ID

    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    total_params = sum(p.numel() for p in model.parameters())

    assert total_params == 3443136, f"Frozen parameter invariant broken: {total_params}"
    assert cfg.vocab_size == 4096, f"Frozen vocab invariant broken: {cfg.vocab_size}"
    assert cfg.max_seq_len == 512, f"Frozen context invariant broken: {cfg.max_seq_len}"
    assert BOS_ID == 0, f"Frozen BOS invariant broken: {BOS_ID}"
    assert EOS_ID == 1, f"Frozen EOS invariant broken: {EOS_ID}"
    assert PAD_ID == 2, f"Frozen PAD invariant broken: {PAD_ID}"
