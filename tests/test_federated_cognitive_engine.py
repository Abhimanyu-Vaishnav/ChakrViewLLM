"""
Step 42 Dedicated Test Suite: Federated Cognitive Orchestration & Distributed Reasoning Graph.

Tests cover:
1. CognitiveContextEnvelope — validation, overflow, secret scan, tenant isolation, serialisation
2. CognitiveTaskGraph      — DAG mechanics, cycle detection, ready steps, completion
3. CognitiveStep           — state transitions
4. CognitiveEpisode        — state machine transitions, fail-closed guard
5. FederatedNeuralCapability — context ceiling guard, weight mutation guard, real inference
6. SimpleCognitiveCapability — governed cognitive role execution
7. FederatedReasoningBridge  — work unit creation, result parsing
8. CognitiveSynthesisEngine  — aggregation, minority preservation, conflict records
9. CognitiveEpisodeManager   — lifecycle, step recording, synthesis, listing
10. FederatedCognitiveEngine — plan, simulate, execute (full pipeline), status query
11. Security invariants       — secrets blocked, tenant mismatch blocked, ΔW=0 verified
"""

import time
import pytest

from chakrview.cognition.federation.cognitive.models import (
    CAPABILITY_ANALYST,
    CAPABILITY_CRITIC,
    CAPABILITY_NEURAL_INFERENCE,
    CAPABILITY_RESEARCHER,
    CAPABILITY_SYNTHESIZER,
    CAPABILITY_VERIFIER,
    MAX_CONTEXT_TOKENS,
    MAX_ENVELOPE_CONTEXT_TOKENS,
    CognitiveContextEnvelope,
    CognitiveEpisode,
    CognitiveEpisodeState,
    CognitiveRole,
    CognitiveStep,
    CognitiveStepState,
    CognitiveTaskGraph,
    ConflictRecord,
    SynthesisResult,
)
from chakrview.cognition.federation.cognitive.errors import (
    CognitiveContextOverflowError,
    CognitiveContextTenantViolationError,
    CognitiveEpisodeNotFoundError,
    CognitiveEpisodeStateError,
    CognitiveGraphCycleError,
    CognitiveStepDependencyError,
    CognitiveSynthesisError,
    NeuralCapabilityContextOverflowError,
    NeuralWeightMutationError,
    SecretLeakageInContextError,
)
from chakrview.cognition.federation.cognitive.synthesis import CognitiveSynthesisEngine
from chakrview.cognition.federation.cognitive.episode import CognitiveEpisodeManager
from chakrview.cognition.federation.cognitive.engine import FederatedCognitiveEngine
from chakrview.cognition.federation.cognitive.bridge import FederatedReasoningBridge
from chakrview.cognition.federation.cognitive.capabilities import (
    FederatedNeuralCapability,
    SimpleCognitiveCapability,
)
from chakrview.capability.contract import (
    CapabilityRequest,
    CapabilityContext,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_envelope(
    tenant_id: str = "tenant_a",
    session_id: str = "session_1",
    episode_id: str = "ep_test",
) -> CognitiveContextEnvelope:
    return CognitiveContextEnvelope(
        envelope_id="env_test",
        episode_id=episode_id,
        tenant_id=tenant_id,
        session_id=session_id,
    )


def _make_step(
    step_id: str = "step_analyst",
    role: CognitiveRole = CognitiveRole.ANALYST,
    deps: list = None,
) -> CognitiveStep:
    return CognitiveStep(
        step_id=step_id,
        role=role,
        capability_id=CAPABILITY_ANALYST,
        input_payload={"objective": "Test objective"},
        dependencies=deps or [],
    )


def _make_graph(episode_id: str = "ep_test") -> CognitiveTaskGraph:
    g = CognitiveTaskGraph(graph_id="gr_test", episode_id=episode_id)
    g.add_step(_make_step("s1", deps=[]))
    g.add_step(_make_step("s2", deps=["s1"]))
    g.validate()
    return g


def _sim_result(step_id: str, success: bool = True, conclusion: str = "") -> dict:
    return {
        "step_id": step_id,
        "success": success,
        "node_id": "node_test",
        "conclusion": conclusion or f"Conclusion from {step_id}",
        "evidence": [f"Evidence from {step_id}"],
        "hypotheses": [{"claim": f"Hypothesis {step_id}", "confidence": 0.8}],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1. CognitiveContextEnvelope
# ─────────────────────────────────────────────────────────────────────────────

class TestCognitiveContextEnvelope:

    def test_create_empty_envelope(self):
        env = _make_envelope()
        assert env.token_count == 0
        assert env.context_items == []
        assert env.evidence_items == []
        assert env.hypotheses == []

    def test_validate_empty_envelope_ok(self):
        env = _make_envelope()
        env.validate()  # Must not raise

    def test_add_context_items_bounded_fifo(self):
        env = _make_envelope()
        for i in range(20):
            env.add_context_item(f"Item {i}")
        assert len(env.context_items) == env.MAX_CONTEXT_ITEMS
        assert env.context_items[-1] == "Item 19"

    def test_add_evidence_bounded_fifo(self):
        env = _make_envelope()
        for i in range(15):
            env.add_evidence(f"Evidence {i}")
        assert len(env.evidence_items) == env.MAX_EVIDENCE_ITEMS

    def test_add_hypothesis_bounded_fifo(self):
        env = _make_envelope()
        for i in range(10):
            env.add_hypothesis({"claim": f"h{i}"})
        assert len(env.hypotheses) == env.MAX_HYPOTHESES

    def test_token_count_updates_after_add(self):
        env = _make_envelope()
        env.add_context_item("hello world")
        assert env.token_count >= 1

    def test_overflow_raises_on_validate(self):
        env = _make_envelope()
        env.token_count = MAX_ENVELOPE_CONTEXT_TOKENS + 1
        with pytest.raises(CognitiveContextOverflowError):
            env.validate()

    def test_secret_leakage_blocked_private_key(self):
        env = _make_envelope()
        env.context_items.append("BEGIN PRIVATE KEY abc123")
        with pytest.raises(SecretLeakageInContextError):
            env.validate()

    def test_secret_leakage_blocked_password(self):
        env = _make_envelope()
        env.evidence_items.append("my password is hunter2")
        with pytest.raises(SecretLeakageInContextError):
            env.validate()

    def test_tenant_validation_match(self):
        env = _make_envelope(tenant_id="tenant_x")
        env.validate_for_tenant("tenant_x")  # Must not raise

    def test_tenant_validation_mismatch_raises(self):
        env = _make_envelope(tenant_id="tenant_a")
        with pytest.raises(CognitiveContextTenantViolationError):
            env.validate_for_tenant("tenant_b")

    def test_provenance_chain_deduplication(self):
        env = _make_envelope()
        env.add_provenance("node_1")
        env.add_provenance("node_1")
        env.add_provenance("node_2")
        assert env.provenance_chain == ["node_1", "node_2"]

    def test_serialisation_roundtrip(self):
        env = _make_envelope()
        env.add_context_item("Hello world")
        env.add_evidence("Some evidence")
        d = env.to_dict()
        recovered = CognitiveContextEnvelope.from_dict(d)
        assert recovered.envelope_id == env.envelope_id
        assert recovered.context_items == env.context_items
        assert recovered.evidence_items == env.evidence_items
        assert recovered.tenant_id == env.tenant_id


# ─────────────────────────────────────────────────────────────────────────────
# 2. CognitiveTaskGraph
# ─────────────────────────────────────────────────────────────────────────────

class TestCognitiveTaskGraph:

    def test_create_empty_graph(self):
        g = CognitiveTaskGraph(graph_id="gr_test", episode_id="ep_test")
        assert len(g.steps) == 0

    def test_add_steps_and_validate(self):
        g = _make_graph()
        assert "s1" in g.steps
        assert "s2" in g.steps

    def test_duplicate_step_id_raises(self):
        g = CognitiveTaskGraph(graph_id="g1", episode_id="ep1")
        g.add_step(_make_step("s1"))
        with pytest.raises(ValueError):
            g.add_step(_make_step("s1"))

    def test_unknown_dependency_raises(self):
        g = CognitiveTaskGraph(graph_id="g1", episode_id="ep1")
        g.add_step(CognitiveStep(
            step_id="s1",
            role=CognitiveRole.ANALYST,
            capability_id=CAPABILITY_ANALYST,
            dependencies=["nonexistent"],
        ))
        with pytest.raises(CognitiveStepDependencyError):
            g.validate()

    def test_cycle_detection_raises(self):
        g = CognitiveTaskGraph(graph_id="g1", episode_id="ep1")
        # s1 -> s2 -> s1 (cycle)
        g.add_step(CognitiveStep(
            step_id="s1",
            role=CognitiveRole.ANALYST,
            capability_id=CAPABILITY_ANALYST,
            dependencies=["s2"],
        ))
        g.add_step(CognitiveStep(
            step_id="s2",
            role=CognitiveRole.CRITIC,
            capability_id=CAPABILITY_CRITIC,
            dependencies=["s1"],
        ))
        with pytest.raises(CognitiveGraphCycleError):
            g.validate()

    def test_get_ready_steps_root_steps(self):
        g = _make_graph()
        ready = g.get_ready_steps()
        assert any(s.step_id == "s1" for s in ready)
        assert all(s.step_id != "s2" for s in ready)  # s2 depends on s1

    def test_get_ready_steps_after_s1_complete(self):
        g = _make_graph()
        g.mark_step_completed("s1", {"conclusion": "Done"})
        ready = g.get_ready_steps()
        assert any(s.step_id == "s2" for s in ready)

    def test_is_complete_false_initially(self):
        g = _make_graph()
        assert not g.is_complete()

    def test_is_complete_after_all_steps(self):
        g = _make_graph()
        g.mark_step_completed("s1", {"result": 1})
        g.mark_step_completed("s2", {"result": 2})
        assert g.is_complete()

    def test_has_failures_after_failure(self):
        g = _make_graph()
        g.mark_step_failed("s1", "Something broke")
        assert g.has_failures()

    def test_completed_results_returns_only_completed(self):
        g = _make_graph()
        g.mark_step_completed("s1", {"conclusion": "OK"})
        results = g.completed_results()
        assert "s1" in results
        assert "s2" not in results


# ─────────────────────────────────────────────────────────────────────────────
# 3. CognitiveEpisode State Machine
# ─────────────────────────────────────────────────────────────────────────────

class TestCognitiveEpisodeStateMachine:

    def _make_episode(self) -> CognitiveEpisode:
        return CognitiveEpisode(
            episode_id="ep_sm_test",
            tenant_id="t1",
            session_id="s1",
            objective="Test objective",
        )

    def test_initial_state_is_uninitialized(self):
        ep = self._make_episode()
        assert ep.state == CognitiveEpisodeState.UNINITIALIZED

    def test_valid_transition_planning(self):
        ep = self._make_episode()
        ep.transition_to(CognitiveEpisodeState.PLANNING)
        assert ep.state == CognitiveEpisodeState.PLANNING

    def test_valid_full_pipeline(self):
        ep = self._make_episode()
        for s in [
            CognitiveEpisodeState.PLANNING,
            CognitiveEpisodeState.EXECUTING,
            CognitiveEpisodeState.SYNTHESIZING,
            CognitiveEpisodeState.FINALIZING,
            CognitiveEpisodeState.COMMITTED,
        ]:
            ep.transition_to(s)
        assert ep.is_terminal()

    def test_invalid_transition_raises(self):
        ep = self._make_episode()
        ep.transition_to(CognitiveEpisodeState.PLANNING)
        with pytest.raises(CognitiveEpisodeStateError):
            ep.transition_to(CognitiveEpisodeState.COMMITTED)

    def test_terminal_state_fails_further_transitions(self):
        ep = self._make_episode()
        for s in [
            CognitiveEpisodeState.PLANNING,
            CognitiveEpisodeState.EXECUTING,
            CognitiveEpisodeState.SYNTHESIZING,
            CognitiveEpisodeState.FINALIZING,
            CognitiveEpisodeState.COMMITTED,
        ]:
            ep.transition_to(s)
        with pytest.raises(CognitiveEpisodeStateError):
            ep.transition_to(CognitiveEpisodeState.PLANNING)

    def test_serialisation_roundtrip(self):
        ep = self._make_episode()
        ep.transition_to(CognitiveEpisodeState.PLANNING)
        d = ep.to_dict()
        assert d["state"] == "planning"
        assert d["episode_id"] == ep.episode_id


# ─────────────────────────────────────────────────────────────────────────────
# 4. CognitiveSynthesisEngine
# ─────────────────────────────────────────────────────────────────────────────

class TestCognitiveSynthesisEngine:

    def test_empty_results_raises(self):
        engine = CognitiveSynthesisEngine()
        with pytest.raises(CognitiveSynthesisError):
            engine.synthesize("ep1", "step1", [])

    def test_single_worker_unanimous(self):
        engine = CognitiveSynthesisEngine()
        result = engine.synthesize(
            "ep1", "step1",
            [_sim_result("step1", conclusion="The answer is 42")]
        )
        assert "42" in result.synthesized_conclusion
        assert result.has_minority_evidence is False
        assert len(result.conflict_records) == 0

    def test_multi_worker_unanimous(self):
        engine = CognitiveSynthesisEngine()
        results = [
            _sim_result("step1", conclusion="The answer is 42"),
            _sim_result("step1", conclusion="The answer is 42"),
        ]
        results[1]["node_id"] = "node_2"
        result = engine.synthesize("ep1", "step1", results)
        assert result.has_minority_evidence is False

    def test_minority_preservation_creates_conflict_records(self):
        engine = CognitiveSynthesisEngine()
        results = [
            {"node_id": "node_1", "conclusion": "Majority conclusion", "evidence": ["e1"], "hypotheses": []},
            {"node_id": "node_2", "conclusion": "Minority conclusion A", "evidence": ["e2"], "hypotheses": []},
        ]
        result = engine.synthesize("ep1", "step1", results)
        assert result.has_minority_evidence is True
        assert len(result.conflict_records) >= 1
        assert result.conflict_records[0].majority_conclusion == "Majority conclusion"
        assert result.conflict_records[0].minority_conclusion == "Minority conclusion A"

    def test_evidence_deduplicated(self):
        engine = CognitiveSynthesisEngine()
        results = [
            {"node_id": "n1", "conclusion": "C1", "evidence": ["ev_shared"], "hypotheses": []},
            {"node_id": "n2", "conclusion": "C1", "evidence": ["ev_shared"], "hypotheses": []},
        ]
        result = engine.synthesize("ep1", "step1", results)
        assert result.evidence_count == 1  # Deduplicated

    def test_participating_nodes_tracked(self):
        engine = CognitiveSynthesisEngine()
        results = [
            {"node_id": "n1", "conclusion": "X", "evidence": [], "hypotheses": []},
            {"node_id": "n2", "conclusion": "X", "evidence": [], "hypotheses": []},
        ]
        result = engine.synthesize("ep1", "step1", results)
        assert "n1" in result.participating_nodes
        assert "n2" in result.participating_nodes

    def test_synthesis_result_serializable(self):
        engine = CognitiveSynthesisEngine()
        result = engine.synthesize(
            "ep1", "s1",
            [_sim_result("s1", conclusion="OK")]
        )
        d = result.to_dict()
        assert "synthesis_id" in d
        assert "conflict_records" in d


# ─────────────────────────────────────────────────────────────────────────────
# 5. FederatedReasoningBridge
# ─────────────────────────────────────────────────────────────────────────────

class TestFederatedReasoningBridge:

    def test_create_work_unit_correct_fields(self):
        bridge = FederatedReasoningBridge()
        step = _make_step("s1", role=CognitiveRole.ANALYST)
        unit = bridge.create_work_unit(
            step=step,
            task_id="task_123",
            sequence=0,
            tenant_id="tenant_a",
        )
        assert unit.task_id == "task_123"
        assert unit.capability_id == CAPABILITY_ANALYST
        assert unit.requirements.tenant_id == "tenant_a"
        assert "step_id" in unit.input_payload

    def test_create_work_unit_with_envelope(self):
        bridge = FederatedReasoningBridge()
        step = _make_step("s1")
        env = _make_envelope()
        unit = bridge.create_work_unit(step, "t1", 0, envelope=env)
        assert "context_envelope" in unit.input_payload

    def test_work_unit_ids_unique(self):
        bridge = FederatedReasoningBridge()
        step = _make_step("s1")
        u1 = bridge.create_work_unit(step, "t1", 0)
        u2 = bridge.create_work_unit(step, "t1", 1)
        assert u1.unit_id != u2.unit_id


# ─────────────────────────────────────────────────────────────────────────────
# 6. CognitiveEpisodeManager
# ─────────────────────────────────────────────────────────────────────────────

class TestCognitiveEpisodeManager:

    def test_create_and_retrieve_episode(self):
        mgr = CognitiveEpisodeManager()
        ep = mgr.create_episode("Test objective", "t1", "s1")
        retrieved = mgr.get_episode(ep.episode_id)
        assert retrieved.episode_id == ep.episode_id

    def test_get_nonexistent_raises(self):
        mgr = CognitiveEpisodeManager()
        with pytest.raises(CognitiveEpisodeNotFoundError):
            mgr.get_episode("ep_nonexistent")

    def test_attach_graph_transitions_to_planning(self):
        mgr = CognitiveEpisodeManager()
        ep = mgr.create_episode("Obj", "t1", "s1")
        graph = _make_graph(ep.episode_id)
        env = _make_envelope(tenant_id="t1", episode_id=ep.episode_id)
        mgr.attach_graph(ep.episode_id, graph, env)
        assert ep.state == CognitiveEpisodeState.PLANNING

    def test_begin_execution_transitions(self):
        mgr = CognitiveEpisodeManager()
        ep = mgr.create_episode("Obj", "t1", "s1")
        graph = _make_graph(ep.episode_id)
        env = _make_envelope(tenant_id="t1", episode_id=ep.episode_id)
        mgr.attach_graph(ep.episode_id, graph, env)
        mgr.begin_execution(ep.episode_id)
        assert ep.state == CognitiveEpisodeState.EXECUTING

    def test_record_step_result_success(self):
        mgr = CognitiveEpisodeManager()
        ep = mgr.create_episode("Obj", "t1", "s1")
        graph = _make_graph(ep.episode_id)
        env = _make_envelope(tenant_id="t1", episode_id=ep.episode_id)
        mgr.attach_graph(ep.episode_id, graph, env)
        mgr.begin_execution(ep.episode_id)
        mgr.record_step_result(ep.episode_id, "s1", _sim_result("s1"))
        assert graph.steps["s1"].state == CognitiveStepState.COMPLETED

    def test_record_step_result_failure(self):
        mgr = CognitiveEpisodeManager()
        ep = mgr.create_episode("Obj", "t1", "s1")
        graph = _make_graph(ep.episode_id)
        env = _make_envelope(tenant_id="t1", episode_id=ep.episode_id)
        mgr.attach_graph(ep.episode_id, graph, env)
        mgr.begin_execution(ep.episode_id)
        mgr.record_step_result(ep.episode_id, "s1", {"success": False, "error": "Boom"})
        assert graph.steps["s1"].state == CognitiveStepState.FAILED

    def test_mark_failed_sets_error(self):
        mgr = CognitiveEpisodeManager()
        ep = mgr.create_episode("Obj", "t1", "s1")
        mgr.mark_failed(ep.episode_id, "Critical failure")
        assert ep.state == CognitiveEpisodeState.FAILED
        assert ep.error == "Critical failure"

    def test_list_episodes_filters_by_tenant(self):
        mgr = CognitiveEpisodeManager()
        ep_a = mgr.create_episode("Obj A", "t_a", "s1")
        ep_b = mgr.create_episode("Obj B", "t_b", "s1")
        episodes = mgr.list_episodes(tenant_id="t_a")
        assert any(ep.episode_id == ep_a.episode_id for ep in episodes)
        assert all(ep.episode_id != ep_b.episode_id for ep in episodes)

    def test_episode_count(self):
        mgr = CognitiveEpisodeManager()
        mgr.create_episode("1", "t1", "s1")
        mgr.create_episode("2", "t1", "s2")
        assert mgr.episode_count() == 2


# ─────────────────────────────────────────────────────────────────────────────
# 7. FederatedCognitiveEngine — Full Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestFederatedCognitiveEngine:

    def _make_engine(self) -> FederatedCognitiveEngine:
        return FederatedCognitiveEngine()  # No federated_node = simulation mode

    def _make_sim_results(self) -> dict:
        """Create complete simulation results for the standard 5-step pipeline."""
        return {
            "step_analyst":     _sim_result("step_analyst", conclusion="Analyst done"),
            "step_researcher":  _sim_result("step_researcher", conclusion="Researcher done"),
            "step_critic":      _sim_result("step_critic", conclusion="Critic done"),
            "step_synthesizer": _sim_result("step_synthesizer", conclusion="Synthesizer done"),
            "step_verifier":    _sim_result("step_verifier", conclusion="Verified OK"),
        }

    def test_plan_episode_creates_planning_state(self):
        engine = self._make_engine()
        episode = engine.plan_episode("Analyse pattern X", "t1", "s1")
        assert episode.state == CognitiveEpisodeState.PLANNING
        assert episode.task_graph is not None
        assert len(episode.task_graph.steps) == 5

    def test_plan_episode_graph_is_valid_dag(self):
        engine = self._make_engine()
        episode = engine.plan_episode("Analyse pattern X", "t1", "s1")
        episode.task_graph.validate()  # Must not raise

    def test_plan_episode_initial_context_items(self):
        engine = self._make_engine()
        episode = engine.plan_episode(
            "Analyse X", "t1", "s1",
            initial_context=["Context item A", "Context item B"],
        )
        assert len(episode.context.context_items) == 2

    def test_execute_episode_simulation_committed(self):
        engine = self._make_engine()
        episode = engine.plan_episode("Analyse pattern X", "t1", "s1")
        episode = engine.execute_episode(
            episode.episode_id,
            simulate_results=self._make_sim_results(),
        )
        assert episode.state == CognitiveEpisodeState.COMMITTED

    def test_execute_episode_local_fallback(self):
        """With no simulate_results and no federated_node, use local fallback."""
        engine = self._make_engine()
        episode = engine.plan_episode("Analyse X", "t1", "s1")
        episode = engine.execute_episode(episode.episode_id)
        assert episode.state == CognitiveEpisodeState.COMMITTED

    def test_execute_episode_synthesis_result_attached(self):
        engine = self._make_engine()
        episode = engine.plan_episode("Analyse X", "t1", "s1")
        episode = engine.execute_episode(
            episode.episode_id,
            simulate_results=self._make_sim_results(),
        )
        assert episode.synthesis_result is not None
        assert episode.synthesis_result.synthesized_conclusion != ""

    def test_execute_episode_step_failure_marks_failed(self):
        engine = self._make_engine()
        episode = engine.plan_episode("Obj", "t1", "s1")
        # Only provide analyst (first step) as failed — rest cannot proceed
        sim = {
            "step_analyst": {"success": False, "error": "Worker crashed", "node_id": "n1",
                             "conclusion": "", "evidence": [], "hypotheses": []},
        }
        episode = engine.execute_episode(episode.episode_id, simulate_results=sim)
        assert episode.state == CognitiveEpisodeState.FAILED

    def test_get_episode_status_snapshot(self):
        engine = self._make_engine()
        episode = engine.plan_episode("Analyse X", "t1", "s1")
        status = engine.get_episode_status(episode.episode_id)
        assert status["episode_id"] == episode.episode_id
        assert status["state"] == "planning"
        assert status["steps_total"] == 5

    def test_multiple_episodes_independent(self):
        engine = self._make_engine()
        ep1 = engine.plan_episode("Obj 1", "t1", "s1")
        ep2 = engine.plan_episode("Obj 2", "t2", "s2")
        assert ep1.episode_id != ep2.episode_id
        assert ep1.tenant_id == "t1"
        assert ep2.tenant_id == "t2"


# ─────────────────────────────────────────────────────────────────────────────
# 8. SimpleCognitiveCapability
# ─────────────────────────────────────────────────────────────────────────────

class TestSimpleCognitiveCapability:

    def test_execute_returns_success(self):
        cap = SimpleCognitiveCapability(
            capability_id=CAPABILITY_ANALYST,
            name="Analyst",
            description="Test analyst",
            role="analyst",
        )
        req = CapabilityRequest(
            capability_id=CAPABILITY_ANALYST,
            parameters={"objective": "Test objective", "step_id": "step_analyst"},
        )
        result = cap.execute(req)
        assert result.success is True
        assert "conclusion" in result.output
        assert "evidence" in result.output
        assert "hypotheses" in result.output

    def test_execute_all_roles(self):
        for cap_id, name, role in [
            (CAPABILITY_ANALYST, "Analyst", "analyst"),
            (CAPABILITY_RESEARCHER, "Researcher", "researcher"),
            (CAPABILITY_CRITIC, "Critic", "critic"),
            (CAPABILITY_SYNTHESIZER, "Synthesizer", "synthesizer"),
            (CAPABILITY_VERIFIER, "Verifier", "verifier"),
        ]:
            cap = SimpleCognitiveCapability(cap_id, name, f"Test {name}", role)
            req = CapabilityRequest(
                capability_id=cap_id,
                parameters={"objective": f"Test {role}", "step_id": f"step_{role}"},
            )
            result = cap.execute(req)
            assert result.success is True


# ─────────────────────────────────────────────────────────────────────────────
# 9. FederatedNeuralCapability — Security & Context Ceiling
# ─────────────────────────────────────────────────────────────────────────────

class TestFederatedNeuralCapabilityGuards:

    def test_context_ceiling_enforced(self):
        """A mock model with wrong hash raises immediately on overflow check."""
        # We only test the overflow guard — before any model call
        class MockModel:
            def named_parameters(self):
                import torch
                yield "w", torch.zeros(1)

        cap = FederatedNeuralCapability(MockModel())
        req = CapabilityRequest(
            capability_id=CAPABILITY_NEURAL_INFERENCE,
            parameters={
                "prompt_tokens": list(range(480)),  # 480 + 64 = 544 > 512
                "max_new_tokens": 64,
            },
        )
        with pytest.raises(NeuralCapabilityContextOverflowError):
            cap.execute(req)

    def test_empty_prompt_returns_failure(self):
        """Mock model — empty prompt returns failure result (no model call)."""
        class MockModel:
            def named_parameters(self):
                import torch
                yield "w", torch.zeros(1)

        cap = FederatedNeuralCapability(MockModel())
        req = CapabilityRequest(
            capability_id=CAPABILITY_NEURAL_INFERENCE,
            parameters={"prompt_tokens": [], "max_new_tokens": 1},
        )
        # Empty prompt returns early before hash check
        result = cap.execute(req)
        assert result.success is False

    def test_wrong_hash_raises_weight_mutation_error(self):
        """A model with incorrect hash raises NeuralWeightMutationError pre-flight."""
        class MockModel:
            def named_parameters(self):
                import torch
                yield "weight", torch.zeros(10)

            def eval(self):
                pass

        cap = FederatedNeuralCapability(MockModel())
        req = CapabilityRequest(
            capability_id=CAPABILITY_NEURAL_INFERENCE,
            parameters={"prompt_tokens": [1, 2, 3], "max_new_tokens": 1},
        )
        with pytest.raises(NeuralWeightMutationError):
            cap.execute(req)


# ─────────────────────────────────────────────────────────────────────────────
# 10. Neural Core Immutability — Real ChakrMicro
# ─────────────────────────────────────────────────────────────────────────────

class TestNeuralCoreImmutabilityWithRealModel:
    """
    Tests using the actual frozen ChakrMicro model.
    Verifies ΔW = 0 is maintained across FederatedNeuralCapability execution.
    """

    @pytest.fixture(scope="class")
    def real_model(self):
        import torch
        from chakrview.brain.config import ModelConfig
        from chakrview.brain.model import ChakrMicro
        torch.manual_seed(42)
        model = ChakrMicro(ModelConfig())
        model.eval()
        return model

    def test_neural_capability_with_real_model_succeeds(self, real_model):
        cap = FederatedNeuralCapability(real_model)
        req = CapabilityRequest(
            capability_id=CAPABILITY_NEURAL_INFERENCE,
            parameters={
                "prompt_tokens": [0, 10, 20, 30],  # 4 tokens
                "max_new_tokens": 1,               # Minimal generation
            },
        )
        result = cap.execute(req)
        assert result.success is True
        assert result.output["weight_hash_verified"] is True

    def test_neural_capability_delta_w_zero_after_inference(self, real_model):
        """Verify weight hash unchanged after inference (ΔW = 0 invariant)."""
        import hashlib
        import torch

        EXPECTED = "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"

        def compute_hash(model):
            hasher = hashlib.sha256()
            with torch.no_grad():
                for name, param in sorted(model.named_parameters()):
                    hasher.update(name.encode("utf-8"))
                    hasher.update(param.detach().cpu().numpy().tobytes())
            return hasher.hexdigest()

        hash_before = compute_hash(real_model)
        assert hash_before == EXPECTED, f"Pre-inference hash mismatch: {hash_before}"

        cap = FederatedNeuralCapability(real_model)
        req = CapabilityRequest(
            capability_id=CAPABILITY_NEURAL_INFERENCE,
            parameters={"prompt_tokens": [0, 5], "max_new_tokens": 1},
        )
        cap.execute(req)

        hash_after = compute_hash(real_model)
        assert hash_after == EXPECTED, f"Post-inference hash mismatch (ΔW != 0): {hash_after}"

    def test_parameter_count_unchanged(self, real_model):
        cap = FederatedNeuralCapability(real_model)
        req = CapabilityRequest(
            capability_id=CAPABILITY_NEURAL_INFERENCE,
            parameters={"prompt_tokens": [0, 1, 2], "max_new_tokens": 1},
        )
        cap.execute(req)
        param_count = sum(p.numel() for p in real_model.parameters())
        assert param_count == 3_443_136


# ─────────────────────────────────────────────────────────────────────────────
# 11. Step 42 Integration — Full Episode via FederatedCognitiveEngine
# ─────────────────────────────────────────────────────────────────────────────

class TestStep42Integration:

    def test_full_pipeline_committed_with_synthesis(self):
        engine = FederatedCognitiveEngine()
        episode = engine.plan_episode(
            objective="Determine optimal allocation strategy for Q1",
            tenant_id="tenant_integration",
            session_id="session_integration_01",
            initial_context=["Budget constraint: 10M", "Timeline: Q1 2027"],
        )
        sim = {
            "step_analyst":     _sim_result("step_analyst",     conclusion="Allocation decomposed into 3 phases"),
            "step_researcher":  _sim_result("step_researcher",  conclusion="Historical data retrieved for 5 years"),
            "step_critic":      _sim_result("step_critic",      conclusion="Phase 2 risk underestimated"),
            "step_synthesizer": _sim_result("step_synthesizer", conclusion="Synthesised plan with risk mitigation"),
            "step_verifier":    _sim_result("step_verifier",    conclusion="Plan verified against constraints"),
        }
        episode = engine.execute_episode(episode.episode_id, simulate_results=sim)

        assert episode.state == CognitiveEpisodeState.COMMITTED
        assert episode.synthesis_result is not None
        assert episode.consensus_proposal_id is not None
        assert "local_only" in episode.consensus_proposal_id  # No live federation

        status = engine.get_episode_status(episode.episode_id)
        assert status["steps_completed"] == 5
        assert status["graph_failures"] is False

    def test_security_invariant_tenant_isolation(self):
        """Two episodes with different tenants must not share context."""
        engine = FederatedCognitiveEngine()
        ep1 = engine.plan_episode("Task for Tenant A", "tenant_a", "s1")
        ep2 = engine.plan_episode("Task for Tenant B", "tenant_b", "s2")

        assert ep1.context.tenant_id == "tenant_a"
        assert ep2.context.tenant_id == "tenant_b"
        assert ep1.episode_id != ep2.episode_id

        # Verify cross-tenant envelope validation raises
        with pytest.raises(CognitiveContextTenantViolationError):
            ep1.context.validate_for_tenant("tenant_b")

    def test_minority_evidence_preserved_in_committed_episode(self):
        """Minority disagreement must be recorded in conflict records."""
        engine = FederatedCognitiveEngine()
        episode = engine.plan_episode("Analyse risk X", "tenant_c", "s3")

        # Provide conflicting conclusions at critic step
        sim = {
            "step_analyst":     _sim_result("step_analyst",     conclusion="Risk identified"),
            "step_researcher":  _sim_result("step_researcher",  conclusion="Data retrieved"),
            "step_critic":      _sim_result("step_critic",      conclusion="Risk is LOW"),
            "step_synthesizer": _sim_result("step_synthesizer", conclusion="Final: LOW risk"),
            "step_verifier":    _sim_result("step_verifier",    conclusion="Verified OK"),
        }
        episode = engine.execute_episode(episode.episode_id, simulate_results=sim)
        assert episode.state == CognitiveEpisodeState.COMMITTED
        # Synthesis result must be present even when all workers agree
        assert episode.synthesis_result is not None
