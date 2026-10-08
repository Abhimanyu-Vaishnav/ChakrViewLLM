"""
Test Suite for Waves 581-590: ChakrView v0.4.0 Candidate Capabilities.
"""

import pytest
import torch

from chakrview.brain.canonical_hash import verify_canonical_baseline
from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro

from chakrview.cognition.v040_candidate.working_memory import WorkingMemory2, TaskConstraint
from chakrview.cognition.v040_candidate.chunk_processor import LongContextChunkProcessor
from chakrview.cognition.v040_candidate.hierarchical_planner import HierarchicalPlanner, SubGoal
from chakrview.cognition.v040_candidate.multi_step_executor import MultiStepGoalExecutor, ActionStatus
from chakrview.cognition.v040_candidate.memory_reasoning_bridge import MemoryReasoningBridge, MemoryBeliefRecord
from chakrview.cognition.v040_candidate.experience_reasoning_bridge import ExperienceReasoningBridge
from chakrview.cognition.v040_candidate.language_cognitive_bridge import LanguageCognitiveBridge
from chakrview.cognition.v040_candidate.adaptive_solver import AdaptiveProblemSolver, ProblemSolvingStrategy
from chakrview.cognition.v040_candidate.unified_benchmark import UnifiedCognitiveBenchmarkRunner
from chakrview.cognition.v040_candidate.v040_master_gate import V040CandidateMasterGate


def test_canonical_baseline_bit_exact():
    """Verify neural baseline remains bit-exact (Delta W = 0)."""
    torch.manual_seed(42)
    m = ChakrMicro(ModelConfig())
    res = verify_canonical_baseline(m)
    assert res["is_bit_exact"] is True
    assert res["delta_w"] == 0
    assert res["total_params"] == 3443136


def test_wave581_working_memory_lifecycle():
    """Verify task-bounded state retention, reset, and checkpoint restoration."""
    wm = WorkingMemory2()
    wm.initialize_task("task_wm_1", "Multi-step reasoning")
    wm.set_variable("task_wm_1", "step_alpha", {"score": 0.98})
    wm.push_result("task_wm_1", "action_1", "done")
    assert wm.get_variable("task_wm_1", "step_alpha")["score"] == 0.98

    # Checkpoint and reset
    chk = wm.checkpoint_state("task_wm_1")
    wm.reset_task("task_wm_1")
    assert "task_wm_1" not in wm.active_tasks

    # Restore
    restored = wm.restore_checkpoint(chk)
    assert restored.task_id == "task_wm_1"
    assert restored.intermediate_variables["step_alpha"]["score"] == 0.98


def test_wave582_long_context_chunk_continuity():
    """Verify long-context streaming processing keeps memory footprint bounded."""
    paragraphs = [
        "In 1969 Apollo 11 landed on the Moon. Armstrong took the first steps.",
        "The lunar module Eagle was piloted by Aldrin. Collins orbited above.",
        "Samples of lunar regolith were collected and returned to Earth safely.",
    ]
    res = LongContextChunkProcessor.process_long_text(paragraphs, chunk_max_chars=60)
    assert res.is_complete is True
    assert res.total_chunks >= 2
    assert "Apollo" in res.extracted_entities or "Moon" in res.extracted_entities
    assert res.peak_working_bytes < 50_000  # Strictly bounded


def test_wave583_hierarchical_planning_decomposition():
    """Verify hierarchical plan creation and dynamic replanning on failure."""
    plan = HierarchicalPlanner.create_plan("Synthesize research paper")
    assert len(plan.subgoals) == 3
    assert plan.subgoals[0].action_type == "RETRIEVE"

    # Test replan insertion
    replanned = HierarchicalPlanner.replan_on_failure(plan, "sg_1_gather", "Source timeout")
    assert replanned.replanned_count == 1
    assert any("Recover" in sg.description for sg in replanned.subgoals)


def test_wave584_multi_step_goal_execution_and_recovery():
    """Verify verified step execution with failure recovery trigger."""
    plan = HierarchicalPlanner.create_plan("Multi-stage computation")

    call_counts = {"RETRIEVE": 0, "VERIFY": 0, "SYNTHESIZE": 0}

    def handler(sg: SubGoal):
        call_counts[sg.action_type] = call_counts.get(sg.action_type, 0) + 1
        # Cause first retrieve step to fail, triggering replanning
        if sg.subgoal_id == "sg_1_gather" and plan.replanned_count == 0:
            return ActionStatus.FAILURE, "Network transient error"
        return ActionStatus.SUCCESS, f"Success on {sg.subgoal_id}"

    outcome = MultiStepGoalExecutor.execute_plan(plan, {
        "RETRIEVE": handler,
        "VERIFY": handler,
        "SYNTHESIZE": handler,
    })
    assert outcome.recovery_triggered is True
    assert outcome.total_steps_executed >= 3


def test_wave585_memory_evidence_precedence():
    """Verify that new verified evidence supersedes older memory beliefs."""
    bridge = MemoryReasoningBridge()
    bridge.store_belief(MemoryBeliefRecord(
        belief_id="bel_1", entity="ServerCluster", attribute="active_node",
        value="Node_Alpha", confidence=0.90, observed_timestamp=1000.0,
    ))

    # Newer evidence at t=2000
    val, rationale, conf = bridge.integrate_new_evidence(
        entity="ServerCluster", attribute="active_node",
        evidence_value="Node_Beta", evidence_timestamp=2000.0,
    )
    assert val == "Node_Beta"
    assert "superseded stale memory" in rationale
    assert bridge.get_belief("ServerCluster", "active_node").value == "Node_Beta"


def test_wave586_experience_strategy_selection():
    """Verify structural matching of successful strategies and rejection of failed ones."""
    bridge = ExperienceReasoningBridge()
    # Should match multihop divide strategy
    best, rat = bridge.match_best_strategy("MULTI_HOP_REASONING")
    assert best is not None
    assert best.strategy_id == "strat_multihop_divide"

    # Unknown handling has only failed strategy -> should return None
    best_unknown, rat_unk = bridge.match_best_strategy("UNKNOWN_HANDLING")
    assert best_unknown is None
    assert "No proven prior strategy found" in rat_unk


def test_wave587_language_cognitive_state_parsing():
    """Verify command vs factual query discrimination and intent parsing."""
    frame_cmd = LanguageCognitiveBridge.parse_to_cognitive_frame("Verify the hash for model checkpoint")
    assert frame_cmd.is_command_or_instruction is True
    assert frame_cmd.detected_language == "ENGLISH"

    frame_hi = LanguageCognitiveBridge.parse_to_cognitive_frame("Yeh kya hai batao")
    assert frame_hi.detected_language == "HINDI"
    assert frame_hi.is_factual_query is True


def test_wave588_adaptive_problem_solving():
    """Verify dynamic strategy selection under low-resource and long-document constraints."""
    # Long doc in low resource profile -> streaming chunk
    dec = AdaptiveProblemSolver.select_strategy(
        task_type="LONG_DOCUMENT",
        input_length_chars=2500,
        is_low_resource_profile=True,
        has_sufficient_evidence=True,
        is_ambiguous=False,
        has_conflict=False,
    )
    assert dec.chosen_strategy == ProblemSolvingStrategy.STREAMING_CHUNK_SYNTHESIS
    assert dec.expected_memory_bytes <= 150_000


def test_wave589_unified_cognitive_capability_benchmark():
    """Run held-out benchmark evaluation across all 8 dimensions."""
    bench_res = UnifiedCognitiveBenchmarkRunner.evaluate_benchmark(partition="FINAL_HELD_OUT")
    assert bench_res["total_cases"] == 40
    assert bench_res["accuracy"] == 1.0  # 40/40 passed


def test_wave590_v040_candidate_master_gate():
    """Evaluate candidate master audit gate for v0.4 release readiness."""
    report = V040CandidateMasterGate.evaluate_candidate()
    assert report.canonical_baseline_intact is True
    assert report.delta_w == 0
    assert report.total_params == 3443136
    assert report.all_gates_pass is True
    assert report.audit_verdict == "V040_RELEASE_CANDIDATE_READY"
