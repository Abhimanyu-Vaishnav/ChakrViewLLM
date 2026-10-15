"""
Unit and Integration Tests for Batches 661-700:
Hierarchical Dialogue Planning, Episodic Memory & Evidence-Grounded Multi-Step Reasoning.
Location: tests/test_batch661_700_dialogue_planning.py
"""
import tempfile
import pytest

from chakrview.runtime.episodic_task_memory import (
    EpisodicTaskMemory,
    Subtask,
    SubtaskStatus,
    TaskPlan,
    TaskPlanStatus,
)
from chakrview.runtime.hierarchical_planner import HierarchicalDialoguePlanner


@pytest.fixture
def temp_memory():
    with tempfile.TemporaryDirectory() as td:
        yield EpisodicTaskMemory(td)


@pytest.fixture
def planner(temp_memory):
    return HierarchicalDialoguePlanner(temp_memory)


def test_episodic_memory_save_and_reload(temp_memory):
    """Verifies atomic plan persistence, recovery, and serialization."""
    plan = TaskPlan(
        plan_id="test_plan_01",
        session_id="session_persist",
        user_goal="Calculate 5 * 5",
        subtasks=[
            Subtask(
                subtask_id="st_01",
                goal_description="Evaluate 5 * 5",
                action_type="tool_call",
                tool_name="calculator",
                tool_arguments={"expression": "5 * 5"},
                status=SubtaskStatus.READY,
            )
        ],
        active_constraints={"env": "test"},
        plan_status=TaskPlanStatus.READY,
    )
    temp_memory.save_plan(plan)

    reloaded = temp_memory.get_plan("session_persist", "test_plan_01")
    assert reloaded is not None
    assert reloaded.user_goal == "Calculate 5 * 5"
    assert len(reloaded.subtasks) == 1
    assert reloaded.subtasks[0].tool_name == "calculator"


def test_compound_arithmetic_dag_execution(planner):
    """Tests decomposition and sequential DAG execution of compound math goals."""
    sess = "sess_math_dag"
    plan = planner.plan_user_goal(sess, "Calculate 15 * 12 and then add 20")
    assert len(plan.subtasks) == 2
    assert plan.subtasks[1].dependencies == ["st_01"]

    executed = planner.execute_plan(sess, plan.plan_id)
    assert executed.plan_status == TaskPlanStatus.COMPLETED
    assert executed.subtasks[0].result_artifact == "180"
    assert executed.subtasks[1].result_artifact == "200"


def test_text_draft_transform_pipeline(planner):
    """Tests text recording and subsequent tool transformation."""
    sess = "sess_text_dag"
    plan = planner.plan_user_goal(sess, "Draft text 'ChakrView Core' and convert it to uppercase")
    assert len(plan.subtasks) == 2

    executed = planner.execute_plan(sess, plan.plan_id)
    assert executed.plan_status == TaskPlanStatus.COMPLETED
    assert executed.subtasks[0].result_artifact == "ChakrView Core"
    assert executed.subtasks[1].result_artifact == "CHAKRVIEW CORE"


def test_missing_information_abstention_and_clarification(planner):
    """Verifies that underspecified goals trigger targeted clarification."""
    sess = "sess_clarify"
    plan = planner.plan_user_goal(sess, "Calculate the total price with discount applied")
    assert plan.plan_status == TaskPlanStatus.AWAITING_CLARIFICATION
    assert plan.subtasks[0].status == SubtaskStatus.AWAITING_CLARIFICATION
    assert "not specified" in plan.subtasks[0].error_message


def test_dynamic_constraint_replanning_and_invalidation(planner):
    """Tests that modifying a downstream parameter invalidates downstream steps while preserving upstream."""
    sess = "sess_replan"
    plan = planner.plan_user_goal(sess, "Calculate 10 * 10 and then add 50")
    executed = planner.execute_plan(sess, plan.plan_id)
    assert executed.subtasks[-1].result_artifact == "150"

    # User changes addition operand from 50 to 100
    new_plan = planner.replan_on_constraint_change(
        session_id=sess,
        active_plan_id=executed.plan_id,
        updated_constraints={"updated_operand": 100},
        modified_subtask_id="st_02",
        new_parameter=("operand", "100"),
    )
    assert new_plan.version == 2
    assert new_plan.subtasks[0].status == SubtaskStatus.COMPLETED  # Upstream preserved!
    assert new_plan.subtasks[1].status == SubtaskStatus.PROPOSED   # Downstream invalidated!

    executed2 = planner.execute_plan(sess, new_plan.plan_id)
    assert executed2.subtasks[-1].result_artifact == "200"


def test_session_restart_and_resume():
    """Simulates complete process restart from on-disk episodic logs."""
    with tempfile.TemporaryDirectory() as td:
        mem1 = EpisodicTaskMemory(td)
        planner1 = HierarchicalDialoguePlanner(mem1)
        sess = "restart_sess"
        plan = planner1.plan_user_goal(sess, "Calculate 20 * 5 and then add 30")
        mem1.save_plan(plan)

        # Process restarts: create fresh memory pointing to same directory
        mem2 = EpisodicTaskMemory(td)
        planner2 = HierarchicalDialoguePlanner(mem2)
        resumed_plan = mem2.get_plan(sess, plan.plan_id)
        assert resumed_plan is not None

        executed = planner2.execute_plan(sess, resumed_plan.plan_id)
        assert executed.plan_status == TaskPlanStatus.COMPLETED
        assert executed.subtasks[-1].result_artifact == "130"


def test_cross_session_privacy_isolation(planner):
    """Verifies that Session B cannot inspect or access active plans of Session A."""
    sess_a = "user_alice_secret_sess"
    sess_b = "user_bob_guest_sess"

    planner.plan_user_goal(sess_a, "Calculate 50 * 50 and then add 100")
    plan_b = planner.memory.get_active_plan(sess_b)
    assert plan_b is None
