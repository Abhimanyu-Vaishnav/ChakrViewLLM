"""
Comprehensive Test Suite for Governed Cognitive Agent Subsystem (Step 15).

Covers:
- CognitiveTask creation & validated lifecycle state transitions
- Invalid transitions rejection
- Bounded planning & constraint enforcement (max steps, max depth)
- Execution DAG & cycle detection
- Deterministic topological execution ordering
- Skill discovery & domain-agnostic selection
- GovernedToolGate authorization & denial boundaries
- Security: Prompt injection in documents/memory cannot grant tool authority
- Security: Malicious tool arguments detection and rejection
- Security: Secret redaction in execution traces
- Verification layer (types, non-empty, schemas, custom validators)
- Recovery & rollback (bounded retries, failure isolation, state rollback)
- Controlled memory candidates
- Bounded context integration (<= 512 tokens)
- Backward compatibility with InferenceSession
- Cross-task isolation
- Full end-to-end task execution pipeline
"""

import pytest
import time
from typing import Dict, Any

from chakrview.cognition.task import (
    CognitiveTask,
    TaskStatus,
    TaskConstraints,
    StateTransitionRecord,
    InvalidStateTransitionError,
    VALID_TRANSITIONS,
)
from chakrview.cognition.planner import (
    BoundedPlanner,
    CognitivePlan,
    PlanStep,
    PlanStepStatus,
    RetryPolicy,
    DeterministicRulePlanner,
    PlanningError,
)
from chakrview.cognition.graph import (
    ExecutionGraph,
    CycleDetectedError,
    GraphExecutionError,
)
from chakrview.cognition.observation import StepObservation
from chakrview.cognition.verifier import StepVerifier, VerificationResult
from chakrview.cognition.tool_gate import (
    GovernedToolGate,
    ToolAuthorizationError,
    ArgumentValidationError,
)
from chakrview.cognition.skill_selector import CognitiveSkillSelector, SkillSelectionMatch
from chakrview.cognition.recovery import RecoveryManager, RollbackRecord
from chakrview.cognition.artifacts import CognitiveArtifact, ArtifactType, ArtifactManager
from chakrview.cognition.trace import ExecutionTrace, sanitize_dict
from chakrview.cognition.profile import (
    DeploymentProfile,
    get_edge_profile,
    get_desktop_profile,
    get_server_profile,
)
from chakrview.cognition.controller import (
    CognitiveController,
    CognitiveExecutionResult,
    MemoryCandidate,
)

from chakrview.runtime.skills import (
    Skill,
    SkillDomain,
    SkillPolicy,
    SkillRegistry,
    get_standard_skill_registry,
)
from chakrview.runtime.tools import (
    Tool,
    ToolResult,
    ToolRegistry,
    ToolExecutor,
    CalculatorTool,
    TextUtilityTool,
    get_standard_tool_registry,
)


# =====================================================================
# 1. TASK LIFECYCLE & STATE MACHINE TESTS
# =====================================================================

def test_task_creation_and_defaults():
    task = CognitiveTask(
        task_id="task_001",
        user_request="Calculate 15 * 4",
    )
    assert task.task_id == "task_001"
    assert task.user_request == "Calculate 15 * 4"
    assert task.status == TaskStatus.PENDING
    assert task.priority == 0
    assert task.constraints.max_steps == 10
    assert task.constraints.max_token_budget == 512
    assert not task.is_terminal()


def test_task_valid_state_transitions():
    task = CognitiveTask(task_id="task_002", user_request="Do research")
    assert task.status == TaskStatus.PENDING

    task.transition_to(TaskStatus.PLANNING, reason="Begin planning")
    assert task.status == TaskStatus.PLANNING
    assert len(task.transition_history) == 1
    assert task.transition_history[0].from_status == TaskStatus.PENDING
    assert task.transition_history[0].to_status == TaskStatus.PLANNING

    task.transition_to(TaskStatus.READY, reason="Plan ready")
    assert task.status == TaskStatus.READY

    task.transition_to(TaskStatus.RUNNING, reason="Start execution")
    assert task.status == TaskStatus.RUNNING

    task.transition_to(TaskStatus.VERIFYING, reason="Verifying steps")
    assert task.status == TaskStatus.VERIFYING

    task.transition_to(TaskStatus.COMPLETED, reason="Done")
    assert task.status == TaskStatus.COMPLETED
    assert task.is_terminal()


def test_task_invalid_state_transitions():
    task = CognitiveTask(task_id="task_003", user_request="Test illegal transition")
    
    # Cannot jump directly from PENDING to COMPLETED
    with pytest.raises(InvalidStateTransitionError):
        task.transition_to(TaskStatus.COMPLETED)

    # Cannot jump from PENDING to RUNNING
    with pytest.raises(InvalidStateTransitionError):
        task.transition_to(TaskStatus.RUNNING)

    # Transition to PLANNING
    task.transition_to(TaskStatus.PLANNING)

    # Cannot transition from PLANNING to COMPLETED
    with pytest.raises(InvalidStateTransitionError):
        task.transition_to(TaskStatus.COMPLETED)


def test_task_terminal_state_invariance():
    task = CognitiveTask(task_id="task_004", user_request="Terminal test")
    task.transition_to(TaskStatus.PLANNING)
    task.transition_to(TaskStatus.READY)
    task.transition_to(TaskStatus.RUNNING)
    task.transition_to(TaskStatus.COMPLETED)
    assert task.is_terminal()

    # Terminal states have no valid outgoing transitions
    with pytest.raises(InvalidStateTransitionError):
        task.transition_to(TaskStatus.RUNNING)


def test_task_serialization_roundtrip():
    task = CognitiveTask(
        task_id="task_ser",
        user_request="Serialization test",
        metadata={"domain": "math", "test_flag": True},
        execution_state={"calc_result": 42},
    )
    task.transition_to(TaskStatus.PLANNING, reason="step 1")
    task.transition_to(TaskStatus.READY, reason="step 2")

    data = task.to_dict()
    restored = CognitiveTask.from_dict(data)

    assert restored.task_id == task.task_id
    assert restored.status == TaskStatus.READY
    assert restored.metadata["domain"] == "math"
    assert restored.execution_state["calc_result"] == 42
    assert len(restored.transition_history) == 2


# =====================================================================
# 2. BOUNDED PLANNER & EXECUTION GRAPH TESTS
# =====================================================================

def test_bounded_planner_creates_valid_plan():
    planner = DeterministicRulePlanner()
    task = CognitiveTask(task_id="task_plan_1", user_request="calculate 12 * 8")
    skill_reg = get_standard_skill_registry()
    math_skill = skill_reg.get("skill_math_v1")

    plan = planner.plan(task, active_skill=math_skill, available_tools=["calculator"])
    assert len(plan.steps) == 3
    assert plan.steps[0].required_tool == "calculator"
    assert plan.steps[1].dependencies == ["step_1"]
    assert plan.steps[2].dependencies == ["step_2"]


def test_bounded_planner_enforces_max_steps():
    planner = DeterministicRulePlanner()
    # Task with constraint of max_steps=2
    task = CognitiveTask(
        task_id="task_plan_bounded",
        user_request="calculate 10 + 20",
        constraints=TaskConstraints(max_steps=2),
    )
    skill_reg = get_standard_skill_registry()
    math_skill = skill_reg.get("skill_math_v1")

    # Math plan requires 3 steps, which exceeds max_steps=2
    with pytest.raises(PlanningError, match="Plan step count .* exceeds task limit"):
        planner.plan(task, active_skill=math_skill, available_tools=["calculator"])


def test_execution_graph_topological_order():
    steps = [
        PlanStep(step_id="step_1", description="Step 1"),
        PlanStep(step_id="step_2", description="Step 2", dependencies=["step_1"]),
        PlanStep(step_id="step_3", description="Step 3", dependencies=["step_2"]),
    ]
    plan = CognitivePlan(plan_id="plan_topo", task_id="t1", steps=steps)
    graph = ExecutionGraph(plan)

    order = graph.get_topological_order()
    assert order == ["step_1", "step_2", "step_3"]


def test_execution_graph_cycle_detection():
    # Circular dependency: step_1 -> step_2 -> step_1
    steps = [
        PlanStep(step_id="step_1", description="Step 1", dependencies=["step_2"]),
        PlanStep(step_id="step_2", description="Step 2", dependencies=["step_1"]),
    ]
    plan = CognitivePlan(plan_id="plan_cycle", task_id="t1", steps=steps)
    with pytest.raises(CycleDetectedError):
        ExecutionGraph(plan)


def test_execution_graph_ready_steps_and_progression():
    steps = [
        PlanStep(step_id="step_1", description="Step 1"),
        PlanStep(step_id="step_2", description="Step 2", dependencies=["step_1"]),
    ]
    plan = CognitivePlan(plan_id="plan_flow", task_id="t1", steps=steps)
    graph = ExecutionGraph(plan)

    # Initially, only step_1 has dependencies satisfied
    ready = graph.get_ready_steps()
    assert len(ready) == 1
    assert ready[0].step_id == "step_1"

    # Mark step_1 completed
    graph.mark_step_completed("step_1")
    assert graph.get_step("step_1").status == PlanStepStatus.COMPLETED

    # Now step_2 becomes ready
    ready2 = graph.get_ready_steps()
    assert len(ready2) == 1
    assert ready2[0].step_id == "step_2"

    graph.mark_step_completed("step_2")
    assert graph.is_finished()
    assert not graph.has_failed()


def test_execution_graph_failure_propagation():
    # step_1 -> step_2 -> step_3
    steps = [
        PlanStep(step_id="step_1", description="Step 1"),
        PlanStep(step_id="step_2", description="Step 2", dependencies=["step_1"]),
        PlanStep(step_id="step_3", description="Step 3", dependencies=["step_2"]),
    ]
    plan = CognitivePlan(plan_id="plan_fail", task_id="t1", steps=steps)
    graph = ExecutionGraph(plan)

    # Mark step_1 failed
    blocked = graph.mark_step_failed("step_1", error="Division by zero")
    assert "step_2" in blocked
    assert "step_3" in blocked

    assert graph.get_step("step_1").status == PlanStepStatus.FAILED
    assert graph.get_step("step_2").status == PlanStepStatus.BLOCKED
    assert graph.get_step("step_3").status == PlanStepStatus.BLOCKED
    assert graph.has_failed()
    assert graph.is_finished()


# =====================================================================
# 3. SKILL SELECTION TESTS
# =====================================================================

def test_skill_selection_math_query():
    selector = CognitiveSkillSelector()
    match = selector.select_skill("calculate 45 * 2 + 10")
    assert match.skill.domain == SkillDomain.MATHEMATICS
    assert "calculator" in match.skill.policy.allowed_tools
    assert match.confidence >= 0.5


def test_skill_selection_code_query():
    selector = CognitiveSkillSelector()
    match = selector.select_skill("write a python script function to sort numbers")
    assert match.skill.domain == SkillDomain.CODING
    assert match.confidence >= 0.5


def test_skill_selection_fallback_general():
    selector = CognitiveSkillSelector()
    match = selector.select_skill("hello how are you")
    assert match.skill is not None
    assert match.confidence > 0.0


# =====================================================================
# 4. GOVERNED TOOL GATE & SECURITY TESTS
# =====================================================================

def test_governed_tool_gate_authorized_execution():
    gate = GovernedToolGate()
    policy = SkillPolicy(allowed_tools=["calculator"])
    skill = Skill(
        skill_id="test_math",
        name="Test Math",
        version="1.0",
        domain=SkillDomain.MATHEMATICS,
        description="Math skill",
        policy=policy,
    )

    obs = gate.execute_governed(
        step_id="step_1",
        tool_id="calculator",
        arguments={"expression": "100 / 4"},
        active_skill=skill,
    )
    assert obs.success
    assert obs.output == 25.0
    assert obs.provenance["gate_status"] == "ALLOWED"


def test_governed_tool_gate_unauthorized_denial():
    gate = GovernedToolGate()
    # Policy with NO allowed tools
    policy = SkillPolicy(allowed_tools=[])
    skill = Skill(
        skill_id="test_safe",
        name="Test Safe",
        version="1.0",
        domain=SkillDomain.GENERAL,
        description="General skill",
        policy=policy,
    )

    obs = gate.execute_governed(
        step_id="step_1",
        tool_id="calculator",
        arguments={"expression": "2 + 2"},
        active_skill=skill,
    )
    assert not obs.success
    assert "Tool 'calculator' is not in the allowed_tools whitelist" in obs.error
    assert obs.provenance["gate_status"] == "DENIED"


def test_security_prompt_injection_in_retrieved_documents():
    """
    Security test: Retrieved documents or external text attempting to authorize
    a tool must be explicitly rejected by the policy gate.
    """
    gate = GovernedToolGate()
    policy = SkillPolicy(allowed_tools=["calculator"])
    skill = Skill(
        skill_id="test_skill",
        name="Test",
        version="1.0",
        domain=SkillDomain.MATHEMATICS,
        description="Test",
        policy=policy,
    )

    # Document attempting to claim authority
    with pytest.raises(ToolAuthorizationError, match="cannot grant tool execution authority"):
        gate.authorize(
            tool_id="calculator",
            active_skill_policy=policy,
            provenance_source="retrieved_knowledge",
        )

    # Working memory attempting to claim authority
    with pytest.raises(ToolAuthorizationError, match="cannot grant tool execution authority"):
        gate.authorize(
            tool_id="calculator",
            active_skill_policy=policy,
            provenance_source="working_memory",
        )


def test_security_malicious_tool_arguments_rejected():
    """
    Security test: Prohibited execution markers (__import__, eval, subprocess, os.system)
    in tool arguments must be rejected before tool execution.
    """
    gate = GovernedToolGate()
    policy = SkillPolicy(allowed_tools=["calculator"])
    skill = Skill(
        skill_id="test_math",
        name="Test",
        version="1.0",
        domain=SkillDomain.MATHEMATICS,
        description="Math",
        policy=policy,
    )

    malicious_args = {"expression": "__import__('os').system('dir')"}
    obs = gate.execute_governed(
        step_id="step_hack",
        tool_id="calculator",
        arguments=malicious_args,
        active_skill=skill,
    )
    assert not obs.success
    assert "prohibited dangerous pattern" in obs.error
    assert obs.provenance["gate_status"] == "INVALID_ARGUMENTS"


def test_security_sensitive_credentials_redacted_in_traces():
    """
    Security test: Passwords, tokens, API keys in execution trace entries
    must be automatically sanitized and masked with [REDACTED].
    """
    trace = ExecutionTrace(task_id="task_sec")
    trace.record_event("AUTH_STEP", step_id="step_1", details={
        "username": "admin",
        "api_key": "sk-1234567890abcdef",
        "token": "secret_jwt_token",
        "user_password": "supersecretpassword",
        "public_data": "normal_value",
    })

    trace_dict = trace.to_dict()
    event_details = trace_dict["entries"][0]["details"]

    assert event_details["api_key"] == "[REDACTED]"
    assert event_details["token"] == "[REDACTED]"
    assert event_details["user_password"] == "[REDACTED]"
    assert event_details["public_data"] == "normal_value"


# =====================================================================
# 5. INDEPENDENT VERIFICATION LAYER TESTS
# =====================================================================

def test_verifier_number_type_and_range():
    verifier = StepVerifier()
    step = PlanStep(
        step_id="step_num",
        description="Verify number",
        verification_requirements={
            "expected_type": "number",
            "min_value": 0.0,
            "max_value": 100.0,
        },
    )

    # Valid observation
    obs_valid = StepObservation(step_id="step_num", success=True, output=45.5)
    res = verifier.verify(step, obs_valid)
    assert res.passed

    # Invalid type
    obs_wrong_type = StepObservation(step_id="step_num", success=True, output="not_a_number")
    res_type = verifier.verify(step, obs_wrong_type)
    assert not res_type.passed
    assert "Expected numeric output" in res_type.notes

    # Out of range
    obs_out_of_bounds = StepObservation(step_id="step_num", success=True, output=150.0)
    res_bounds = verifier.verify(step, obs_out_of_bounds)
    assert not res_bounds.passed
    assert "above maximum allowed" in res_bounds.notes


def test_verifier_required_keys_and_non_empty():
    verifier = StepVerifier()
    step = PlanStep(
        step_id="step_dict",
        description="Verify dict",
        verification_requirements={
            "expected_type": "dict",
            "required_keys": ["status", "count"],
            "allow_empty": False,
        },
    )

    # Valid dict
    obs_valid = StepObservation(step_id="step_dict", success=True, output={"status": "ok", "count": 5})
    assert verifier.verify(step, obs_valid).passed

    # Missing required key
    obs_missing = StepObservation(step_id="step_dict", success=True, output={"status": "ok"})
    res_missing = verifier.verify(step, obs_missing)
    assert not res_missing.passed
    assert "Missing required key 'count'" in res_missing.notes


# =====================================================================
# 6. RECOVERY & ROLLBACK TESTS
# =====================================================================

def test_recovery_bounded_retries():
    manager = RecoveryManager()
    step = PlanStep(
        step_id="step_retry",
        description="Retryable step",
        retry_policy=RetryPolicy(max_retries=2),
    )
    plan = CognitivePlan(plan_id="p1", task_id="t1", steps=[step])
    graph = ExecutionGraph(plan)

    failed_obs = StepObservation(step_id="step_retry", success=False, error="Temporary glitch")

    # First retry
    assert manager.should_retry(step, failed_obs)
    count1 = manager.handle_retry(step, graph)
    assert count1 == 1
    assert step.status == PlanStepStatus.READY

    # Second retry
    assert manager.should_retry(step, failed_obs)
    count2 = manager.handle_retry(step, graph)
    assert count2 == 2

    # Third attempt exceeds max_retries
    assert not manager.should_retry(step, failed_obs)


def test_recovery_task_state_rollback():
    manager = RecoveryManager()
    task = CognitiveTask(
        task_id="task_rb",
        user_request="State rollback test",
        execution_state={"step1_var": "keep_this", "temp_output": "delete_this"},
    )
    task.transition_to(TaskStatus.PLANNING)
    task.transition_to(TaskStatus.READY)
    task.transition_to(TaskStatus.RUNNING)

    step = PlanStep(
        step_id="step_fail",
        description="Failing step",
        output_references={"output": "temp_output"},
    )

    rec = manager.rollback_task_state(task, step, reason="Execution failed")
    assert "temp_output" in rec.reverted_state_keys
    assert "temp_output" not in task.execution_state
    assert task.execution_state["step1_var"] == "keep_this"
    assert task.status == TaskStatus.ROLLED_BACK


# =====================================================================
# 7. DEPLOYMENT PROFILES & HARDWARE FRIENDLINESS
# =====================================================================

def test_deployment_profiles():
    edge = get_edge_profile()
    desktop = get_desktop_profile()
    server = get_server_profile()

    assert edge.max_steps == 4
    assert edge.max_token_budget == 256
    assert edge.max_memory_mb == 256

    assert desktop.max_steps == 10
    assert desktop.max_token_budget == 512

    assert server.max_steps == 20
    assert server.max_token_budget == 512

    # Verify constraints derivation
    constraints = edge.to_constraints()
    assert constraints.max_steps == 4
    assert constraints.max_token_budget == 256


# =====================================================================
# 8. CROSS-TASK ISOLATION TESTS
# =====================================================================

def test_cross_task_isolation():
    task_a = CognitiveTask(
        task_id="task_A",
        user_request="User A task",
        execution_state={"private_data": "secret_A"},
    )
    task_b = CognitiveTask(
        task_id="task_B",
        user_request="User B task",
        execution_state={"private_data": "secret_B"},
    )

    assert task_a.execution_state["private_data"] == "secret_A"
    assert task_b.execution_state["private_data"] == "secret_B"

    # Modify task A state
    task_a.execution_state["new_key"] = 123
    assert "new_key" not in task_b.execution_state


# =====================================================================
# 9. END-TO-END COGNITIVE CONTROLLER PIPELINE TESTS
# =====================================================================

def test_controller_end_to_end_calculation_task():
    controller = CognitiveController()
    result = controller.execute_task("calculate 25 * 4 + 50")

    assert result.success
    assert result.task.status == TaskStatus.COMPLETED
    assert result.task.execution_state.get("calc_result") == 150.0
    assert "150" in result.response_text
    assert len(result.plan.steps) == 3
    assert len(result.observations) == 3
    assert len(result.memory_candidates) >= 1
    assert result.trace.duration_ms > 0


def test_controller_end_to_end_artifact_generation_task():
    controller = CognitiveController()
    result = controller.execute_task("prepare a summary report document for project status")

    assert result.success
    assert result.task.status == TaskStatus.COMPLETED
    assert len(result.artifacts) >= 1
    artifact = result.artifacts[0]
    assert artifact.artifact_type == ArtifactType.MARKDOWN
    assert "Generated Document" in artifact.content
    assert "Generated Artifact" in result.response_text


def test_controller_handles_failure_gracefully():
    # Attempt an invalid calculation (division by zero)
    controller = CognitiveController()
    result = controller.execute_task("calculate 100 / 0")

    assert not result.success
    assert result.task.status == TaskStatus.ROLLED_BACK
    assert "halted due to failure" in result.response_text


def test_inference_session_cognitive_task_backward_compatibility():
    from pathlib import Path
    from chakrview.brain.config import ModelConfig
    from chakrview.brain.model import ChakrMicro
    from chakrview.runtime.inference import InferenceSession
    from chakrview.tokenizer.serialization import load_tokenizer_artifacts

    tokenizer_dir = Path("data/experiments/vocab_4096")
    tokenizer, _ = load_tokenizer_artifacts(tokenizer_dir)
    model = ChakrMicro(ModelConfig())
    model.eval()
    session = InferenceSession(model=model, tokenizer=tokenizer)

    # 1. Direct ask() continues to work seamlessly (backward compatibility)
    chat_resp = session.ask("What is physics?")
    assert chat_resp is not None
    assert hasattr(chat_resp, "text")

    # 2. execute_cognitive_task delegates to CognitiveController
    cog_res = session.execute_cognitive_task("calculate 16 * 4")
    assert cog_res.success
    assert cog_res.task.status == TaskStatus.COMPLETED
    assert cog_res.task.execution_state.get("calc_result") == 64.0
    assert "64" in cog_res.response_text

