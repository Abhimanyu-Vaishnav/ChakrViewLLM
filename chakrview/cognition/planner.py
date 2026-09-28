"""
Bounded Cognitive Planner for ChakrView (Step 15).

Provides deterministic planning abstractions with strict depth and step limits,
dependency tracking, tool/skill mapping, retry policies, and verification requirements.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum
import time
from typing import Dict, List, Optional, Any, Set, Tuple
import uuid

from chakrview.cognition.task import CognitiveTask, TaskConstraints
from chakrview.runtime.skills import Skill, SkillRegistry, SkillPolicy


class PlanStepStatus(str, Enum):
    """Lifecycle status of a planned step in an execution plan."""
    PENDING = "PENDING"
    READY = "READY"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    BLOCKED = "BLOCKED"


@dataclass
class RetryPolicy:
    """
    Retry configuration for an individual step.
    """
    max_retries: int = 2
    retry_count: int = 0
    retry_delay_seconds: float = 0.0

    def can_retry(self) -> bool:
        return self.retry_count < self.max_retries

    def record_retry(self) -> int:
        self.retry_count += 1
        return self.retry_count

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetryPolicy":
        return cls(**data)


@dataclass
class PlanStep:
    """
    An individual bounded execution step within a CognitivePlan.

    Attributes:
        step_id: Unique identifier for this step.
        description: Functional description of what this step does.
        dependencies: List of step_ids that must complete before this step runs.
        required_skill: Optional skill ID associated with this step.
        required_tool: Optional tool ID required to execute this step.
        tool_arguments: Explicit arguments to pass to the tool.
        status: Current step execution status.
        input_references: Keys in task execution state used as inputs.
        output_references: Keys in task execution state where results will be written.
        retry_policy: Bounded retry configuration.
        verification_requirements: Specification of expected output checks.
        metadata: Additional diagnostic or domain metadata.
    """
    step_id: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    required_skill: Optional[str] = None
    required_tool: Optional[str] = None
    tool_arguments: Dict[str, Any] = field(default_factory=dict)
    status: PlanStepStatus = PlanStepStatus.PENDING
    input_references: Dict[str, str] = field(default_factory=dict)
    output_references: Dict[str, str] = field(default_factory=dict)
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    verification_requirements: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        data["retry_policy"] = self.retry_policy.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlanStep":
        d = dict(data)
        d["status"] = PlanStepStatus(d["status"])
        if isinstance(d.get("retry_policy"), dict):
            d["retry_policy"] = RetryPolicy.from_dict(d["retry_policy"])
        return cls(**d)


@dataclass
class CognitivePlan:
    """
    A bounded, ordered multi-step execution plan for a CognitiveTask.

    Attributes:
        plan_id: Unique identifier for this plan.
        task_id: Associated task identifier.
        steps: Ordered list of PlanStep objects.
        created_at: Epoch timestamp of creation.
        metadata: Extra plan metadata.
    """
    plan_id: str
    task_id: str
    steps: List[PlanStep] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_step(self, step_id: str) -> Optional[PlanStep]:
        for s in self.steps:
            if s.step_id == step_id:
                return s
        return None

    def total_steps(self) -> int:
        return len(self.steps)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "task_id": self.task_id,
            "steps": [s.to_dict() for s in self.steps],
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CognitivePlan":
        d = dict(data)
        d["steps"] = [PlanStep.from_dict(s) for s in d.get("steps", [])]
        return cls(**d)


class PlanningError(ValueError):
    """Raised when a plan cannot be generated or exceeds bounded constraints."""
    pass


class BoundedPlanner(ABC):
    """
    Abstract base for bounded cognitive planners.
    Enforces maximum planning depth, maximum steps, and safe recursion limits.
    """

    def __init__(self, default_constraints: Optional[TaskConstraints] = None) -> None:
        self.default_constraints = default_constraints or TaskConstraints()

    @abstractmethod
    def plan(
        self,
        task: CognitiveTask,
        active_skill: Optional[Skill] = None,
        available_tools: Optional[List[str]] = None,
    ) -> CognitivePlan:
        """Generate a bounded execution plan for the given task."""
        pass

    def validate_plan_bounds(self, plan: CognitivePlan, constraints: TaskConstraints) -> None:
        """
        Ensure plan strictly obeys maximum steps and depth.
        """
        if len(plan.steps) > constraints.max_steps:
            raise PlanningError(
                f"Plan step count ({len(plan.steps)}) exceeds task limit ({constraints.max_steps})."
            )

        # Compute maximum dependency depth
        step_map = {s.step_id: s for s in plan.steps}
        memo: Dict[str, int] = {}

        def get_depth(step_id: str, visited: Set[str]) -> int:
            if step_id in visited:
                raise PlanningError(f"Cycle detected in plan dependencies at step '{step_id}'.")
            if step_id in memo:
                return memo[step_id]

            step = step_map.get(step_id)
            if not step or not step.dependencies:
                memo[step_id] = 1
                return 1

            visited.add(step_id)
            max_dep = 0
            for dep in step.dependencies:
                if dep in step_map:
                    max_dep = max(max_dep, get_depth(dep, visited.copy()))
            visited.remove(step_id)

            memo[step_id] = 1 + max_dep
            return memo[step_id]

        max_depth = 0
        for step in plan.steps:
            max_depth = max(max_depth, get_depth(step.step_id, set()))

        if max_depth > constraints.max_depth:
            raise PlanningError(
                f"Plan depth ({max_depth}) exceeds task limit ({constraints.max_depth})."
            )


class DeterministicRulePlanner(BoundedPlanner):
    """
    Deterministic rule-based planner that constructs bounded execution plans
    without relying on non-deterministic external LLMs or arbitrary recursion.
    """

    def plan(
        self,
        task: CognitiveTask,
        active_skill: Optional[Skill] = None,
        available_tools: Optional[List[str]] = None,
    ) -> CognitivePlan:
        constraints = task.constraints or self.default_constraints
        plan_id = f"plan_{uuid.uuid4().hex[:8]}"
        req = task.user_request.strip().lower()
        tools = set(available_tools or [])
        if active_skill and active_skill.policy:
            tools.update(active_skill.policy.allowed_tools)

        steps: List[PlanStep] = []

        # Check for calculation requests
        is_calc = (
            "calculate" in req or
            ("+" in req or "-" in req or "*" in req or "/" in req) and
            not any(w in req for w in ["what", "who", "why", "how", "when", "explain"])
        )

        # Check for report / artifact generation
        is_artifact = any(w in req for w in ["report", "document", "artifact", "summary sheet", "file"])

        # Check for retrieval requirement
        needs_retrieval = (
            active_skill.policy.requires_knowledge if (active_skill and active_skill.policy)
            else any(w in req for w in ["search", "find", "retrieve", "lookup", "knowledge", "history", "record"])
        )

        step_idx = 1

        if is_calc and "calculator" in tools:
            # Step 1: Execute Calculator Tool
            expr = task.user_request
            if "calculate" in req:
                parts = task.user_request.split("calculate", 1)
                if len(parts) > 1 and parts[1].strip():
                    expr = parts[1].strip()
            
            s1 = PlanStep(
                step_id=f"step_{step_idx}",
                description="Evaluate mathematical expression safely via CalculatorTool",
                required_skill=active_skill.skill_id if active_skill else None,
                required_tool="calculator",
                tool_arguments={"expression": expr},
                retry_policy=RetryPolicy(max_retries=min(2, constraints.max_retries)),
                verification_requirements={"expected_type": "number"},
                output_references={"result": "calc_result"},
            )
            steps.append(s1)
            step_idx += 1

            # Step 2: Verify Result
            s2 = PlanStep(
                step_id=f"step_{step_idx}",
                description="Verify mathematical calculation output",
                dependencies=[s1.step_id],
                input_references={"result": "calc_result"},
                verification_requirements={"allow_empty": False},
                retry_policy=RetryPolicy(max_retries=1),
                output_references={"verified": "verified_result"},
            )
            steps.append(s2)
            step_idx += 1

            # Step 3: Synthesize Response
            s3 = PlanStep(
                step_id=f"step_{step_idx}",
                description="Synthesize final mathematical answer",
                dependencies=[s2.step_id],
                input_references={"verified": "verified_result"},
                output_references={"final_text": "response_text"},
            )
            steps.append(s3)

        elif is_artifact:
            # Multi-step artifact workflow: Retrieve/Prepare -> Draft Artifact -> Verify Artifact -> Finalize
            s1 = PlanStep(
                step_id=f"step_{step_idx}",
                description="Gather context and requirements for artifact generation",
                required_skill=active_skill.skill_id if active_skill else None,
                retry_policy=RetryPolicy(max_retries=min(2, constraints.max_retries)),
                output_references={"context": "artifact_context"},
            )
            steps.append(s1)
            step_idx += 1

            s2 = PlanStep(
                step_id=f"step_{step_idx}",
                description="Draft structured artifact content",
                dependencies=[s1.step_id],
                input_references={"context": "artifact_context"},
                verification_requirements={"expected_type": "str", "allow_empty": False},
                output_references={"artifact_content": "draft_content"},
            )
            steps.append(s2)
            step_idx += 1

            s3 = PlanStep(
                step_id=f"step_{step_idx}",
                description="Verify artifact completeness and structural formatting",
                dependencies=[s2.step_id],
                input_references={"artifact_content": "draft_content"},
                verification_requirements={"allow_empty": False},
                output_references={"verified_artifact": "final_artifact"},
            )
            steps.append(s3)
            step_idx += 1

            s4 = PlanStep(
                step_id=f"step_{step_idx}",
                description="Prepare response summary and present artifact",
                dependencies=[s3.step_id],
                input_references={"verified_artifact": "final_artifact"},
                output_references={"final_text": "response_text"},
            )
            steps.append(s4)

        elif needs_retrieval:
            # Retrieval -> Analysis -> Synthesis
            s1 = PlanStep(
                step_id=f"step_{step_idx}",
                description="Retrieve relevant knowledge and memory records",
                required_skill=active_skill.skill_id if active_skill else None,
                retry_policy=RetryPolicy(max_retries=min(2, constraints.max_retries)),
                output_references={"retrieved_data": "retrieved_context"},
            )
            steps.append(s1)
            step_idx += 1

            s2 = PlanStep(
                step_id=f"step_{step_idx}",
                description="Analyze retrieved information and verify relevance",
                dependencies=[s1.step_id],
                input_references={"retrieved_data": "retrieved_context"},
                output_references={"analysis": "verified_analysis"},
            )
            steps.append(s2)
            step_idx += 1

            s3 = PlanStep(
                step_id=f"step_{step_idx}",
                description="Synthesize final verified response",
                dependencies=[s2.step_id],
                input_references={"analysis": "verified_analysis"},
                output_references={"final_text": "response_text"},
            )
            steps.append(s3)

        else:
            # Standard bounded 2-step plan: Reason/Execute -> Synthesize
            s1 = PlanStep(
                step_id=f"step_{step_idx}",
                description="Analyze user request and execute core logic",
                required_skill=active_skill.skill_id if active_skill else None,
                output_references={"execution_data": "step1_output"},
            )
            steps.append(s1)
            step_idx += 1

            s2 = PlanStep(
                step_id=f"step_{step_idx}",
                description="Synthesize grounded response",
                dependencies=[s1.step_id],
                input_references={"execution_data": "step1_output"},
                output_references={"final_text": "response_text"},
            )
            steps.append(s2)

        plan = CognitivePlan(
            plan_id=plan_id,
            task_id=task.task_id,
            steps=steps,
            metadata={"planner": "DeterministicRulePlanner"},
        )

        self.validate_plan_bounds(plan, constraints)
        return plan
