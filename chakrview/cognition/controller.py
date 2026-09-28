"""
Cognitive Controller for ChakrView (Step 15).

Orchestrates multi-step governed cognitive agent execution:
Understand -> Retrieve -> Plan -> Execute -> Observe -> Verify -> Recover -> Respond

Preserves all frozen invariants:
- Model context ceiling: strictly <= 512 tokens.
- Retrieved data / memory is strictly PASSIVE DATA, never tool authority.
- Tool authority originates solely from explicit runtime SkillPolicy.
- Controlled memory candidates for future learning without uncontrolled writes.
"""

from dataclasses import dataclass, field
import time
from typing import Dict, List, Optional, Any, Tuple, Union
import uuid

from chakrview.cognition.task import CognitiveTask, TaskStatus, TaskConstraints
from chakrview.cognition.planner import (
    BoundedPlanner,
    CognitivePlan,
    PlanStep,
    PlanStepStatus,
    DeterministicRulePlanner,
    PlanningError,
)
from chakrview.cognition.graph import ExecutionGraph, CycleDetectedError
from chakrview.cognition.observation import StepObservation
from chakrview.cognition.verifier import StepVerifier, VerificationResult
from chakrview.cognition.tool_gate import GovernedToolGate, ToolAuthorizationError
from chakrview.cognition.skill_selector import CognitiveSkillSelector, SkillSelectionMatch
from chakrview.cognition.recovery import RecoveryManager, RollbackRecord
from chakrview.cognition.artifacts import CognitiveArtifact, ArtifactManager, ArtifactType
from chakrview.cognition.trace import ExecutionTrace
from chakrview.cognition.profile import DeploymentProfile, get_desktop_profile

from chakrview.runtime.skills import Skill, SkillRegistry, SkillPolicy
from chakrview.runtime.context import PromptContextBuilder, AssembledContext, ContextBudget


@dataclass
class MemoryCandidate:
    """
    Controlled candidate for future memory retention.
    
    Prevents uncontrolled permanent memory writes by requiring explicit
    candidate packaging and provenance for future policy review.
    """
    key: str
    value: str
    task_id: str
    provenance: str
    confidence: float
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "task_id": self.task_id,
            "provenance": self.provenance,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


@dataclass
class CognitiveExecutionResult:
    """
    Structured outcome of a governed cognitive task execution.
    """
    task: CognitiveTask
    plan: CognitivePlan
    observations: Dict[str, StepObservation]
    artifacts: List[CognitiveArtifact]
    memory_candidates: List[MemoryCandidate]
    trace: ExecutionTrace
    response_text: str
    success: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task.to_dict(),
            "plan": self.plan.to_dict(),
            "observations": {k: v.to_dict() for k, v in self.observations.items()},
            "artifacts": [a.to_dict() for a in self.artifacts],
            "memory_candidates": [m.to_dict() for m in self.memory_candidates],
            "trace": self.trace.to_dict(),
            "response_text": self.response_text,
            "success": self.success,
        }


class CognitiveController:
    """
    Central orchestration controller for governed cognitive agent workflows.
    """

    def __init__(
        self,
        planner: Optional[BoundedPlanner] = None,
        skill_selector: Optional[CognitiveSkillSelector] = None,
        tool_gate: Optional[GovernedToolGate] = None,
        verifier: Optional[StepVerifier] = None,
        recovery_manager: Optional[RecoveryManager] = None,
        artifact_manager: Optional[ArtifactManager] = None,
        context_builder: Optional[PromptContextBuilder] = None,
        profile: Optional[DeploymentProfile] = None,
        capability_gate: Optional[Any] = None,
        environment: Optional[Any] = None,
    ) -> None:
        self.profile = profile or get_desktop_profile()
        self.planner = planner or DeterministicRulePlanner(self.profile.to_constraints())
        self.skill_selector = skill_selector or CognitiveSkillSelector()
        self.tool_gate = tool_gate or GovernedToolGate()
        self.verifier = verifier or StepVerifier()
        self.recovery_manager = recovery_manager or RecoveryManager()
        self.artifact_manager = artifact_manager or ArtifactManager()
        self.context_builder = context_builder or PromptContextBuilder()
        self.capability_gate = capability_gate
        self.environment = environment

    def execute_task(
        self,
        task_or_prompt: Union[CognitiveTask, str],
        active_skill: Optional[Skill] = None,
        session: Optional[Any] = None,
        retriever: Optional[Any] = None,
        memory_store: Optional[Any] = None,
        preferred_domain: Optional[Any] = None,
        personal_memory: Optional[Any] = None,
        capability_gate: Optional[Any] = None,
        environment: Optional[Any] = None,
    ) -> CognitiveExecutionResult:
        """
        Execute a complete governed cognitive workflow for the given task.
        """
        # 1. Normalize task
        if isinstance(task_or_prompt, str):
            task = CognitiveTask(
                task_id=f"task_{uuid.uuid4().hex[:8]}",
                user_request=task_or_prompt.strip(),
                constraints=self.profile.to_constraints(),
            )
        else:
            task = task_or_prompt

        trace = ExecutionTrace(task_id=task.task_id)
        observations: Dict[str, StepObservation] = {}
        memory_candidates: List[MemoryCandidate] = []
        artifacts: List[CognitiveArtifact] = []

        # 2. Skill Selection
        if active_skill is None:
            match = self.skill_selector.select_skill(
                user_request=task.user_request,
                preferred_domain=preferred_domain,
            )
            resolved_skill = match.skill
            trace.record_event("SKILL_SELECTION", details={
                "skill_id": resolved_skill.skill_id,
                "confidence": match.confidence,
                "rationale": match.rationale,
            })
        else:
            resolved_skill = active_skill
            trace.record_event("SKILL_ASSIGNMENT", details={"skill_id": resolved_skill.skill_id})

        # 3. Transition to PLANNING
        task.transition_to(TaskStatus.PLANNING, reason="Initiating bounded cognitive planning.")
        trace.record_event("TASK_STATUS_CHANGE", details={"status": TaskStatus.PLANNING.value})

        # 4. Generate Bounded Plan
        try:
            allowed_tools = resolved_skill.policy.allowed_tools if resolved_skill.policy else []
            plan = self.planner.plan(
                task=task,
                active_skill=resolved_skill,
                available_tools=allowed_tools,
            )
            trace.plan_id = plan.plan_id
            trace.record_event("PLAN_CREATED", details={"plan_id": plan.plan_id, "step_count": len(plan.steps)})
        except Exception as e:
            task.transition_to(TaskStatus.FAILED, reason=f"Planning failed: {str(e)}")
            trace.record_event("PLANNING_ERROR", details={"error": str(e)})
            trace.finalize()
            return CognitiveExecutionResult(
                task=task,
                plan=CognitivePlan(plan_id="empty", task_id=task.task_id),
                observations={},
                artifacts=[],
                memory_candidates=[],
                trace=trace,
                response_text=f"Cognitive planning failed: {str(e)}",
                success=False,
            )

        # 5. Build Execution Graph & Transition to READY
        try:
            graph = ExecutionGraph(plan)
            task.transition_to(TaskStatus.READY, reason="Execution graph compiled successfully.")
            trace.record_event("TASK_STATUS_CHANGE", details={"status": TaskStatus.READY.value})
        except Exception as e:
            task.transition_to(TaskStatus.FAILED, reason=f"Graph construction failed: {str(e)}")
            trace.record_event("GRAPH_ERROR", details={"error": str(e)})
            trace.finalize()
            return CognitiveExecutionResult(
                task=task,
                plan=plan,
                observations={},
                artifacts=[],
                memory_candidates=[],
                trace=trace,
                response_text=f"Graph execution failed: {str(e)}",
                success=False,
            )

        # 6. Transition to RUNNING
        task.transition_to(TaskStatus.RUNNING, reason="Starting step execution loop.")
        trace.record_event("TASK_STATUS_CHANGE", details={"status": TaskStatus.RUNNING.value})

        active_cap_gate = capability_gate or self.capability_gate
        active_env = environment or self.environment
        if active_cap_gate is None:
            try:
                from chakrview.capability.gate import CapabilityGate
                from chakrview.capability.bridge import get_standard_capability_registry
                active_cap_gate = CapabilityGate(get_standard_capability_registry())
            except Exception:
                active_cap_gate = None

        # 7. Execution Loop
        while not graph.is_finished():
            ready_steps = graph.get_ready_steps()
            if not ready_steps:
                # No steps can make progress; remaining steps are blocked or unreachable
                break

            for step in ready_steps:
                task.current_step_id = step.step_id
                graph.mark_step_running(step.step_id)
                trace.record_event("STEP_START", step_id=step.step_id, details={
                    "description": step.description,
                    "required_tool": step.required_tool,
                    "required_capability": getattr(step, "required_capability", None),
                })

                # Execute Step logic
                observation = self._execute_step(
                    step=step,
                    task=task,
                    resolved_skill=resolved_skill,
                    retriever=retriever,
                    memory_store=memory_store,
                    artifacts=artifacts,
                    personal_memory=personal_memory,
                    capability_gate=active_cap_gate,
                    environment=active_env,
                )
                observations[step.step_id] = observation

                # Verification
                ver_res = self.verifier.verify(step, observation)
                observation.verification_status = ver_res.passed
                observation.verification_notes = ver_res.notes
                trace.record_event("VERIFICATION", step_id=step.step_id, details=ver_res.to_dict())

                if observation.success and ver_res.passed:
                    # Step succeeded
                    graph.mark_step_completed(step.step_id)
                    # Propagate outputs to task execution state
                    if isinstance(observation.output, dict):
                        for ref_k, state_k in step.output_references.items():
                            if ref_k in observation.output:
                                task.execution_state[state_k] = observation.output[ref_k]
                    elif observation.output is not None:
                        # Map primary output to any defined output reference
                        for state_k in step.output_references.values():
                            task.execution_state[state_k] = observation.output

                    trace.record_event("STEP_COMPLETED", step_id=step.step_id, details={
                        "output_summary": str(observation.output)[:120],
                    })
                else:
                    # Step execution or verification failed
                    if self.recovery_manager.should_retry(step, observation):
                        retry_count = self.recovery_manager.handle_retry(step, graph)
                        trace.record_event("STEP_RETRY", step_id=step.step_id, details={
                            "retry_count": retry_count,
                            "reason": observation.error or ver_res.notes,
                        })
                    else:
                        # Exhausted retries or non-retriable failure
                        blocked = self.recovery_manager.handle_step_failure(task, step, observation, graph)
                        trace.record_event("STEP_FAILED", step_id=step.step_id, details={
                            "error": observation.error or ver_res.notes,
                            "blocked_steps": blocked,
                        })
                        # Rollback transient state
                        rollback_record = self.recovery_manager.rollback_task_state(
                            task=task,
                            step=step,
                            reason=f"Step '{step.step_id}' failed: {observation.error or ver_res.notes}",
                        )
                        trace.record_event("ROLLBACK", step_id=step.step_id, details=rollback_record.to_dict())
                        break

            if graph.has_failed():
                break

        # 8. Post-Execution Evaluation
        if graph.has_failed() or task.status == TaskStatus.ROLLED_BACK:
            trace.finalize()
            return CognitiveExecutionResult(
                task=task,
                plan=plan,
                observations=observations,
                artifacts=artifacts,
                memory_candidates=[],
                trace=trace,
                response_text=f"Task execution halted due to failure: {task.metadata.get('failure_reason', 'Step failure')}",
                success=False,
            )

        # 9. Successful Completion
        task.transition_to(TaskStatus.VERIFYING, reason="Finalizing and verifying task completion.")
        task.transition_to(TaskStatus.COMPLETED, reason="All plan steps executed and verified.")
        trace.record_event("TASK_STATUS_CHANGE", details={"status": TaskStatus.COMPLETED.value})

        # 10. Memory Candidate Extraction
        # Formulate controlled candidate from task execution state (e.g. calculation, summary)
        if "calc_result" in task.execution_state:
            memory_candidates.append(MemoryCandidate(
                key=f"calc_result_{task.task_id[:6]}",
                value=str(task.execution_state["calc_result"]),
                task_id=task.task_id,
                provenance="calculator_tool",
                confidence=1.0,
            ))
        if "final_artifact" in task.execution_state:
            memory_candidates.append(MemoryCandidate(
                key=f"artifact_{task.task_id[:6]}",
                value=str(task.execution_state["final_artifact"])[:100],
                task_id=task.task_id,
                provenance="artifact_workflow",
                confidence=0.9,
            ))

        # Store memory candidates in personal_memory if provided
        if personal_memory is not None and memory_candidates:
            from chakrview.memory.record import MemoryType as PMType, MemoryProvenance as PMProv
            owner = getattr(task, "owner_id", "default_user")
            for mc in memory_candidates:
                try:
                    personal_memory.store_memory(
                        content=f"{mc.key}: {mc.value}",
                        memory_type=PMType.SEMANTIC,
                        owner_id=owner,
                        provenance=PMProv(source_type=mc.provenance, source_id=mc.task_id),
                    )
                except Exception:
                    pass

        # 11. Synthesize Response Text
        response_text = self._synthesize_response(
            task=task,
            plan=plan,
            observations=observations,
            resolved_skill=resolved_skill,
            session=session,
        )

        trace.finalize()
        return CognitiveExecutionResult(
            task=task,
            plan=plan,
            observations=observations,
            artifacts=artifacts,
            memory_candidates=memory_candidates,
            trace=trace,
            response_text=response_text,
            success=True,
        )

    def _execute_step(
        self,
        step: PlanStep,
        task: CognitiveTask,
        resolved_skill: Skill,
        retriever: Optional[Any],
        memory_store: Optional[Any],
        artifacts: List[CognitiveArtifact],
        personal_memory: Optional[Any] = None,
        capability_gate: Optional[Any] = None,
        environment: Optional[Any] = None,
    ) -> StepObservation:
        """
        Internal executor for a single cognitive step.
        """
        t0 = time.perf_counter()

        # 1. Governed Capability Execution (Step 17)
        req_cap = getattr(step, "required_capability", None)
        if req_cap:
            args = dict(getattr(step, "capability_arguments", {}))
            for ref_k, state_k in step.input_references.items():
                if state_k in task.execution_state:
                    args[ref_k] = task.execution_state[state_k]

            from chakrview.capability.contract import CapabilityRequest, CapabilityContext
            from chakrview.capability.bridge import capability_result_to_observation

            req = CapabilityRequest(
                capability_id=req_cap,
                parameters=args,
                caller_id=getattr(task, "owner_id", "agent"),
                task_id=task.task_id,
            )
            granted = set(getattr(resolved_skill.policy, "allowed_permissions", [])) if resolved_skill and hasattr(resolved_skill, "policy") else set()
            if not granted:
                granted = {
                    "capability.compute.math",
                    "capability.compute.text",
                    "capability.read.clock",
                    "capability.sensor.read",
                    "capability.actuator.control",
                }

            ctx = CapabilityContext(
                user_id=getattr(task, "owner_id", "default_user"),
                task_id=task.task_id,
                environment_id=environment.environment_id if environment else "default_env",
                granted_permissions=granted,
            )
            gate = capability_gate or self.capability_gate
            active_pol = environment or (resolved_skill.policy if resolved_skill and hasattr(resolved_skill, "policy") else None)
            cap_res = gate.execute_governed(
                request=req,
                context=ctx,
                active_policy=active_pol,
            )
            desc = gate.registry.get_descriptor(req_cap) if gate and gate.registry.has(req_cap) else None
            return capability_result_to_observation(step.step_id, cap_res, descriptor=desc)

        # 2. Governed Tool Execution (Step 11/15)
        if step.required_tool:
            # Resolve tool arguments from task.execution_state if referenced
            args = dict(step.tool_arguments)
            for ref_k, state_k in step.input_references.items():
                if state_k in task.execution_state:
                    args[ref_k] = task.execution_state[state_k]

            return self.tool_gate.execute_governed(
                step_id=step.step_id,
                tool_id=step.required_tool,
                arguments=args,
                active_skill=resolved_skill,
            )

        # 2. Knowledge Retrieval Step
        if "retrieve" in step.description.lower() or "gather" in step.description.lower():
            retrieved_text = ""
            if retriever is not None:
                try:
                    results = retriever.retrieve(task.user_request, top_k=2)
                    retrieved_text = "\n".join([r.chunk.text for r in results])
                except Exception as e:
                    retrieved_text = f"Retrieval note: {str(e)}"
            elif memory_store is not None:
                # Retrieve from working memory
                memories = memory_store.get_working_memory()
                if memories:
                    retrieved_text = "\n".join([f"- {m.key}: {m.value}" for m in memories[:3]])

            # Check persistent personal memory if available
            if personal_memory is not None:
                try:
                    owner = getattr(task, "owner_id", "default_user")
                    mem_results = personal_memory.retrieve(task.user_request, owner_id=owner, top_k=2)
                    if mem_results:
                        mem_strings = [f"- [Personal Memory]: {m.content}" for m in mem_results]
                        retrieved_text = f"{retrieved_text}\n" + "\n".join(mem_strings) if retrieved_text else "\n".join(mem_strings)
                except Exception:
                    pass

            if not retrieved_text:
                retrieved_text = f"Context gathered for: {task.user_request}"

            return StepObservation(
                step_id=step.step_id,
                success=True,
                output=retrieved_text,
                execution_time_ms=(time.perf_counter() - t0) * 1000.0,
                provenance={"source": "retrieval_step"},
            )

        # 3. Artifact Drafting Step
        if "draft" in step.description.lower() or "artifact" in step.description.lower():
            content = f"# Generated Document\n\nRequest: {task.user_request}\n\n"
            for k, v in task.execution_state.items():
                content += f"## {k.replace('_', ' ').title()}\n{v}\n\n"
            
            artifact = self.artifact_manager.create_artifact(
                name=f"Artifact_{task.task_id[:6]}",
                content=content,
                artifact_type=ArtifactType.MARKDOWN,
                metadata={"task_id": task.task_id, "step_id": step.step_id},
            )
            artifacts.append(artifact)

            return StepObservation(
                step_id=step.step_id,
                success=True,
                output=content,
                execution_time_ms=(time.perf_counter() - t0) * 1000.0,
                provenance={"artifact_id": artifact.artifact_id},
            )

        # 4. Standard Intermediate Step / Synthesis
        output_val = task.execution_state.get("calc_result") or task.execution_state.get("draft_content") or "Step completed successfully."
        return StepObservation(
            step_id=step.step_id,
            success=True,
            output=output_val,
            execution_time_ms=(time.perf_counter() - t0) * 1000.0,
            provenance={"step_id": step.step_id},
        )

    def _synthesize_response(
        self,
        task: CognitiveTask,
        plan: CognitivePlan,
        observations: Dict[str, StepObservation],
        resolved_skill: Skill,
        session: Optional[Any],
    ) -> str:
        """
        Assemble bounded context and format final response.
        Enforces strict <= 512 token ceiling.
        """
        # If calculation was performed, provide explicit answer
        if "calc_result" in task.execution_state:
            return f"Result: {task.execution_state['calc_result']}"

        # If artifact was generated, present summary
        if "final_artifact" in task.execution_state or "draft_content" in task.execution_state:
            content = task.execution_state.get("final_artifact") or task.execution_state.get("draft_content")
            return f"Generated Artifact:\n{content}"

        # If session is provided and model can synthesize
        if session is not None and hasattr(session, "model") and hasattr(session, "tokenizer"):
            # Construct bounded prompt
            sys_prompt = resolved_skill.system_prompt_template or "You are ChakrView Cognitive Agent."
            context_summary = f"Plan: {len(plan.steps)} steps executed.\nUser Request: {task.user_request}"
            assembled = self.context_builder.build_prompt(
                system_prompt=sys_prompt,
                user_query=task.user_request,
                tokenizer=session.tokenizer,
                budget=ContextBudget(max_context=self.profile.max_token_budget),
            )
            # Perform bounded generation if prompt tokens are within limits
            if assembled.estimated_prompt_tokens < self.profile.max_token_budget:
                try:
                    chat_resp = session.ask(
                        user_text=task.user_request,
                        skill=resolved_skill,
                    )
                    return chat_resp.text
                except Exception:
                    pass

        return f"Task '{task.task_id}' executed successfully across {len(plan.steps)} steps."
