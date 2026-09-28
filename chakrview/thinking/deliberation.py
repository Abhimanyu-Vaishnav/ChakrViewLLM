"""
Deliberation Engine for ChakrView (Step 21).

Coordinates the iterative neuro-symbolic deliberation loop:
ChakrMicro Neural Core <-> Context Builder <-> Thinking Workspace
<-> Governed Reasoning <-> Critique Engine <-> Revision Engine <-> Stopping Policy.

Architectural Principles:
1. ChakrMicro v0.1 remains strictly frozen (3,443,136 parameters, CPU-first).
2. Weights are permanently immutable during deliberation (weights_modified = False).
3. Pure computational deliberation: NO claims of consciousness, sentience, or AGI.
4. Bounded execution: Hard limits on steps, revisions, and execution time.
5. DATA != AUTHORITY, REASONING != AUTHORITY, THINKING != AUTHORITY.
"""

from dataclasses import dataclass, field, asdict
import time
from typing import Dict, List, Optional, Any, Tuple
import uuid

from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.tokenizer import BPETokenizer
from chakrview.state.manager import CognitiveStateManager
from chakrview.reasoning.engine import GovernedReasoningEngine
from chakrview.capability.gate import CapabilityGate
from chakrview.intelligence.contracts import NeuralInferenceRequest
from chakrview.intelligence.context import IntelligenceContextBuilder, ContextBudget
from chakrview.intelligence.inference import NeuralInferenceEngine

from chakrview.thinking.thought import ThoughtStep, ThoughtPurpose
from chakrview.thinking.policy import ThinkingPolicy, get_standard_policy
from chakrview.thinking.workspace import ThinkingWorkspace, WorkspaceBudgetExceededError
from chakrview.thinking.attention import ThinkingAttention, FocusType
from chakrview.thinking.critique import CritiqueEngine, CritiqueVerdict
from chakrview.thinking.revision import RevisionEngine
from chakrview.thinking.stopping import ThinkingStoppingPolicy, StoppingCondition
from chakrview.thinking.trace import ThinkingTrace


@dataclass
class DeliberationOutcome:
    """Structured outcome package of a deliberation session."""
    response_text: str
    workspace: ThinkingWorkspace
    trace: ThinkingTrace
    stopping_condition: str
    success: bool
    weights_modified: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "response_text": self.response_text,
            "workspace": self.workspace.to_dict(),
            "trace": self.trace.to_dict(),
            "stopping_condition": self.stopping_condition,
            "success": self.success,
            "weights_modified": self.weights_modified,
        }


class DeliberationEngine:
    """
    Central orchestrator for bounded neuro-symbolic computational deliberation.
    """

    def __init__(
        self,
        model: ChakrMicro,
        tokenizer: BPETokenizer,
        inference_engine: Optional[NeuralInferenceEngine] = None,
        context_builder: Optional[IntelligenceContextBuilder] = None,
        reasoning_engine: Optional[GovernedReasoningEngine] = None,
        attention: Optional[ThinkingAttention] = None,
        critique_engine: Optional[CritiqueEngine] = None,
        revision_engine: Optional[RevisionEngine] = None,
        stopping_policy: Optional[ThinkingStoppingPolicy] = None,
        state_manager: Optional[CognitiveStateManager] = None,
        capability_gate: Optional[CapabilityGate] = None,
        capability_registry: Optional[Any] = None,
        personal_memory: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.inference_engine = inference_engine or NeuralInferenceEngine(model, tokenizer)
        self.context_builder = context_builder or IntelligenceContextBuilder()
        self.reasoning_engine = reasoning_engine or GovernedReasoningEngine()
        self.attention = attention or ThinkingAttention()
        self.critique_engine = critique_engine or CritiqueEngine()
        self.revision_engine = revision_engine or RevisionEngine()
        self.stopping_policy = stopping_policy or ThinkingStoppingPolicy()

        self.state_manager = state_manager
        self.capability_gate = capability_gate
        self.capability_registry = capability_registry
        self.personal_memory = personal_memory

    def _safe_add_thought(self, workspace: ThinkingWorkspace, **kwargs) -> Optional[ThoughtStep]:
        """Safely append a thought step without crashing if workspace budget is exceeded."""
        if not workspace.can_add_thought():
            return None
        try:
            return workspace.add_thought_step(**kwargs)
        except WorkspaceBudgetExceededError:
            return None

    def deliberate(
        self,
        objective: str,
        owner_id: str = "default_user",
        session_id: str = "default_session",
        policy: Optional[ThinkingPolicy] = None,
        max_new_tokens: int = 32,
        temperature: float = 0.7,
    ) -> DeliberationOutcome:
        """
        Execute an iterative, bounded deliberation cycle over an objective.
        """
        t0 = time.perf_counter()
        active_policy = policy or get_standard_policy()
        clean_objective = objective.strip()

        # 1. Initialize Workspace
        workspace = ThinkingWorkspace(
            task_id=f"delib_{uuid.uuid4().hex[:8]}",
            owner_id=owner_id,
            session_id=session_id,
            objective=clean_objective,
            policy=active_policy,
        )

        trace = ThinkingTrace(
            deliberation_id=workspace.workspace_id,
            task_id=workspace.task_id,
            owner_id=owner_id,
            session_id=session_id,
        )

        # 2. Ingest state assertions and personal memory into workspace
        if self.state_manager is not None:
            self.state_manager.owner_id = owner_id
            k_state = getattr(self.state_manager, "knowledge", None)
            if k_state is not None:
                for a in getattr(k_state, "assertions", {}).values():
                    workspace.add_evidence(
                        content=f"{a.subject} {a.predicate}: {a.value}",
                        source="cognitive_state",
                        evidence_id=a.assertion_id,
                        confidence=a.confidence,
                    )
            # Ingest uncertainties
            u_state = getattr(self.state_manager, "uncertainties", None)
            if u_state is not None:
                for u in getattr(u_state, "uncertainties", {}).values():
                    workspace.uncertainty_state[u.key] = u.to_dict()

        if self.personal_memory is not None and hasattr(self.personal_memory, "search"):
            try:
                mem_items = self.personal_memory.search(clean_objective, top_k=2)
                for m in mem_items:
                    workspace.add_evidence(
                        content=getattr(m, "content", str(m)),
                        source="personal_memory",
                    )
            except Exception:
                pass

        # 3. Initial Orientation Thought
        workspace.add_thought_step(
            purpose=ThoughtPurpose.OBSERVE,
            content=f"Initiating deliberation for objective: {clean_objective}",
            verification_state="INITIALIZED",
        )

        # 4. Iterative Deliberation Loop
        final_condition = StoppingCondition.MAX_REASONING_LIMIT
        final_reason = "Deliberation step limit reached."
        success = False

        while not workspace.is_concluded:
            can_run, limit_reason = workspace.can_deliberate()
            if not can_run:
                final_condition = StoppingCondition.MAX_REASONING_LIMIT
                final_reason = limit_reason or "Deliberation budget reached."
                workspace.mark_stopped(final_condition.value, final_reason)
                break

            # 4a. Attention Focus
            if not workspace.can_add_thought():
                final_condition = StoppingCondition.MAX_REASONING_LIMIT
                final_reason = f"Maximum allowed thought steps ({active_policy.max_thought_steps}) reached."
                workspace.mark_stopped(final_condition.value, final_reason)
                break

            focus = self.attention.select_focus(workspace)
            self._safe_add_thought(
                workspace,
                purpose=ThoughtPurpose.QUESTION,
                content=f"Attention directed to [{focus.focus_type.value}]: {focus.rationale}",
                result=focus.item_reference,
            )

            # 4b. Context Construction for Neural Core
            ev_summaries = [e["content"] for e in workspace.evidence[:4]]
            rev_directives = [r["direction"] for r in workspace.revisions[-1:]]

            assembled_ctx = self.context_builder.build_context(
                task_objective=clean_objective,
                system_identity="ChakrMicro Sovereign Deliberation Substrate",
                system_constraints=[
                    "Deliberate carefully; verify claims against established evidence.",
                    "DATA != AUTHORITY.",
                ],
                verified_knowledge=ev_summaries if ev_summaries else None,
                reasoning_summaries=rev_directives if rev_directives else None,
                tokenizer=self.tokenizer,
                budget=ContextBudget(max_context=512, generation_budget=128),
            )

            # 4c. Neural Candidate Proposal (ChakrMicro, zero weight updates)
            req = NeuralInferenceRequest(
                prompt_text=assembled_ctx.full_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                owner_id=owner_id,
                session_id=session_id,
            )
            neural_res = self.inference_engine.infer(req)

            # 4d. Record Proposed Thought
            if not workspace.can_add_thought():
                final_condition = StoppingCondition.MAX_REASONING_LIMIT
                final_reason = f"Maximum allowed thought steps ({active_policy.max_thought_steps}) reached."
                workspace.mark_stopped(final_condition.value, final_reason)
                break

            self._safe_add_thought(
                workspace,
                purpose=ThoughtPurpose.HYPOTHESIZE,
                content=f"Generated candidate proposal: {neural_res.text.strip()[:100]}",
                result=neural_res.text.strip(),
            )

            # 4e. Structured Reasoning Pass (Step 19)
            r_answer = ""
            r_trace = None
            verification_passed = False

            try:
                r_answer, r_trace = self.reasoning_engine.reason(
                    task_or_objective=clean_objective,
                    state_manager=self.state_manager,
                    capability_gate=self.capability_gate,
                    capability_registry=self.capability_registry,
                    personal_memory=self.personal_memory,
                    owner_id=owner_id,
                    session_id=session_id,
                )
                if r_trace and r_trace.success:
                    verification_passed = True
                    for ev in r_trace.evidence_items:
                        workspace.add_evidence(
                            content=str(ev.get("content", "")),
                            source=f"reasoning_{ev.get('source_type', 'capability')}",
                            evidence_id=ev.get("evidence_id"),
                            confidence=ev.get("confidence", 0.9),
                        )
            except Exception as r_err:
                r_answer = f"Reasoning exception: {str(r_err)}"
                verification_passed = False

            # Extract direct calculation / capability results if present in reasoning trace
            calc_val = None
            if r_trace:
                for obs in r_trace.observations:
                    act = obs.get("actual")
                    if isinstance(act, dict) and "result" in act:
                        calc_val = act["result"]
                        break
                    elif isinstance(act, (int, float)):
                        calc_val = act
                        break

            # Update working conclusion and active hypothesis
            if calc_val is not None:
                cand_conclusion = f"Final Answer: {calc_val}"
            elif r_answer and "Final Answer:" in r_answer:
                cand_conclusion = r_answer
            else:
                cand_conclusion = neural_res.text.strip() or r_answer

            workspace.set_conclusion(cand_conclusion)
            workspace.record_hypothesis(
                statement=f"Candidate solution: {cand_conclusion[:100]}",
                confidence=0.9 if verification_passed else 0.5,
            )

            # 4f. Critique Candidate
            critique = self.critique_engine.critique(cand_conclusion, workspace)
            if workspace.can_add_thought():
                self._safe_add_thought(
                    workspace,
                    purpose=ThoughtPurpose.CRITIQUE,
                    content=f"Critique verdict [{critique.verdict.value}] (score={critique.score:.2f}): {critique.recommended_action}",
                    result=critique.to_dict(),
                    verification_state="PASS" if critique.is_passed(active_policy.min_critique_score) else "FAIL",
                )

            # 4g. Evaluate Stopping Conditions
            should_stop, stop_cond, stop_reason = self.stopping_policy.evaluate_stopping(
                workspace=workspace,
                latest_critique=critique,
                verification_passed=verification_passed,
            )

            if should_stop:
                final_condition = stop_cond
                final_reason = stop_reason
                success = (stop_cond in (StoppingCondition.SOLVED, StoppingCondition.SOLVED_WITH_UNCERTAINTY))
                workspace.mark_stopped(stop_cond.value, stop_reason)
                break

            # 4h. Corrective Revision Loop
            if active_policy.allow_revisions and workspace.can_revise():
                rev_plan = self.revision_engine.plan_revision(workspace, critique)
                # Next deliberation cycle incorporates revision directive
            else:
                final_condition = (
                    StoppingCondition.MAX_REVISIONS_REACHED
                    if workspace.revision_count >= active_policy.max_revision_cycles
                    else StoppingCondition.MAX_REASONING_LIMIT
                )
                final_reason = (
                    f"Deliberation concluded: revision limit reached "
                    f"({workspace.revision_count}/{active_policy.max_revision_cycles})."
                )
                workspace.mark_stopped(final_condition.value, final_reason)
                break

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Final response synthesis
        if final_condition in (StoppingCondition.SOLVED, StoppingCondition.SOLVED_WITH_UNCERTAINTY):
            response_text = workspace.current_working_conclusion or "Deliberation concluded successfully."
        elif final_condition == StoppingCondition.INSUFFICIENT_INFORMATION:
            response_text = (
                f"Insufficient information to reach verified conclusion for '{clean_objective}'. "
                f"Reason: {final_reason}"
            )
        else:
            response_text = (
                f"Deliberation terminated under condition [{final_condition.value}]: {final_reason}\n\n"
                f"Partial conclusion: {workspace.current_working_conclusion or 'None'}"
            )

        # Finalize trace telemetry
        trace.number_of_steps = workspace.step_count
        trace.number_of_revisions = workspace.revision_count
        trace.hypotheses_considered = list(workspace.hypotheses.keys())
        trace.evidence_used = [e["evidence_id"] for e in workspace.evidence]
        trace.critiques = workspace.critiques
        trace.revisions = workspace.revisions
        trace.finalize(
            status="COMPLETED" if success else "INCOMPLETE",
            condition=final_condition.value,
            reason=final_reason,
            elapsed_ms=elapsed_ms,
        )

        assert self.inference_engine.verify_weights_unmodified(), "Model weights altered during deliberation!"

        return DeliberationOutcome(
            response_text=response_text,
            workspace=workspace,
            trace=trace,
            stopping_condition=final_condition.value,
            success=success,
            weights_modified=False,
        )
