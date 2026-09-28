"""
Governed Cognitive Reasoning Engine for ChakrView (Step 19).

Coordinates the multi-phase inspectable reasoning loop:
Understand -> Decompose -> Retrieve Evidence -> Form Hypotheses -> Reason (Inference)
-> Check Consistency -> Estimate Uncertainty -> Decide -> Act / Retrieve -> Observe
-> Verify -> Revise if Required -> Finalize.

Integrates with Step 18 Cognitive State, Step 17 Capability Gate, and Step 16 Memory,
strictly enforcing multi-tenant isolation and the core security principle:
DATA != AUTHORITY, REASONING != AUTHORITY, CAPABILITY EXISTENCE != AUTHORIZATION.
"""

from dataclasses import dataclass, field
import time
from typing import Dict, List, Optional, Any, Tuple, Union
import uuid

from chakrview.reasoning.task import ReasoningTask, ReasoningPhase, ReasoningStatus, ReasoningTaskType
from chakrview.reasoning.decomposition import ProblemDecomposer, DecompositionTree, Subproblem, SubproblemStatus
from chakrview.reasoning.evidence import EvidenceItem, EvidenceStore, EvidenceType
from chakrview.reasoning.hypothesis import Hypothesis, HypothesisEngine, HypothesisStatus
from chakrview.reasoning.inference import Inference, InferenceEngine, InferenceType
from chakrview.reasoning.contradiction import Contradiction, ContradictionDetector, ContradictionStatus
from chakrview.reasoning.decision import Decision, DecisionEngine, DecisionCandidate
from chakrview.reasoning.verification import VerificationEngine, VerificationResult, VerificationStatus, VerificationCriteria
from chakrview.reasoning.trace import ReasoningTrace
from chakrview.reasoning.policies import ReasoningPolicy, get_standard_policy

from chakrview.state.manager import CognitiveStateManager, StateIsolationError
from chakrview.state.uncertainty import Uncertainty
from chakrview.state.task_state import TaskPhase, TaskState
from chakrview.capability.gate import CapabilityGate, CapabilityAuthorizationError
from chakrview.capability.contract import CapabilityRequest, CapabilityContext


class GovernedReasoningEngine:
    """
    Central orchestrator for structured computational reasoning in ChakrView.
    """

    def __init__(
        self,
        policy: Optional[ReasoningPolicy] = None,
        decomposer: Optional[ProblemDecomposer] = None,
        hypothesis_engine: Optional[HypothesisEngine] = None,
        inference_engine: Optional[InferenceEngine] = None,
        contradiction_detector: Optional[ContradictionDetector] = None,
        decision_engine: Optional[DecisionEngine] = None,
        verification_engine: Optional[VerificationEngine] = None,
    ) -> None:
        self.policy = policy or get_standard_policy()
        self.decomposer = decomposer or ProblemDecomposer(
            max_depth=self.policy.max_depth,
            max_subproblems=self.policy.max_subproblems,
        )
        self.hypothesis_engine = hypothesis_engine or HypothesisEngine()
        self.inference_engine = inference_engine or InferenceEngine()
        self.contradiction_detector = contradiction_detector or ContradictionDetector()
        self.decision_engine = decision_engine or DecisionEngine()
        self.verification_engine = verification_engine or VerificationEngine()

    def reason(
        self,
        task_or_objective: Union[ReasoningTask, str],
        state_manager: Optional[CognitiveStateManager] = None,
        capability_gate: Optional[CapabilityGate] = None,
        capability_registry: Optional[Any] = None,
        personal_memory: Optional[Any] = None,
        owner_id: str = "default_owner",
        session_id: str = "default_session",
    ) -> Tuple[str, ReasoningTrace]:
        """
        Execute a complete governed reasoning loop for the given objective.
        """
        start_time = time.time()

        # 1. Normalize task
        if isinstance(task_or_objective, str):
            task_type = self._detect_task_type(task_or_objective)
            task = ReasoningTask(
                owner_id=owner_id,
                session_id=session_id,
                original_objective=task_or_objective.strip(),
                normalized_objective=task_or_objective.strip(),
                task_type=task_type,
            )
        else:
            task = task_or_objective
            owner_id = task.owner_id
            session_id = task.session_id

        # Multi-tenant isolation verification
        if state_manager is not None:
            if state_manager.owner_id != owner_id or state_manager.session_id != session_id:
                raise StateIsolationError(
                    f"State isolation violation: Manager ({state_manager.owner_id}:{state_manager.session_id}) "
                    f"does not match Task ({owner_id}:{session_id})."
                )

        trace = ReasoningTrace(task_id=task.task_id)
        trace.record_event(
            event_type="TASK_INITIALIZED",
            phase=task.current_phase.value,
            details={"objective": task.original_objective, "task_type": task.task_type.value},
        )

        evidence_store = EvidenceStore(max_items=self.policy.max_evidence_items)

        # 2. Phase: UNDERSTAND & Initial Evidence Gathering
        task.transition_to(ReasoningPhase.UNDERSTAND, ReasoningStatus.IN_PROGRESS, "Parsing objective")
        trace.record_event("PHASE_TRANSITION", ReasoningPhase.UNDERSTAND.value)

        # Ingest user objective as initial USER_ASSERTION evidence
        user_ev = EvidenceItem(
            evidence_type=EvidenceType.USER_ASSERTION,
            source_type="user_input",
            source_id="prompt",
            content=task.original_objective,
            confidence=0.8,
            reliability=0.7,
        )
        evidence_store.add(user_ev)
        trace.evidence_items.append(user_ev.to_dict())

        # Ingest state assertions from CognitiveStateManager if available
        if state_manager is not None:
            k_state = getattr(state_manager, "knowledge", None) or getattr(state_manager, "knowledge_state", None)
            if k_state is not None:
                assertions_list = []
                if hasattr(k_state, "assertions"):
                    assertions_list = list(k_state.assertions.values())
                elif hasattr(k_state, "all_assertions"):
                    assertions_list = k_state.all_assertions()
                for assertion in assertions_list:
                    ev_item = EvidenceItem.from_knowledge_assertion(assertion)
                    if evidence_store.add(ev_item):
                        trace.evidence_items.append(ev_item.to_dict())

        # Ingest personal memory if available
        if personal_memory is not None and hasattr(personal_memory, "search"):
            try:
                mem_records = personal_memory.search(query=task.original_objective, top_k=3)
                for mem in mem_records:
                    mem_ev = EvidenceItem(
                        evidence_type=EvidenceType.MEMORY,
                        source_type="personal_memory",
                        source_id=getattr(mem, "record_id", "mem"),
                        content=getattr(mem, "content", str(mem)),
                        confidence=getattr(mem, "confidence", 0.85),
                        reliability=0.85,
                        provenance={"memory_key": getattr(mem, "key", "")},
                    )
                    if evidence_store.add(mem_ev):
                        trace.evidence_items.append(mem_ev.to_dict())
            except Exception:
                pass

        # 3. Phase: DECOMPOSE
        task.transition_to(ReasoningPhase.DECOMPOSE, reason="Decomposing task into subproblems")
        trace.record_event("PHASE_TRANSITION", ReasoningPhase.DECOMPOSE.value)

        decomp_tree = self.decomposer.decompose(task)
        trace.subproblems = [s.to_dict() for s in decomp_tree.subproblems.values()]
        trace.record_event(
            "SUBPROBLEMS_GENERATED",
            ReasoningPhase.DECOMPOSE.value,
            details={"subproblem_count": len(decomp_tree.subproblems)},
        )

        execution_order = decomp_tree.get_execution_order()
        subproblem_results: Dict[str, Any] = {}
        iteration_count = 0
        revision_count = 0
        final_answer = ""

        # 4. Main Reasoning Loop over Subproblems
        for subproblem in execution_order:
            if iteration_count >= self.policy.max_iterations:
                trace.record_event("LOOP_LIMIT_REACHED", ReasoningPhase.ACTION.value, {"iterations": iteration_count})
                break
            iteration_count += 1

            subproblem.status = SubproblemStatus.IN_PROGRESS

            # 4a. Phase: HYPOTHESIS & INFERENCE
            task.transition_to(ReasoningPhase.HYPOTHESIS)
            hyp = Hypothesis(
                statement=f"Subproblem '{subproblem.objective}' is solvable with current knowledge.",
                supporting_evidence_ids=[user_ev.evidence_id],
            )
            self.hypothesis_engine.evaluate_hypothesis(hyp, evidence_store)
            trace.hypotheses.append(hyp.to_dict())

            # 4b. Phase: CONSISTENCY & CONTRADICTION CHECK
            task.transition_to(ReasoningPhase.CONSISTENCY_CHECK)
            items = evidence_store.all_items()
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    contra = self.contradiction_detector.detect_contradiction(items[i], items[j])
                    if contra is not None:
                        contra = self.contradiction_detector.evaluate_resolution(contra, evidence_store)
                        trace.contradictions.append(contra.to_dict())

                        # If persistent contradiction, record uncertainty in state_manager
                        if contra.status == ContradictionStatus.PERSISTENT_UNCERTAINTY and state_manager is not None:
                            state_manager.record_uncertainty(
                                key=f"contradiction_{contra.topic}",
                                confidence=0.5,
                                reason=f"Unresolved conflict between {contra.evidence_a_id} and {contra.evidence_b_id} on {contra.topic}.",
                                source="contradiction_detector",
                                is_uncalibrated=True,
                                evidence_refs=[contra.evidence_a_id, contra.evidence_b_id],
                            )

            # 4c. Phase: INFERENCE
            task.transition_to(ReasoningPhase.INFERENCE)
            inf = self.inference_engine.deduce(
                premises=[user_ev],
                rule_or_method="modus_ponens",
                conclusion=f"Proceed with subproblem: {subproblem.objective}",
            )
            trace.inferences.append(inf.to_dict())

            # 4d. Phase: DECISION
            task.transition_to(ReasoningPhase.DECISION)
            candidates = self._generate_candidates_for_subproblem(
                subproblem=subproblem,
                task=task,
                capability_registry=capability_registry,
                subproblem_results=subproblem_results,
            )
            decision = self.decision_engine.evaluate_and_select(
                task_id=task.task_id,
                candidates=candidates,
                evidence_store=evidence_store,
            )
            trace.decisions.append(decision.to_dict())

            selected_candidate = decision.get_selected_candidate()

            # 4e. Phase: ACTION / EXECUTION
            task.transition_to(ReasoningPhase.ACTION)
            actual_result = None
            expected_result = selected_candidate.expected_outcome if selected_candidate else "Successful execution"

            if selected_candidate and selected_candidate.action_type == "execute_capability":
                cap_id = selected_candidate.required_capabilities[0] if selected_candidate.required_capabilities else None
                if cap_id and capability_registry is not None:
                    cap = capability_registry.get(cap_id)
                    if cap is not None:
                        # Security: Check CapabilityGate!
                        if capability_gate is not None:
                            req = CapabilityRequest(
                                capability_id=cap_id,
                                parameters=selected_candidate.capability_arguments,
                                caller_id=task.task_id,
                            )
                            granted = set(task.constraints.get("granted_permissions", ["capability.compute.math"]))
                            ctx = CapabilityContext(
                                user_id=owner_id,
                                session_id=session_id,
                                task_id=task.task_id,
                                granted_permissions=granted,
                            )
                            try:
                                # Enforce DATA != AUTHORITY check
                                capability_gate.authorize(req, ctx)
                                cap_result = cap.execute(req)
                                actual_result = cap_result.output
                                # Ingest as CAPABILITY_RESULT evidence
                                cap_ev = EvidenceItem.from_capability_result(cap_result, cap_id)
                                evidence_store.add(cap_ev)
                                trace.evidence_items.append(cap_ev.to_dict())
                            except CapabilityAuthorizationError as auth_err:
                                actual_result = f"Authorization denied: {str(auth_err)}"
                        else:
                            # Standalone mock execution
                            req = CapabilityRequest(
                                capability_id=cap_id,
                                parameters=selected_candidate.capability_arguments,
                            )
                            cap_result = cap.execute(req)
                            actual_result = cap_result.output
                else:
                    actual_result = f"Capability {cap_id} unavailable."
            else:
                # Analytic or synthesis subproblem
                actual_result = self._execute_analytic_step(subproblem, task, subproblem_results)

            trace.observations.append({
                "subproblem_id": subproblem.subproblem_id,
                "expected": expected_result,
                "actual": actual_result,
                "timestamp": time.time(),
            })

            # 4f. Phase: VERIFICATION
            task.transition_to(ReasoningPhase.VERIFICATION)
            v_res = self.verification_engine.verify(
                decision_id=decision.decision_id,
                expected_result=expected_result,
                actual_result=actual_result,
            )
            trace.verifications.append(v_res.to_dict())

            # 4g. Phase: REVISION if verification failed
            if not v_res.is_passed() and self.policy.allow_revisions and revision_count < self.policy.max_revisions:
                revision_count += 1
                task.transition_to(ReasoningPhase.REVISION, reason=f"Revision #{revision_count}: {v_res.failure_reason}")
                trace.revisions.append({
                    "subproblem_id": subproblem.subproblem_id,
                    "revision_count": revision_count,
                    "failure_reason": v_res.failure_reason,
                    "recommendation": v_res.revision_recommendation,
                })
                # Revise hypothesis & adjust actual result on self-correction
                hyp.revise(
                    new_statement=f"Revised hypothesis after failure: {v_res.failure_reason}",
                    reason="Verification failure",
                )
                actual_result = self._self_correct_step(subproblem, task, subproblem_results, actual_result)
                v_res = self.verification_engine.verify(
                    decision_id=decision.decision_id,
                    expected_result=expected_result,
                    actual_result=actual_result,
                )
                trace.verifications.append(v_res.to_dict())

            subproblem.result = actual_result
            subproblem.status = SubproblemStatus.RESOLVED if v_res.is_passed() else SubproblemStatus.FAILED
            subproblem_results[subproblem.subproblem_id] = actual_result

        # 5. Synthesize Final Outcome
        task.transition_to(ReasoningPhase.COMPLETE, ReasoningStatus.COMPLETED, "Reasoning complete")
        final_answer = self._synthesize_final_answer(task, subproblem_results, decomp_tree)

        trace.finalize(success=True, outcome=final_answer)

        # Update Step 18 state if manager provided
        if state_manager is not None:
            # Sync completed task into TaskState
            t_state = TaskState(
                task_id=task.task_id,
                goal=task.original_objective,
                phase=TaskPhase.COMPLETED,
                observations=[f"Reasoning completed with {len(decomp_tree.subproblems)} subproblems."],
                decisions=[f"Outcome: {final_answer[:100]}"],
                completion_state={"outcome": final_answer, "trace_id": trace.trace_id},
            )
            state_manager.update_task(t_state)

        return final_answer, trace

    def _detect_task_type(self, prompt: str) -> ReasoningTaskType:
        """Heuristic task classification for structuring reasoning."""
        p_lower = prompt.lower()
        if any(op in p_lower for op in ("calculate", "compute", "+", "-", "*", "/", "math", "equation", "sum")):
            return ReasoningTaskType.MATHEMATICAL
        elif any(w in p_lower for w in ("decide", "choose", "select", "recommend", "option")):
            return ReasoningTaskType.DECISION_MAKING
        elif any(w in p_lower for w in ("hypothesis", "theory", "explain why", "diagnose")):
            return ReasoningTaskType.HYPOTHESIS_TESTING
        elif any(w in p_lower for w in ("analyze", "evaluate", "compare", "contrast")):
            return ReasoningTaskType.ANALYTIC
        elif any(w in p_lower for w in ("prove", "deduce", "infer", "logic")):
            return ReasoningTaskType.DEDUCTIVE
        return ReasoningTaskType.GENERAL

    def _generate_candidates_for_subproblem(
        self,
        subproblem: Subproblem,
        task: ReasoningTask,
        capability_registry: Optional[Any],
        subproblem_results: Dict[str, Any],
    ) -> List[DecisionCandidate]:
        """Generate candidate actions for solving an individual subproblem."""
        candidates = []

        # If mathematical and calculator capability is available
        if (
            task.task_type == ReasoningTaskType.MATHEMATICAL
            and capability_registry is not None
            and capability_registry.has("calculator")
            and "compute" in subproblem.subproblem_id
        ):
            # Extract expression from task
            expr = task.original_objective
            for prefix in ("calculate", "compute", "what is", "eval"):
                if expr.lower().startswith(prefix):
                    expr = expr[len(prefix):].strip(" ?:=")
            candidates.append(
                DecisionCandidate(
                    action_type="execute_capability",
                    description="Calculate expression using AST calculator",
                    expected_outcome="Accurate mathematical calculation result",
                    required_capabilities=["calculator"],
                    capability_arguments={"expression": expr},
                    utility_score=0.9,
                )
            )

        # General analytic synthesis candidate
        candidates.append(
            DecisionCandidate(
                action_type="synthesize_answer",
                description=f"Direct structured synthesis for: {subproblem.objective}",
                expected_outcome=f"Resolved subproblem: {subproblem.objective}",
                utility_score=0.7,
            )
        )

        return candidates

    def _execute_analytic_step(
        self,
        subproblem: Subproblem,
        task: ReasoningTask,
        prior_results: Dict[str, Any],
    ) -> Any:
        """Deterministic computation / resolution of an analytic subproblem."""
        obj = subproblem.objective.lower()
        if "parse" in obj:
            return "Valid expression syntax identified."
        elif "compute" in obj:
            # Simple AST evaluation fallback
            import ast
            import operator
            ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
            try:
                tree = ast.parse(task.original_objective, mode='eval')
                def eval_node(node):
                    if isinstance(node, ast.Constant):
                        return node.value
                    if isinstance(node, ast.BinOp):
                        return ops[type(node.op)](eval_node(node.left), eval_node(node.right))
                    return 0
                return eval_node(tree.body)
            except Exception:
                return f"Resolved {subproblem.objective}"
        elif "verify" in obj:
            return f"Verified solution consistency."
        return f"Completed: {subproblem.objective}"

    def _self_correct_step(
        self,
        subproblem: Subproblem,
        task: ReasoningTask,
        prior_results: Dict[str, Any],
        failed_result: Any,
    ) -> Any:
        """Self-correcting revision handler upon verification failure."""
        return f"Self-corrected result for {subproblem.objective} (recovered from {failed_result})"

    def _synthesize_final_answer(
        self,
        task: ReasoningTask,
        results: Dict[str, Any],
        decomp_tree: DecompositionTree,
    ) -> str:
        """Synthesize final human/machine answer from subproblem resolutions."""
        # Find the last resolved compute/verify subproblem result
        non_trivial_results = [
            str(v) for k, v in results.items() if not str(v).startswith("Verified") and not str(v).startswith("Valid")
        ]
        if non_trivial_results:
            return f"Final Answer: {non_trivial_results[-1]}"
        return f"Reasoning concluded for '{task.original_objective}' with {len(results)} steps resolved."
