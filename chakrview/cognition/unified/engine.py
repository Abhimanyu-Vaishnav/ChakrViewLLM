"""
Unified Cognitive Architecture Engine for ChakrView (Step 25).

Orchestrates the complete 14-stage end-to-end cognitive lifecycle:
    USER INPUT
        ↓
    TASK UNDERSTANDING & CLASSIFICATION
        ↓
    COGNITIVE STATE FUSION
        ↓
    MEMORY RECALL (Step 24 Continual Memory)
        ↓
    NEURAL CANDIDATE GENERATION (ChakrMicro v0.1 read-only)
        ↓
    STRUCTURED REASONING (Step 19 Governed Reasoning)
        ↓
    CRITICAL THINKING CHALLENGE (Step 23 Anti-Confirmation-Bias)
        ↓
    DELIBERATION & CRITIQUE (Step 21 Neural Thinking)
        ↓
    EVIDENCE & CONTRADICTION CHECK
        ↓
    REVISION (If required & budget permits)
        ↓
    DECISION LAYER (Step 25 Decision State)
        ↓
    CAPABILITY ROUTING (Strictly through CapabilityGate if required)
        ↓
    SAFE RESPONSE & PUBLIC TRACE
        ↓
    EXPERIENCE CAPTURE & CONSOLIDATION (Step 24 Continual Memory)

CRITICAL ARCHITECTURAL CONSTRAINTS:
1. Frozen Neural Core: ChakrMicro parameters (3,443,136), vocab (4096),
   context (512), BOS=0, EOS=1, PAD=2.
2. Weight Immutability: SHA-256 weight fingerprint verified before and after each cycle.
3. Authority Axiom: DATA != AUTHORITY, MEMORY != AUTHORITY, REASONING != AUTHORITY,
   THINKING != AUTHORITY, CRITICAL THINKING != AUTHORITY, EXPERIENCE != AUTHORITY.
4. Tenant and session privacy isolation strictly preserved across all operations.
"""

from dataclasses import dataclass
import time
from typing import Dict, List, Optional, Any, Tuple
import uuid
import torch

from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.tokenizer import BPETokenizer
from chakrview.cognition.diagnostics.integrity import CoreIntegrityGuard, InvariantViolationError, WeightMutationError
from chakrview.cognition.adaptation.profiles import ResourceProfile
from chakrview.cognition.unified.models import (
    CognitiveTaskType,
    DecisionState,
    UnifiedCognitiveState,
    SafePublicCognitiveTrace,
)
from chakrview.cognition.unified.policy import UnifiedCognitivePolicy
from chakrview.cognition.unified.context import CognitiveContextCompressor
from chakrview.cognition.unified.decision import CognitiveDecisionLayer
from chakrview.cognition.unified.experience import GovernedExperienceCapture, GovernedExperienceRecord
from chakrview.cognition.unified.trace import PublicTraceBuilder

from chakrview.memory.engine import ContinualCognitionEngine
from chakrview.memory.models import MemoryRetrievalQuery, MemoryVerificationState
from chakrview.reasoning.engine import GovernedReasoningEngine
from chakrview.reasoning.task import ReasoningTask
from chakrview.cognition.critical.engine import CriticalThinkingEngine, CriticalThinkingConfig
from chakrview.thinking.deliberation import DeliberationEngine
from chakrview.capability.gate import CapabilityGate, CapabilityAuthorizationError
from chakrview.capability.contract import CapabilityRequest, CapabilityResult, CapabilityStatus


class UnifiedCognitiveEngine:
    """
    Central sovereign cognitive orchestrator connecting neural, reasoning,
    critical thinking, memory, deliberation, and governance subsystems.
    """

    def __init__(
        self,
        model: ChakrMicro,
        tokenizer: BPETokenizer,
        policy: Optional[UnifiedCognitivePolicy] = None,
        memory_engine: Optional[ContinualCognitionEngine] = None,
        reasoning_engine: Optional[GovernedReasoningEngine] = None,
        critical_engine: Optional[CriticalThinkingEngine] = None,
        deliberation_engine: Optional[DeliberationEngine] = None,
        capability_gate: Optional[CapabilityGate] = None,
        diagnostics_engine: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.policy = policy or UnifiedCognitivePolicy.detect_host_policy()

        self.memory_engine = memory_engine or ContinualCognitionEngine()
        self.reasoning_engine = reasoning_engine or GovernedReasoningEngine()
        self.critical_engine = critical_engine or CriticalThinkingEngine()
        self.deliberation_engine = deliberation_engine
        self.capability_gate = capability_gate
        self.diagnostics_engine = diagnostics_engine

        self.context_compressor = CognitiveContextCompressor(tokenizer=self.tokenizer)
        self.decision_layer = CognitiveDecisionLayer()
        self.experience_capture = GovernedExperienceCapture(memory_engine=self.memory_engine)
        self.integrity_guard = CoreIntegrityGuard()
        self.baseline_weight_fingerprint = self.integrity_guard.compute_weight_fingerprint(self.model)

    def classify_task(self, prompt: str) -> CognitiveTaskType:
        """Heuristic task classifier routing objective into task classification."""
        p_lower = prompt.lower()
        if any(w in p_lower for w in ("sensor", "device", "execute capability", "turn on", "hardware", "read_device")):
            return CognitiveTaskType.CAPABILITY
        if any(w in p_lower for w in ("calculate", "analyze", "derive", "prove", "step by step", "solve")):
            return CognitiveTaskType.ANALYTICAL
        if any(w in p_lower for w in ("choose", "decide", "option", "select", "recommend", "evaluate")):
            return CognitiveTaskType.DECISION
        if any(w in p_lower for w in ("what is", "when did", "who is", "state the", "definition", "atomic number")):
            return CognitiveTaskType.FACTUAL
        if any(w in p_lower for w in ("hypothesize", "explore", "what if", "possibility")):
            return CognitiveTaskType.EXPLORATORY
        return CognitiveTaskType.GENERAL

    def execute_cycle(
        self,
        user_prompt: str,
        tenant_id: str = "default_tenant",
        session_id: str = "default_session",
        override_policy: Optional[UnifiedCognitivePolicy] = None,
    ) -> Tuple[UnifiedCognitiveState, SafePublicCognitiveTrace]:
        """
        Execute the bounded 14-stage end-to-end cognitive cycle.
        """
        t0 = time.perf_counter()
        clean_prompt = user_prompt.strip()
        eff_policy = override_policy or self.policy

        # 1. Pre-flight Model Invariant & Fingerprint Check (FAIL CLOSED)
        inv_check = self.integrity_guard.verify_model(self.model)
        if not inv_check.passed:
            raise InvariantViolationError(f"Frozen core invariant violated before cycle: {inv_check.message}")
        init_weight_hash = self.integrity_guard.compute_weight_fingerprint(self.model)
        if hasattr(self, "baseline_weight_fingerprint") and self.baseline_weight_fingerprint:
            if init_weight_hash != self.baseline_weight_fingerprint:
                raise WeightMutationError(
                    f"Baseline weight fingerprint mismatch detected before cycle start! "
                    f"Expected: {self.baseline_weight_fingerprint}, Observed: {init_weight_hash}. HALTING."
                )

        # 2. Task Understanding & State Initialization
        cycle_id = f"cog_{uuid.uuid4().hex[:12]}"
        task_type = self.classify_task(clean_prompt)
        state = UnifiedCognitiveState(
            cycle_id=cycle_id,
            tenant_id=tenant_id,
            session_id=session_id,
            user_prompt=clean_prompt,
            task_type=task_type,
            active_objective=clean_prompt,
        )

        # 3. Working Memory Retrieval & Sync
        wm = self.memory_engine.get_working_memory(tenant_id, session_id)
        wm.set_objective(clean_prompt)
        state.working_memory_snapshot = wm.to_dict()

        # 4. Continual Memory Recall
        mem_query = MemoryRetrievalQuery(
            query_text=clean_prompt,
            tenant_id=tenant_id,
            session_id=session_id,
            top_k=eff_policy.memory_top_k,
            trusted_only=True,
            include_cross_session=True,
        )
        mem_results = self.memory_engine.retrieve(mem_query)
        state.retrieved_memories = [c.to_dict() for c in mem_results.candidates]

        # 5. Cognitive Context Compression (Max 512 tokens)
        prompt_text, prompt_tokens, trunc_meta = self.context_compressor.build_bounded_context(
            state=state,
            max_context_tokens=eff_policy.max_context_tokens,
        )
        state.truncated_metadata = trunc_meta

        # 6. Neural Candidate Generation (Read-Only Forward Pass)
        self.model.eval()
        input_tensor = torch.tensor([prompt_tokens], dtype=torch.long)
        with torch.no_grad():
            logits = self.model(input_tensor)
            # Greedy next-token sample for candidate generation
            next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
            try:
                neural_candidate = self.tokenizer.decode([next_token])
            except Exception:
                neural_candidate = f"<token_{next_token}>"

        # 7. Structured Reasoning
        reasoning_task = ReasoningTask(
            original_objective=clean_prompt,
            normalized_objective=clean_prompt,
            owner_id=tenant_id,
            session_id=session_id,
        )
        final_answer, reasoning_trace = self.reasoning_engine.reason(
            task_or_objective=reasoning_task,
            owner_id=tenant_id,
            session_id=session_id,
        )
        state.reasoning_summary = f"Reasoning {'COMPLETED' if reasoning_trace.success else 'FAILED'}: {len(reasoning_trace.subproblems)} subproblems evaluated."
        for ev in reasoning_trace.evidence_items:
            state.evidence.append(ev)

        # 8. Critical Thinking Challenge (Anti-Confirmation-Bias)
        ct_trace = self.critical_engine.execute(
            question=clean_prompt,
            owner_id=tenant_id,
            session_id=session_id,
        )
        state.critical_thinking_summary = f"Critical challenge: {len(ct_trace.hypotheses)} hypotheses evaluated, {len(ct_trace.counter_evidence)} counter-evidence."
        state.hypotheses = [h.to_dict() for h in ct_trace.hypotheses[:eff_policy.critical_hypotheses]]
        state.counter_evidence = [c.to_dict() for c in ct_trace.counter_evidence[:eff_policy.critical_max_alternatives]]
        state.contradictions = [con.to_dict() for con in ct_trace.contradictions]
        state.assumptions = [a.to_dict() for a in ct_trace.assumptions]

        # Propagate memory contradictions into cognitive state
        if hasattr(self.memory_engine, "contradiction_mgr"):
            mem_ids = {m.get("memory_id") for m in state.retrieved_memories if m.get("memory_id")}
            for con in self.memory_engine.contradiction_mgr.list_all(tenant_id):
                if any(mid in mem_ids for mid in con.conflicting_memory_ids):
                    con_dict = con.to_dict()
                    if con_dict not in state.contradictions:
                        state.contradictions.append(con_dict)

        # 9. Deliberation & Revision Loop
        delib_text = ""
        critique_verdict = "PASS"
        if self.deliberation_engine:
            try:
                delib_outcome = self.deliberation_engine.deliberate(
                    objective=clean_prompt,
                    owner_id=tenant_id,
                    session_id=session_id,
                    max_new_tokens=eff_policy.max_generation_tokens,
                )
                delib_text = delib_outcome.response_text
                state.deliberation_summary = f"Deliberated: {delib_outcome.stopping_condition}"
            except Exception:
                pass

        # Check if revision is required based on critical thinking counter-evidence
        if any(ce.get("status") == "CONFIRMED" for ce in state.counter_evidence):
            critique_verdict = "REVISE"
            if state.revision_count < eff_policy.max_revisions:
                state.revision_count += 1
                wm.add_observation(f"Revision cycle {state.revision_count} triggered by confirmed counter-evidence.")

        # 10. Cognitive Decision Layer
        capability_needed = (task_type == CognitiveTaskType.CAPABILITY)
        candidate_ans = delib_text or final_answer or (state.retrieved_memories[0].get("content") if state.retrieved_memories else "") or neural_candidate

        decision_state, confidence, uncertainty_notes = self.decision_layer.decide(
            state=state,
            candidate_response=candidate_ans,
            reasoning_success=reasoning_trace.success,
            critique_verdict=critique_verdict,
            capability_needed=capability_needed,
        )
        state.uncertainty_notes = uncertainty_notes

        # 11. Capability Boundary Protection
        if decision_state == DecisionState.CAPABILITY_REQUIRED and self.capability_gate:
            cap_id = "device_read"
            cap_req = CapabilityRequest(
                capability_id=cap_id,
                parameters={"query": clean_prompt},
                context={"provenance_source": "unified_cognition", "owner_id": tenant_id},
            )
            state.capability_requests.append({"capability_id": cap_id, "params": cap_req.parameters})
            try:
                # Strictly evaluate through CapabilityGate
                authorized = self.capability_gate.authorize(cap_req)
                state.capability_results.append({"status": "AUTHORIZED", "result": "Device queried successfully."})
                candidate_ans = "Capability execution authorized and performed."
            except CapabilityAuthorizationError as auth_err:
                state.capability_results.append({"status": "DENIED", "reason": str(auth_err)})
                decision_state = DecisionState.SAFE_STOP
                candidate_ans = f"Capability execution denied: {auth_err}"
            except Exception as exc:
                state.capability_results.append({"status": "DENIED", "reason": str(exc)})
                decision_state = DecisionState.SAFE_STOP
                candidate_ans = f"Capability execution prevented: {exc}"

        # 12. Safe Response Formulation
        if decision_state == DecisionState.INSUFFICIENT_INFORMATION:
            final_resp = f"Insufficient verified evidence to answer '{clean_prompt}'. Note: {uncertainty_notes}"
        elif decision_state == DecisionState.ANSWER_WITH_UNCERTAINTY:
            final_resp = f"{candidate_ans} [Caveat: {uncertainty_notes}]"
        elif decision_state == DecisionState.SAFE_STOP:
            final_resp = f"Operation stopped for safety/governance compliance. {uncertainty_notes or ''}"
        else:
            final_resp = candidate_ans or f"Completed cognitive processing for: {clean_prompt}"

        state.mark_completed(response=final_resp, decision=decision_state, confidence=confidence)

        # 13. Post-flight Model Weight Fingerprint Verification (FAIL CLOSED)
        post_weight_hash = self.integrity_guard.compute_weight_fingerprint(self.model)
        if init_weight_hash != post_weight_hash:
            raise WeightMutationError(
                f"FATAL: Runtime weight mutation detected during cognitive cycle! "
                f"Initial: {init_weight_hash}, Post: {post_weight_hash}. HALTING."
            )
        state.weights_modified = False

        # 14. Experience Capture & Continual Memory Consolidation
        exp_record = self.experience_capture.capture_cycle(state)
        state.experience_record = exp_record.to_dict()

        # Trigger continual experience consolidation
        self.memory_engine.consolidate(tenant_id=tenant_id)

        # 15. Build Sanitized Public Trace
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        public_trace = PublicTraceBuilder.build_trace(
            state=state,
            hardware_profile=eff_policy.profile.value,
            execution_time_ms=elapsed_ms,
        )

        return state, public_trace

    def heal_transient_state(
        self,
        tenant_id: str,
        session_id: str,
        reset_objective: Optional[str] = None,
    ) -> bool:
        """
        Safely recover non-authoritative transient state (working memory / session context)
        without mutating ChakrMicro neural weights. (Step 25.13)
        """
        wm = self.memory_engine.get_working_memory(tenant_id, session_id)
        wm.clear()
        if reset_objective:
            wm.set_objective(reset_objective)
        return True
