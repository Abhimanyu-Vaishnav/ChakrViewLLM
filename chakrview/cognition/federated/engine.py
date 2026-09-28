"""
Federated Cognition Engine (Step 26).

Central orchestrator for bounded, cooperative multi-agent intelligence operating
around the SINGLE sovereign frozen ChakrMicro v0.1 neural core.

COGNITIVE LIFECYCLE:
1.  Pre-flight Invariant & SHA-256 Fingerprint Audit (FAIL CLOSED)
2.  Task Decomposition into Bounded Subtasks
3.  Agent Provisioning & Contract Verification (AgentRegistry)
4.  Bounded Execution Rounds & Message Dispatch
5.  Fault Isolation, Retries & Safe Degradation
6.  Evidence Aggregation (Distinguishing Claims vs Ground Evidence)
7.  Conflict Detection & Minority Opinion Preservation
8.  Consensus Synthesis & Uncertainty Quantification
9.  Capability Gate Authorization (DATA != AUTHORITY, AGENT != AUTHORITY)
10. Post-flight Model Weight Fingerprint Verification (FAIL CLOSED)
11. Governed Experience Capture to Continual Memory
12. Sanitized Public Federated Audit Trace Generation
"""

from dataclasses import asdict
import time
from typing import Dict, List, Optional, Any, Tuple
import uuid

import torch

from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.tokenizer import BPETokenizer
from chakrview.capability.gate import CapabilityGate, CapabilityAuthorizationError
from chakrview.capability.contract import CapabilityRequest
from chakrview.cognition.diagnostics.integrity import (
    CoreIntegrityGuard,
    InvariantViolationError,
    WeightMutationError,
)
from chakrview.cognition.adaptation.profiles import ResourceProfile
from chakrview.memory.engine import ContinualCognitionEngine
from chakrview.reasoning.engine import GovernedReasoningEngine
from chakrview.cognition.critical.engine import CriticalThinkingEngine
from chakrview.thinking.deliberation import DeliberationEngine
from chakrview.cognition.unified.models import (
    CognitiveTaskType,
    DecisionState,
)
from chakrview.cognition.federated.models import (
    AgentIdentity,
    AgentContract,
    AgentRole,
    AgentStatus,
    AgentTask,
    AgentMessage,
    MessageType,
    MessagePriority,
    FederatedConflictRecord,
    FederatedSynthesisCandidate,
    SafePublicFederatedTrace,
)
from chakrview.cognition.federated.policy import (
    FederatedExecutionPolicy,
    HARD_CEILING_FEDERATED_AGENTS,
    HARD_CEILING_FEDERATED_ROUNDS,
    HARD_CEILING_FEDERATED_MESSAGES,
)
from chakrview.cognition.federated.protocol import (
    FederatedProtocolValidator,
    MessageValidationError,
    TenantRoutingError,
)
from chakrview.cognition.federated.registry import (
    AgentRegistry,
    DuplicateAgentIdentityError,
    RegistryCapacityExceededError,
)
from chakrview.cognition.federated.decomposition import FederatedTaskDecomposer
from chakrview.cognition.federated.agents import (
    FederatedAgent,
    AnalystAgent,
    ResearcherAgent,
    CriticAgent,
    PlannerAgent,
    SynthesizerAgent,
    VerifierAgent,
    TenantIsolationError,
    AgentExecutionError,
)
from chakrview.cognition.federated.evidence import FederatedEvidenceAggregator
from chakrview.cognition.federated.conflict import FederatedConflictResolver
from chakrview.cognition.federated.synthesis import FederatedSynthesizer


class FederatedCognitionEngine:
    """
    Central coordinator orchestrating bounded multi-agent federated cognition.
    """

    def __init__(
        self,
        model: ChakrMicro,
        tokenizer: BPETokenizer,
        memory_engine: Optional[ContinualCognitionEngine] = None,
        capability_gate: Optional[CapabilityGate] = None,
        reasoning_engine: Optional[GovernedReasoningEngine] = None,
        critical_engine: Optional[CriticalThinkingEngine] = None,
        deliberation_engine: Optional[DeliberationEngine] = None,
        policy: Optional[FederatedExecutionPolicy] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.memory_engine = memory_engine or ContinualCognitionEngine()
        self.capability_gate = capability_gate
        self.reasoning_engine = reasoning_engine or GovernedReasoningEngine()
        self.critical_engine = critical_engine or CriticalThinkingEngine()
        self.deliberation_engine = deliberation_engine

        self.policy = policy or FederatedExecutionPolicy.from_resource_profile(ResourceProfile.STANDARD)
        self.registry = AgentRegistry(max_agents=HARD_CEILING_FEDERATED_AGENTS)
        self.decomposer = FederatedTaskDecomposer()
        self.evidence_aggregator = FederatedEvidenceAggregator()
        self.conflict_resolver = FederatedConflictResolver(critical_engine=self.critical_engine)
        self.synthesizer = FederatedSynthesizer()

        self.integrity_guard = CoreIntegrityGuard()
        self.baseline_weight_fingerprint = self.integrity_guard.compute_weight_fingerprint(self.model)

    def provision_standard_agents(self, tenant_id: str, session_id: str) -> Dict[AgentRole, FederatedAgent]:
        """Provision and register standard logical role agents for a tenant/session."""
        agents: Dict[AgentRole, FederatedAgent] = {}

        roles_to_setup = [
            (AgentRole.RESEARCHER, ResearcherAgent, [AgentRole.RESEARCHER]),
            (AgentRole.ANALYST, AnalystAgent, [AgentRole.ANALYST]),
            (AgentRole.CRITIC, CriticAgent, [AgentRole.CRITIC]),
            (AgentRole.PLANNER, PlannerAgent, [AgentRole.PLANNER]),
            (AgentRole.SYNTHESIZER, SynthesizerAgent, [AgentRole.SYNTHESIZER]),
            (AgentRole.VERIFIER, VerifierAgent, [AgentRole.VERIFIER]),
        ]

        for role, agent_cls, caps in roles_to_setup:
            agent_id = f"agent_{role.value.lower()}_{tenant_id[:6]}"
            ident = AgentIdentity(
                agent_id=agent_id,
                role=role,
                tenant_id=tenant_id,
                session_id=session_id,
            )
            contract = AgentContract(identity=ident)
            try:
                self.registry.register(contract)
            except DuplicateAgentIdentityError:
                pass  # already registered

            if agent_cls == ResearcherAgent:
                ag = ResearcherAgent(ident, contract, self.model, self.tokenizer, self.memory_engine)
            elif agent_cls == AnalystAgent:
                ag = AnalystAgent(ident, contract, self.model, self.tokenizer, self.reasoning_engine)
            elif agent_cls == CriticAgent:
                ag = CriticAgent(ident, contract, self.model, self.tokenizer, self.critical_engine)
            else:
                ag = agent_cls(ident, contract, self.model, self.tokenizer)

            agents[role] = ag

        return agents

    def execute_federated_cycle(
        self,
        objective: str,
        tenant_id: str = "default_tenant",
        session_id: str = "default_session",
        override_policy: Optional[FederatedExecutionPolicy] = None,
        custom_agents: Optional[Dict[str, FederatedAgent]] = None,
    ) -> Tuple[FederatedSynthesisCandidate, SafePublicFederatedTrace]:
        """
        Execute a bounded multi-agent federated cognitive cycle.
        """
        t0 = time.perf_counter()
        clean_prompt = objective.strip()
        eff_policy = override_policy or self.policy

        # 1. Pre-flight Model Invariant & Fingerprint Check (FAIL CLOSED)
        inv_check = self.integrity_guard.verify_model(self.model)
        if not inv_check.passed:
            raise InvariantViolationError(f"Frozen core invariant violated before federated cycle: {inv_check.message}")
        init_weight_hash = self.integrity_guard.compute_weight_fingerprint(self.model)
        if self.baseline_weight_fingerprint and init_weight_hash != self.baseline_weight_fingerprint:
            raise WeightMutationError(
                f"Baseline weight fingerprint mismatch detected! "
                f"Expected: {self.baseline_weight_fingerprint}, Got: {init_weight_hash}. HALTING."
            )

        task_id = f"fed_task_{uuid.uuid4().hex[:10]}"
        trace_events: List[Dict[str, Any]] = []

        # 2. Task Classification & Decomposition
        task_type = self._classify_task(clean_prompt)
        subtasks = self.decomposer.decompose(
            objective=clean_prompt,
            task_type=task_type,
            tenant_id=tenant_id,
            session_id=session_id,
            parent_task_id=task_id,
            max_depth=eff_policy.max_delegation_depth,
        )
        trace_events.append({"event": "TASK_DECOMPOSED", "subtasks_count": len(subtasks)})

        # 3. Agent Provisioning
        if custom_agents is not None:
            active_agents = custom_agents
        else:
            active_agents = self.provision_standard_agents(tenant_id, session_id)

        # 4. Cooperative Execution Rounds
        collected_messages: List[AgentMessage] = []
        failures_count = 0
        retries_count = 0
        total_message_count = 0

        self.evidence_aggregator.clear()
        rounds_executed = 0

        # Group tasks into execution rounds
        for round_idx in range(eff_policy.max_rounds):
            if rounds_executed >= eff_policy.max_rounds or not subtasks:
                break
            rounds_executed += 1
            round_messages_count = 0

            # Execute active subtasks
            current_batch = subtasks[:eff_policy.max_messages_per_round]
            subtasks = subtasks[eff_policy.max_messages_per_round:]

            for subtask in current_batch:
                if total_message_count >= eff_policy.max_total_messages:
                    break

                target_role = subtask.assigned_role
                agent = active_agents.get(target_role)

                if agent is None:
                    # Search by role in custom_agents if keyed by ID
                    for ag in active_agents.values():
                        if ag.identity.role == target_role:
                            agent = ag
                            break

                if agent is None:
                    failures_count += 1
                    trace_events.append({"event": "AGENT_MISSING", "role": target_role.value})
                    continue

                subtask.assigned_agent_id = agent.identity.agent_id

                # Fault Isolation: Execute with retry protection
                msg = None
                try:
                    msg = agent.execute_task(
                        subtask,
                        context={"prior_messages": collected_messages, "task_id": task_id},
                    )
                except Exception as exc:
                    failures_count += 1
                    trace_events.append({"event": "AGENT_FAILURE", "agent_id": agent.identity.agent_id, "error": str(exc)})
                    # Attempt safe retry once within policy
                    if retries_count < 2:
                        retries_count += 1
                        try:
                            msg = agent.execute_task(
                                subtask,
                                context={"prior_messages": collected_messages, "task_id": task_id, "retry": True},
                            )
                        except Exception:
                            # Isolate fault, continue with reduced federation
                            agent.identity.status = AgentStatus.FAILED
                            continue

                if msg is not None:
                    # Validate message envelope & tenant isolation
                    try:
                        FederatedProtocolValidator.validate_message(
                            message=msg,
                            expected_tenant_id=tenant_id,
                            expected_session_id=session_id,
                        )
                        collected_messages.append(msg)
                        self.evidence_aggregator.ingest_message(msg)
                        total_message_count += 1
                        round_messages_count += 1
                        trace_events.append({"event": "MESSAGE_ROUTED", "from": msg.sender_agent_id, "type": msg.message_type.value})
                    except (MessageValidationError, TenantRoutingError) as proto_err:
                        failures_count += 1
                        trace_events.append({"event": "MESSAGE_REJECTED", "error": str(proto_err)})

        # 5. Evidence Aggregation & Conflict Detection
        corroborated_ev = self.evidence_aggregator.get_corroborated_evidence()
        conflicts = self.conflict_resolver.detect_conflicts(
            task_id=task_id,
            messages=collected_messages,
        )

        # 6. Consensus / Synthesis Layer
        is_cap_task = (task_type == CognitiveTaskType.CAPABILITY)
        synthesis_candidate = self.synthesizer.synthesize(
            task_id=task_id,
            objective=clean_prompt,
            messages=collected_messages,
            corroborated_evidence=corroborated_ev,
            conflicts=conflicts,
            critical_counter_evidence=self.evidence_aggregator.counter_evidence,
            is_capability_task=is_cap_task,
        )

        # 7. Capability Boundary Protection
        if synthesis_candidate.recommended_decision_state == DecisionState.CAPABILITY_REQUIRED:
            if self.capability_gate:
                cap_id = "device_read"
                cap_req = CapabilityRequest(
                    capability_id=cap_id,
                    parameters={"query": clean_prompt},
                    context={"provenance_source": "federated_cognition", "owner_id": tenant_id},
                )
                try:
                    authorized = self.capability_gate.authorize(cap_req)
                    synthesis_candidate.primary_synthesis += " [Capability execution authorized by gate.]"
                except CapabilityAuthorizationError as auth_err:
                    synthesis_candidate.recommended_decision_state = DecisionState.SAFE_STOP
                    synthesis_candidate.uncertainty_notes = f"Capability denied by CapabilityGate: {auth_err}"
                    synthesis_candidate.primary_synthesis = f"Capability execution denied: {auth_err}"
                except Exception as exc:
                    synthesis_candidate.recommended_decision_state = DecisionState.SAFE_STOP
                    synthesis_candidate.uncertainty_notes = f"Capability execution prevented: {exc}"
                    synthesis_candidate.primary_synthesis = f"Capability execution prevented: {exc}"

        # 8. Post-flight Model Weight Fingerprint Verification (FAIL CLOSED)
        post_weight_hash = self.integrity_guard.compute_weight_fingerprint(self.model)
        if init_weight_hash != post_weight_hash or post_weight_hash != self.baseline_weight_fingerprint:
            raise WeightMutationError(
                f"FATAL: Runtime weight mutation detected during federated cycle! "
                f"Initial: {init_weight_hash}, Post: {post_weight_hash}. HALTING."
            )

        # 9. Governed Experience Capture
        self.memory_engine.record_experience(
            tenant_id=tenant_id,
            session_id=session_id,
            situation=clean_prompt,
            action_or_response=f"FederatedCognition(agents={len(active_agents)}, rounds={rounds_executed})",
            outcome=synthesis_candidate.primary_synthesis[:200],
            task_id=task_id,
            metadata={
                "task_id": task_id,
                "decision_state": synthesis_candidate.recommended_decision_state.value,
                "weights_modified": False,
            },
        )
        self.memory_engine.consolidate(tenant_id=tenant_id)

        # 10. Sanitized Public Federated Trace
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        trace_fp = FederatedProtocolValidator.compute_trace_fingerprint(
            events=trace_events,
            task_id=task_id,
            tenant_id=tenant_id,
        )

        participating = [
            {"agent_id": ag.identity.agent_id, "role": ag.identity.role.value}
            for ag in active_agents.values()
        ]

        public_trace = SafePublicFederatedTrace(
            trace_id=f"ftrace_{uuid.uuid4().hex[:10]}",
            task_id=task_id,
            tenant_id=tenant_id,
            session_id=session_id,
            participating_agents=participating,
            execution_rounds=rounds_executed,
            total_messages_exchanged=total_message_count,
            evidence_items_count=len(corroborated_ev),
            conflicts_detected_count=len(conflicts),
            decision_state=synthesis_candidate.recommended_decision_state.value,
            latency_ms=elapsed_ms,
            failures_count=failures_count,
            retries_count=retries_count,
            hardware_profile=eff_policy.profile.value,
            trace_fingerprint=trace_fp,
            status="COMPLETED" if failures_count == 0 else "COMPLETED_WITH_DEGRADATION",
            weights_modified=False,
        )

        return synthesis_candidate, public_trace

    def _classify_task(self, prompt: str) -> CognitiveTaskType:
        """Route prompt into appropriate task classification."""
        p_lower = prompt.lower()
        if any(w in p_lower for w in ("sensor", "device", "execute capability", "turn on", "hardware", "read_device")):
            return CognitiveTaskType.CAPABILITY
        if any(w in p_lower for w in ("calculate", "analyze", "derive", "prove", "step by step", "solve", "spike")):
            return CognitiveTaskType.ANALYTICAL
        if any(w in p_lower for w in ("choose", "decide", "option", "select", "recommend", "evaluate")):
            return CognitiveTaskType.DECISION
        if any(w in p_lower for w in ("what is", "when did", "who is", "state the", "definition", "atomic number")):
            return CognitiveTaskType.FACTUAL
        return CognitiveTaskType.GENERAL
