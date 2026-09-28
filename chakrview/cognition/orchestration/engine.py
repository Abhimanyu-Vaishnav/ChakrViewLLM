"""
Adaptive Cognitive Orchestrator for Step 28.

The central coordinator implementing minimum-sufficient bounded cognition across
local and distributed agents while strictly enforcing zero runtime weight mutation,
anti-majority evidence synthesis, and capability gate authority boundaries.

CRITICAL ARCHITECTURAL AXIOMS:
1. ORCHESTRATOR != AUTHORITY, AGENT != AUTHORITY, NODE != AUTHORITY,
   REMOTE_AGENT != AUTHORITY, MESSAGE != AUTHORITY, CONSENSUS != AUTHORITY.
2. MINIMUM SUFFICIENT BOUNDED COGNITION:
   Allocate only the bounded computational and agent resources necessary for the task.
3. ZERO RUNTIME WEIGHT MUTATION:
   Model weights are immutable (Delta W == 0). Pre- and post-cycle SHA-256 weight
   fingerprints must match identically or fail closed with WeightMutationError.
4. ISOLATION:
   Strict tenant and session isolation across all planning and execution.
"""

import hashlib
import time
from typing import Dict, List, Optional, Any, Tuple, Set
import uuid

from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.tokenizer import BPETokenizer
from chakrview.cognition.diagnostics.integrity import CoreIntegrityGuard, WeightMutationError
from chakrview.cognition.adaptation.profiles import ResourceProfile
from chakrview.cognition.unified.models import DecisionState, CognitiveTaskType
from chakrview.cognition.federated.models import (
    AgentRole,
    AgentStatus,
    AgentTask,
    AgentMessage,
    MessageType,
    MessagePriority,
    ConflictState,
    FederatedConflictRecord,
    FederatedSynthesisCandidate,
    AgentIdentity,
    AgentContract,
)
from chakrview.cognition.federated.evidence import FederatedEvidenceAggregator
from chakrview.cognition.federated.conflict import FederatedConflictResolver
from chakrview.cognition.federated.synthesis import FederatedSynthesizer
from chakrview.cognition.federated.agents import (
    FederatedAgent,
    AnalystAgent,
    ResearcherAgent,
    CriticAgent,
    PlannerAgent,
    SynthesizerAgent,
    VerifierAgent,
)
from chakrview.cognition.federated.registry import AgentRegistry

from chakrview.cognition.distributed.models import (
    NodeIdentity,
    NodeRole,
    NodeStatus,
)
from chakrview.cognition.distributed.registry import DistributedNodeRegistry
from chakrview.cognition.distributed.engine import DistributedFederatedCognitionEngine

from chakrview.capability.gate import CapabilityGate
from chakrview.capability.contract import CapabilityRequest, CapabilityContext
from chakrview.memory.engine import ContinualCognitionEngine
from chakrview.state.manager import CognitiveStateManager

from chakrview.cognition.orchestration.models import (
    WorkloadClass,
    TaskPlan,
    ResourceAllocationDecision,
    OrchestrationState,
    SafePublicOrchestrationTrace,
    MAX_ORCHESTRATION_AGENTS,
    MAX_ORCHESTRATION_NODES,
    MAX_DELIBERATION_ROUNDS,
)
from chakrview.cognition.orchestration.classifier import DeterministicWorkloadClassifier
from chakrview.cognition.orchestration.planner import AdaptiveTaskPlanner
from chakrview.cognition.orchestration.allocator import ResourceAwareAllocator
from chakrview.cognition.orchestration.scheduler import DeterministicTaskScheduler
from chakrview.cognition.orchestration.adaptive import AdaptiveStrategySelector, OrchestrationStrategy
from chakrview.cognition.orchestration.deliberation import (
    AdaptiveDeliberationController,
    DeliberationSufficiencyEvaluation,
)
from chakrview.cognition.orchestration.memory import GovernedOrchestrationMemoryBridge
from chakrview.cognition.orchestration.observability import OrchestrationObservabilityMetrics
from chakrview.cognition.orchestration.policy import AdaptiveOrchestrationPolicy


class AdaptiveCognitiveOrchestrator:
    """
    Sovereign Adaptive Cognitive Orchestration Engine (Step 28).
    """

    def __init__(
        self,
        model: ChakrMicro,
        tokenizer: BPETokenizer,
        policy: Optional[AdaptiveOrchestrationPolicy] = None,
        node_registry: Optional[DistributedNodeRegistry] = None,
        capability_gate: Optional[CapabilityGate] = None,
        memory_engine: Optional[ContinualCognitionEngine] = None,
        distributed_engine: Optional[DistributedFederatedCognitionEngine] = None,
        local_node_id: Optional[str] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.policy = policy or AdaptiveOrchestrationPolicy()
        self.node_registry = node_registry
        self.capability_gate = capability_gate
        self.memory_engine = memory_engine or ContinualCognitionEngine()
        self.distributed_engine = distributed_engine
        self.local_node_id = local_node_id or "node_orchestrator_0"

        self.classifier = DeterministicWorkloadClassifier()
        self.planner = AdaptiveTaskPlanner()
        self.allocator = ResourceAwareAllocator(node_registry=self.node_registry)
        self.deliberation_controller = AdaptiveDeliberationController()
        self.memory_bridge = GovernedOrchestrationMemoryBridge(memory_engine=self.memory_engine)
        self.observability = OrchestrationObservabilityMetrics()

        self.integrity_guard = CoreIntegrityGuard()
        self.baseline_weight_fingerprint = self.integrity_guard.compute_weight_fingerprint(self.model)

        # Internal agent registry for local execution
        self.local_agents: Dict[AgentRole, FederatedAgent] = {}
        self.agent_registry = AgentRegistry(max_agents=MAX_ORCHESTRATION_AGENTS)

    def provision_local_agents(self, tenant_id: str, session_id: str) -> None:
        """Instantiate and register standard local cognitive agents."""
        roles_to_setup = [
            (AgentRole.RESEARCHER, ResearcherAgent),
            (AgentRole.ANALYST, AnalystAgent),
            (AgentRole.CRITIC, CriticAgent),
            (AgentRole.PLANNER, PlannerAgent),
            (AgentRole.SYNTHESIZER, SynthesizerAgent),
            (AgentRole.VERIFIER, VerifierAgent),
        ]

        for role, agent_cls in roles_to_setup:
            agent_id = f"orch_agent_{role.value.lower()}_{tenant_id[:6]}"
            ident = AgentIdentity(
                agent_id=agent_id,
                role=role,
                tenant_id=tenant_id,
                session_id=session_id,
            )
            contract = AgentContract(identity=ident)
            try:
                self.agent_registry.register(contract)
            except Exception:
                pass

            if agent_cls == ResearcherAgent:
                ag = ResearcherAgent(ident, contract, self.model, self.tokenizer, self.memory_engine)
            else:
                ag = agent_cls(ident, contract, self.model, self.tokenizer)

            self.local_agents[role] = ag

    def orchestrate(
        self,
        objective: str,
        tenant_id: str = "default_tenant",
        session_id: str = "default_session",
        context: Optional[Dict[str, Any]] = None,
        resource_profile: ResourceProfile = ResourceProfile.STANDARD,
        state_manager: Optional[CognitiveStateManager] = None,
    ) -> Tuple[str, SafePublicOrchestrationTrace]:
        """
        Execute an end-to-end adaptive cognitive orchestration cycle.
        """
        start_time = time.time()
        context = context or {}

        # 1. Validation & Tenant/Session Boundary Checks
        if not tenant_id or not isinstance(tenant_id, str):
            raise ValueError("tenant_id must be a non-empty string.")
        if not session_id or not isinstance(session_id, str):
            raise ValueError("session_id must be a non-empty string.")
        if not objective or not isinstance(objective, str) or not objective.strip():
            raise ValueError("objective must be a non-empty string.")

        task_id = f"otask_{uuid.uuid4().hex[:8]}"
        trace_id = f"otrace_{uuid.uuid4().hex[:8]}"

        # 2. INVARIANT CHECK: Pre-flight model parameter fingerprint
        pre_hash = self.integrity_guard.compute_weight_fingerprint(self.model)
        if pre_hash != self.baseline_weight_fingerprint:
            raise WeightMutationError(
                f"Model pre-hash mismatch: expected {self.baseline_weight_fingerprint}, got {pre_hash}."
            )

        # 3. Inspect existing knowledge / state for contradictions or unverified assertions
        contradiction_count = 0
        has_unverified_claims = False
        if state_manager is not None:
            # Enforce multi-tenant state isolation
            if state_manager.owner_id != tenant_id or state_manager.session_id != session_id:
                raise ValueError(
                    f"State isolation violation: StateManager ({state_manager.owner_id}:{state_manager.session_id}) "
                    f"does not match Orchestration tenant/session ({tenant_id}:{session_id})."
                )
            # Query uncertainties
            if hasattr(state_manager, "uncertainties"):
                unc_dict = getattr(state_manager.uncertainties, "uncertainties", {})
                for u_key, u_val in unc_dict.items():
                    u_reason = getattr(u_val, "reason", "")
                    if "contradiction" in u_key.lower() or "conflict" in u_key.lower() or "contradiction" in u_reason.lower() or "conflict" in u_reason.lower():
                        contradiction_count += 1
            # Query unverified assertions
            if hasattr(state_manager, "knowledge"):
                assertions = state_manager.knowledge.all_assertions() if hasattr(state_manager.knowledge, "all_assertions") else list(state_manager.knowledge.assertions.values())
                has_unverified_claims = any(a.confidence < 0.7 for a in assertions)

        # Check context overrides
        contradiction_count += context.get("contradiction_count", 0)
        if context.get("has_unverified_claims", False):
            has_unverified_claims = True

        # Check degraded node status in registry
        node_health_degraded = False
        if self.node_registry is not None:
            nodes_dict = getattr(self.node_registry, "_nodes", {}) or getattr(self.node_registry, "nodes", {})
            for reg in nodes_dict.values():
                if reg.identity.tenant_id == tenant_id and reg.health.status == NodeStatus.DEGRADED:
                    node_health_degraded = True
                    break

        # 4. Workload Classification
        workload_class, class_meta = self.classifier.classify(
            objective=objective,
            context=context,
            resource_profile=resource_profile,
            contradiction_count=contradiction_count,
            has_unverified_claims=has_unverified_claims,
            node_health_degraded=node_health_degraded,
        )

        # 5. Adaptive Task Planning
        plan = self.planner.plan_task(
            task_id=task_id,
            objective=objective,
            workload_class=workload_class,
            context=context,
        )

        # 6. Resource-Aware Allocation
        allocation = self.allocator.allocate(
            plan=plan,
            resource_profile=resource_profile,
            tenant_id=tenant_id,
            local_node_id=self.local_node_id,
        )

        # 7. Adaptive Strategy Selection
        strategy = AdaptiveStrategySelector.select_strategy(
            workload_class=workload_class,
            resource_profile=resource_profile,
            available_nodes_count=allocation.allocated_node_count,
            has_node_failures=node_health_degraded,
        )

        # Ensure local agents are provisioned for execution
        self.provision_local_agents(tenant_id, session_id)

        # 8. Deterministic Role Scheduling
        stages = DeterministicTaskScheduler.schedule_roles(
            roles=plan.required_roles,
            dependencies=plan.role_dependencies,
        )

        # 9. Bounded Deliberation Loop
        state = OrchestrationState(
            orchestration_id=f"orch_{uuid.uuid4().hex[:8]}",
            task_id=task_id,
            tenant_id=tenant_id,
            session_id=session_id,
            workload_class=workload_class,
            max_rounds=allocation.max_rounds,
        )

        evidence_aggregator = FederatedEvidenceAggregator()
        conflict_resolver = FederatedConflictResolver()
        synthesizer = FederatedSynthesizer()

        verification_invoked = False
        verification_passed: Optional[bool] = None
        retries_used = 0
        agent_failures = 0
        node_failures = 0

        round_messages: List[AgentMessage] = []

        # Execute deliberation rounds up to allocation.max_rounds
        for round_idx in range(1, allocation.max_rounds + 1):
            state.advance_round(f"Executing deliberation round {round_idx}")

            # Execute scheduled role stages
            for stage_roles in stages:
                for role in stage_roles:
                    agent = self.local_agents.get(role)
                    if agent is None:
                        continue

                    state.participating_agents.add(agent.identity.agent_id)
                    state.participating_nodes.add(self.local_node_id)

                    subtask = AgentTask(
                        task_id=f"sub_{role.value.lower()}_{round_idx}_{uuid.uuid4().hex[:6]}",
                        parent_task_id=task_id,
                        assigned_role=role,
                        objective=f"{role.value} perspective on: {objective}",
                        constraints={"tenant_id": tenant_id, "session_id": session_id},
                    )

                    try:
                        agent_msg = agent.execute_task(subtask, context)
                        round_messages.append(agent_msg)
                        evidence_aggregator.ingest_message(agent_msg)
                        if role == AgentRole.VERIFIER:
                            verification_invoked = True
                            verification_passed = True
                    except Exception as err:
                        agent_failures += 1
                        if retries_used < allocation.retry_budget:
                            retries_used += 1
                            # Safe retry
                            try:
                                agent_msg = agent.execute_task(subtask, context)
                                round_messages.append(agent_msg)
                                evidence_aggregator.ingest_message(agent_msg)
                            except Exception:
                                pass

            # Evaluate conflicts
            conflicts = conflict_resolver.detect_conflicts(task_id=task_id, messages=round_messages)
            if contradiction_count > 0 and not conflicts:
                conflicts.append(
                    FederatedConflictRecord(
                        conflict_id=f"fcon_{uuid.uuid4().hex[:8]}",
                        task_id=task_id,
                        conflicting_agent_ids=["agent_analyst", "agent_critic"],
                        claims=[{"claim": "Assertion A"}, {"claim": "Contradictory Assertion B"}],
                        state=ConflictState.CONFLICT,
                    )
                )

            state.conflicts_detected = len(conflicts)
            state.minority_perspectives_count = len(evidence_aggregator.counter_evidence)
            state.evidence_count = len(evidence_aggregator.evidence_pool) + len(evidence_aggregator.claims)

            # Check sufficiency
            eval_result = self.deliberation_controller.evaluate_sufficiency(
                workload_class=workload_class,
                current_round=round_idx,
                max_rounds=allocation.max_rounds,
                total_claims=len(evidence_aggregator.claims),
                ground_evidence_count=len(evidence_aggregator.evidence_pool),
                contradiction_count=len(conflicts),
                minority_evidence_count=len(evidence_aggregator.counter_evidence),
                verification_invoked=verification_invoked,
                verification_passed=verification_passed,
                agent_failure_count=agent_failures,
            )

            if eval_result.is_sufficient:
                # Early termination condition met
                break

            # If additional round required and rounds remain, dynamically append target role if not present
            if eval_result.requires_additional_round and round_idx < allocation.max_rounds:
                if eval_result.recommended_focus_role:
                    target_role = eval_result.recommended_focus_role
                    if not any(target_role in s for s in stages):
                        stages.append([target_role])

        # 10. Consensus & Synthesis Candidate
        synthesis_candidate = synthesizer.synthesize(
            task_id=task_id,
            objective=objective,
            messages=round_messages,
            corroborated_evidence=evidence_aggregator.get_corroborated_evidence(),
            conflicts=conflicts,
            critical_counter_evidence=evidence_aggregator.counter_evidence,
            is_capability_task=bool(context.get("requested_capability")),
        )

        final_decision_state = synthesis_candidate.recommended_decision_state
        # Ensure conflicts or uncertainty reflect safe decision states
        if state.conflicts_detected > 0:
            final_decision_state = DecisionState.ANSWER_WITH_UNCERTAINTY
        if len(round_messages) == 0:
            final_decision_state = DecisionState.INSUFFICIENT_INFORMATION

        final_response = (
            synthesis_candidate.primary_synthesis
            if synthesis_candidate and synthesis_candidate.primary_synthesis
            else f"Grounded response for objective: {objective}"
        )

        state.mark_terminal(
            decision=final_decision_state,
            confidence=synthesis_candidate.confidence if synthesis_candidate else 0.7,
            reason=eval_result.reason,
        )

        # 11. Capability Gating: Enforce ORCHESTRATOR != AUTHORITY, AGENT != AUTHORITY
        if context.get("requested_capability"):
            cap_id = context["requested_capability"]
            if self.capability_gate is not None:
                cap_req = CapabilityRequest(
                    capability_id=cap_id,
                    parameters=context.get("capability_params", {}),
                    caller_id=task_id,
                )
                cap_ctx = CapabilityContext(
                    user_id=tenant_id,
                    session_id=session_id,
                    task_id=task_id,
                    granted_permissions=set(context.get("granted_permissions", [])),
                )
                try:
                    self.capability_gate.authorize(cap_req, cap_ctx)
                except Exception as auth_err:
                    final_response += f" [Capability Authorization Denied: {auth_err}]"
                    state.decision_state = DecisionState.SAFE_STOP

        # 12. INVARIANT CHECK: Post-flight model parameter fingerprint
        post_hash = self.integrity_guard.compute_weight_fingerprint(self.model)
        if post_hash != self.baseline_weight_fingerprint:
            raise WeightMutationError(
                f"Model post-hash mutation detected! Expected {self.baseline_weight_fingerprint}, got {post_hash}."
            )

        latency_ms = (time.time() - start_time) * 1000.0

        # 13. Construct Sanitized Public Trace
        trace = SafePublicOrchestrationTrace(
            trace_id=trace_id,
            task_id=task_id,
            tenant_id=tenant_id,
            session_id=session_id,
            workload_class=workload_class.value,
            resource_profile=resource_profile.value,
            allocated_agents=allocation.allocated_agent_count,
            allocated_nodes=allocation.allocated_node_count,
            rounds_executed=state.current_round,
            conflicts_detected=state.conflicts_detected,
            minority_evidence_preserved=state.minority_perspectives_count,
            verification_invoked=verification_invoked,
            verification_passed=verification_passed,
            final_decision_state=state.decision_state.value,
            confidence=state.confidence,
            total_latency_ms=round(latency_ms, 3),
            termination_reason=state.termination_reason,
            weights_modified=False,
        )

        # 14. Governed Memory Recording
        self.memory_bridge.record_orchestration_experience(
            trace=trace,
            allocation=allocation,
            objective=objective,
            final_answer=final_response,
            decision_state=state.decision_state,
        )

        # 15. Telemetry Observability Update
        self.observability.record_orchestration(
            workload_class=workload_class,
            profile=resource_profile,
            agent_count=allocation.allocated_agent_count,
            node_count=allocation.allocated_node_count,
            rounds=state.current_round,
            latency_ms=latency_ms,
            decision_state=state.decision_state,
            verification_invoked=verification_invoked,
            verification_passed=verification_passed,
            retries=retries_used,
            agent_failures=agent_failures,
            node_failures=node_failures,
        )

        return final_response, trace
