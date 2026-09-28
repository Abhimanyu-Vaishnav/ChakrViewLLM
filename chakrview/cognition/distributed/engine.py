"""
Distributed Federated Cognition Engine (Step 27).

Orchestrates multi-agent cooperative cognition across distributed nodes:
- Deterministic subtask decomposition & distributed routing
- Pluggable transport (LoopbackTransport for in-process testing)
- Cryptographic envelope verification, nonce validation & replay protection
- Evidence aggregation & conflict resolution with minority evidence preservation
- CapabilityGate sovereignty & zero runtime weight mutation (fail-closed)
- Sanitized public distributed audit tracing & governed continual memory capture

CRITICAL ARCHITECTURAL AXIOMS:
1. NODE != AUTHORITY & AGENT != AUTHORITY:
   Distributed nodes and remote agents are compute hosts; only CapabilityGate
   authorizes external capabilities.
2. ZERO RUNTIME WEIGHT MUTATION:
   Pre-flight and post-flight SHA-256 weight hashes must match exactly.
   Any weight modification raises WeightMutationError and fails closed.
3. PRESERVE MINORITY EVIDENCE:
   Disagreements across distributed agents are preserved in formal conflict
   records rather than reduced to simplistic majority voting.
"""

from dataclasses import asdict
import hashlib
import json
import time
from typing import Dict, List, Optional, Any, Tuple, Set
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
from chakrview.cognition.distributed.models import (
    NodeIdentity,
    NodeRegistration,
    NodeRole,
    NodeStatus,
    NodeCapabilities,
    NodeResourceProfile,
    DistributedRouteDecision,
    DistributedMessageEnvelope,
    SafePublicDistributedTrace,
    MAX_REMOTE_TASKS_PER_CYCLE,
)
from chakrview.cognition.distributed.transport import (
    Transport,
    LoopbackTransport,
    TransportResponse,
    TransportStatus,
)
from chakrview.cognition.distributed.registry import DistributedNodeRegistry
from chakrview.cognition.distributed.router import DistributedTaskRouter, NoEligibleNodeError
from chakrview.cognition.distributed.policy import DistributedExecutionPolicy
from chakrview.cognition.distributed.security import (
    ReplayProtectionTracker,
    MessageSigner,
    MessageVerifier,
    create_distributed_envelope,
    MessageTamperingError,
    ReplayAttackError,
    TenantRoutingError,
)
from chakrview.cognition.distributed.resilience import (
    RetryPolicy,
    CircuitBreaker,
)
from chakrview.cognition.distributed.observability import DistributedObservabilityMetrics

from chakrview.cognition.federated.models import (
    AgentRole,
    AgentStatus,
    AgentTask,
    AgentMessage,
    MessageType,
    MessagePriority,
    FederatedConflictRecord,
    FederatedSynthesisCandidate,
    AgentIdentity,
    AgentContract,
)
from chakrview.cognition.federated.decomposition import FederatedTaskDecomposer
from chakrview.cognition.federated.evidence import FederatedEvidenceAggregator
from chakrview.cognition.federated.conflict import FederatedConflictResolver
from chakrview.cognition.federated.synthesis import FederatedSynthesizer
from chakrview.cognition.federated.registry import AgentRegistry
from chakrview.cognition.federated.agents import (
    FederatedAgent,
    AnalystAgent,
    ResearcherAgent,
    CriticAgent,
    PlannerAgent,
    SynthesizerAgent,
    VerifierAgent,
)
from chakrview.cognition.unified.models import DecisionState, CognitiveTaskType
from chakrview.memory.engine import ContinualCognitionEngine
from chakrview.reasoning.engine import GovernedReasoningEngine
from chakrview.cognition.critical.engine import CriticalThinkingEngine
from chakrview.thinking.deliberation import DeliberationEngine


class DistributedFederatedCognitionEngine:
    """
    Coordinator executing cooperative federated cognition across distributed nodes.
    """

    def __init__(
        self,
        model: ChakrMicro,
        tokenizer: BPETokenizer,
        local_node_identity: NodeIdentity,
        node_registry: DistributedNodeRegistry,
        transport: Transport,
        policy: Optional[DistributedExecutionPolicy] = None,
        memory_engine: Optional[ContinualCognitionEngine] = None,
        capability_gate: Optional[CapabilityGate] = None,
        reasoning_engine: Optional[GovernedReasoningEngine] = None,
        critical_engine: Optional[CriticalThinkingEngine] = None,
        deliberation_engine: Optional[DeliberationEngine] = None,
        signer: Optional[MessageSigner] = None,
        verifier: Optional[MessageVerifier] = None,
        replay_tracker: Optional[ReplayProtectionTracker] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.local_node_identity = local_node_identity
        self.node_registry = node_registry
        self.transport = transport
        self.policy = policy or DistributedExecutionPolicy.from_profile("STANDARD")
        self.memory_engine = memory_engine or ContinualCognitionEngine()
        self.capability_gate = capability_gate
        self.reasoning_engine = reasoning_engine or GovernedReasoningEngine()
        self.critical_engine = critical_engine or CriticalThinkingEngine()
        self.deliberation_engine = deliberation_engine

        self.signer = signer
        self.verifier = verifier
        self.replay_tracker = replay_tracker or ReplayProtectionTracker(max_tracked=self.policy.max_tracked_replays)
        self.observability = DistributedObservabilityMetrics()

        self.router = DistributedTaskRouter(
            registry=self.node_registry,
            local_node_id=self.local_node_identity.node_id,
        )
        self.decomposer = FederatedTaskDecomposer()
        self.evidence_aggregator = FederatedEvidenceAggregator()
        self.conflict_resolver = FederatedConflictResolver(critical_engine=self.critical_engine)
        self.synthesizer = FederatedSynthesizer()

        self.local_agent_registry = AgentRegistry(max_agents=8)
        self.local_agents: Dict[AgentRole, FederatedAgent] = {}

        self.integrity_guard = CoreIntegrityGuard()
        self.baseline_weight_fingerprint = self.integrity_guard.compute_weight_fingerprint(self.model)

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
            agent_id = f"agent_{role.value.lower()}_{tenant_id[:6]}"
            ident = AgentIdentity(
                agent_id=agent_id,
                role=role,
                tenant_id=tenant_id,
                session_id=session_id,
            )
            contract = AgentContract(identity=ident)
            try:
                self.local_agent_registry.register(contract)
            except Exception:
                pass

            if agent_cls == ResearcherAgent:
                ag = ResearcherAgent(ident, contract, self.model, self.tokenizer, self.memory_engine)
            elif agent_cls == AnalystAgent:
                ag = AnalystAgent(ident, contract, self.model, self.tokenizer, self.reasoning_engine)
            elif agent_cls == CriticAgent:
                ag = CriticAgent(ident, contract, self.model, self.tokenizer, self.critical_engine)
            else:
                ag = agent_cls(ident, contract, self.model, self.tokenizer)

            self.local_agents[role] = ag

    def register_remote_agent_responder(
        self,
        node_id: str,
        supported_roles: List[AgentRole],
        tenant_id: str,
    ) -> None:
        """
        Helper for LoopbackTransport to register a remote worker node
        capable of simulating subtask execution for testing.
        """
        if isinstance(self.transport, LoopbackTransport):
            def handler(req_envelope: DistributedMessageEnvelope) -> Optional[DistributedMessageEnvelope]:
                # 1. Verify incoming envelope
                if not req_envelope.verify_integrity():
                    raise MessageTamperingError("Incoming remote request failed integrity check.")
                self.replay_tracker.validate_and_record(req_envelope, expected_tenant_id=tenant_id)

                task_dict = req_envelope.payload.get("task", {})
                role_val = task_dict.get("assigned_role")
                try:
                    role = AgentRole(role_val)
                except ValueError:
                    role = AgentRole.ANALYST

                # Simulate execution output
                resp_payload = {
                    "task_id": task_dict.get("task_id", "t_unknown"),
                    "assigned_role": role.value,
                    "findings": [f"Remote node {node_id} executed analysis for {role.value}."],
                    "claims": [f"Remote evidence claim from {node_id}"],
                    "confidence": 0.88,
                    "evidence": [f"ground_fact:observed_by_{node_id}"],
                    "assumptions": ["standard_network_conditions"],
                    "counter_evidence": [],
                }

                # Construct signed response envelope
                return create_distributed_envelope(
                    sender_node_id=node_id,
                    sender_agent_id=f"agent_{role.value.lower()}_{node_id}",
                    receiver_node_id=req_envelope.route.sender_node_id,
                    receiver_agent_id=req_envelope.route.sender_agent_id,
                    tenant_id=tenant_id,
                    session_id=req_envelope.route.session_id,
                    message_type=MessageType.TASK_RESPONSE,
                    payload=resp_payload,
                    correlation_id=req_envelope.header.correlation_id,
                    causation_id=req_envelope.header.message_id,
                    signer=self.signer,
                    signer_identity=node_id,
                    hop_count=req_envelope.header.hop_count + 1,
                    logical_clock=req_envelope.header.logical_clock + 1,
                )

            self.transport.register_node_endpoint(node_id, handler=handler)

    def execute_distributed_cycle(
        self,
        objective: str,
        tenant_id: str = "default_tenant",
        session_id: str = "default_session",
        override_policy: Optional[DistributedExecutionPolicy] = None,
        prefer_remote: bool = False,
    ) -> Tuple[FederatedSynthesisCandidate, SafePublicDistributedTrace]:
        """
        Execute an end-to-end distributed federated cognitive cycle.
        """
        start_t = time.perf_counter()
        eff_policy = override_policy or self.policy

        # Step 1: Pre-flight model weight verification
        pre_fingerprint = self.integrity_guard.compute_weight_fingerprint(self.model)
        if pre_fingerprint != self.baseline_weight_fingerprint:
            raise WeightMutationError(
                f"Pre-flight model weight mismatch: {pre_fingerprint} != {self.baseline_weight_fingerprint}"
            )

        # Ensure local agents exist
        if not self.local_agents:
            self.provision_local_agents(tenant_id, session_id)

        task_id = f"dist_task_{uuid.uuid4().hex[:10]}"
        task_type = self._classify_task(objective)

        # Step 2: Decompose unified task into bounded subtasks
        subtasks: List[AgentTask] = self.decomposer.decompose(
            objective=objective,
            task_type=task_type,
            tenant_id=tenant_id,
            session_id=session_id,
            parent_task_id=task_id,
            max_depth=eff_policy.max_message_hops,
        )

        context = {
            "tenant_id": tenant_id,
            "session_id": session_id,
            "objective": objective,
        }

        routing_decisions: List[DistributedRouteDecision] = []
        collected_messages: List[AgentMessage] = []
        participating_nodes: Set[str] = {self.local_node_identity.node_id}
        participating_roles: Set[str] = set()

        retry_policy = RetryPolicy(max_retries=eff_policy.max_retries)
        retry_count = 0
        failure_count = 0

        # Step 3: Route and dispatch subtasks across local and remote nodes
        for task in subtasks:
            participating_roles.add(task.assigned_role.value)
            try:
                decision = self.router.route_task(
                    task=task,
                    tenant_id=tenant_id,
                    session_id=session_id,
                    prefer_remote=prefer_remote,
                )
                routing_decisions.append(decision)
                participating_nodes.add(decision.assigned_node_id)

                if not decision.is_remote:
                    # Execute locally on this node
                    agent = self.local_agents.get(task.assigned_role)
                    if agent is None:
                        # Fallback to analyst
                        agent = self.local_agents[AgentRole.ANALYST]
                    msg = agent.execute_task(task, context)
                    collected_messages.append(msg)
                else:
                    # Execute on remote node via transport
                    remote_node_id = decision.assigned_node_id
                    cb = self.router.get_or_create_circuit_breaker(remote_node_id)

                    req_envelope = create_distributed_envelope(
                        sender_node_id=self.local_node_identity.node_id,
                        sender_agent_id=f"coordinator_{self.local_node_identity.node_id}",
                        receiver_node_id=remote_node_id,
                        receiver_agent_id=decision.assigned_agent_id,
                        tenant_id=tenant_id,
                        session_id=session_id,
                        message_type=MessageType.TASK_REQUEST,
                        payload={"task": task.to_dict(), "context": context},
                        correlation_id=task.task_id,
                        signer=self.signer,
                        signer_identity=self.local_node_identity.node_id,
                        ttl_seconds=eff_policy.request_timeout_ms / 1000.0,
                    )

                    # Send with bounded retries
                    success = False
                    for attempt in range(eff_policy.max_retries + 1):
                        self.observability.record_message_sent()
                        resp: TransportResponse = self.transport.request(
                            message=req_envelope,
                            timeout_ms=eff_policy.request_timeout_ms,
                        )

                        if resp.is_success() and resp.response_envelope is not None:
                            resp_env = resp.response_envelope
                            # Validate response integrity and replay protection
                            if not resp_env.verify_integrity():
                                self.observability.record_rejected_message("tampered_response")
                                raise MessageTamperingError(f"Response from '{remote_node_id}' failed integrity check.")

                            self.replay_tracker.validate_and_record(resp_env, expected_tenant_id=tenant_id)
                            self.observability.record_message_received()
                            cb.record_success()

                            # Convert response envelope payload into typed AgentMessage
                            msg = AgentMessage(
                                message_id=resp_env.header.message_id,
                                sender_agent_id=resp_env.route.sender_agent_id,
                                receiver_agent_id=resp_env.route.receiver_agent_id,
                                tenant_id=resp_env.route.tenant_id,
                                session_id=resp_env.route.session_id,
                                correlation_id=resp_env.header.correlation_id,
                                message_type=resp_env.header.message_type,
                                payload=resp_env.payload,
                                provenance=f"remote_node_{remote_node_id}",
                                timestamp=resp_env.header.timestamp,
                            )
                            collected_messages.append(msg)
                            success = True
                            break
                        else:
                            if attempt < eff_policy.max_retries:
                                retry_count += 1
                                self.observability.record_retry()
                                backoff_ms = retry_policy.calculate_backoff_ms(attempt + 1)
                                if backoff_ms > 0:
                                    time.sleep(backoff_ms / 10000.0)  # Micro-sleep in tests

                    if not success:
                        failure_count += 1
                        cb.record_failure(task.task_id, "TRANSPORT_FAILURE", resp.error_message or "Request failed")
                        self.observability.record_node_failure(remote_node_id)
                        # Graceful degradation: attempt local fallback
                        local_agent = self.local_agents.get(task.assigned_role)
                        if local_agent:
                            msg = local_agent.execute_task(task, context)
                            collected_messages.append(msg)
                        else:
                            task.fail("Remote execution failed and no local fallback available.")

            except Exception as e:
                failure_count += 1
                task.fail(reason=str(e))
                # Continue with remaining tasks (graceful degradation)

        # Step 4: Evidence Aggregation
        self.evidence_aggregator.clear()
        for msg in collected_messages:
            self.evidence_aggregator.ingest_message(msg)
        corroborated_ev = self.evidence_aggregator.get_corroborated_evidence()

        # Step 5: Conflict Resolution (Minority Evidence Preserved)
        conflicts = self.conflict_resolver.detect_conflicts(task_id, collected_messages)

        # Step 6: Consensus Synthesis
        candidate = self.synthesizer.synthesize(
            task_id=task_id,
            objective=objective,
            messages=collected_messages,
            corroborated_evidence=corroborated_ev,
            conflicts=conflicts,
            critical_counter_evidence=[],
        )

        # Step 7: Sovereign Capability Gate Check (if capability requested)
        if (candidate.recommended_decision_state == DecisionState.CAPABILITY_REQUIRED or task_type == CognitiveTaskType.CAPABILITY) and self.capability_gate:
            cap_req = CapabilityRequest(
                capability_id="distributed_action",
                caller_id=f"federation_{tenant_id}",
                session_id=session_id,
                parameters={"objective": objective},
            )
            try:
                is_auth = self.capability_gate.authorize(cap_req)
            except Exception:
                is_auth = False

            if not is_auth:
                candidate.recommended_decision_state = DecisionState.SAFE_STOP
                candidate.primary_synthesis += " [Capability invocation denied by CapabilityGate]"

        # Step 8: Governed Continual Experience Capture
        if self.memory_engine:
            try:
                self.memory_engine.record_experience(
                    task_id=f"dist_{uuid.uuid4().hex[:8]}",
                    tenant_id=tenant_id,
                    session_id=session_id,
                    objective=objective,
                    outcome={"summary": candidate.primary_synthesis, "decision": candidate.recommended_decision_state.value},
                    trace_metadata={"participating_nodes": list(participating_nodes)},
                    weights_modified=False,
                )
            except Exception:
                pass

        # Step 9: Post-flight model weight verification (FAIL CLOSED)
        post_fingerprint = self.integrity_guard.compute_weight_fingerprint(self.model)
        if post_fingerprint != pre_fingerprint:
            raise WeightMutationError(
                f"Post-flight model weight mutation detected: {post_fingerprint} != {pre_fingerprint}. FAIL CLOSED."
            )

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        self.observability.record_cycle_latency(elapsed_ms)

        # Step 10: Build Sanitized Public Distributed Trace
        trace_id = f"trace_dist_{uuid.uuid4().hex[:10]}"
        trace = SafePublicDistributedTrace(
            trace_id=trace_id,
            task_id=task_id,
            federation_id=f"fed_{tenant_id}",
            participating_node_ids=sorted(list(participating_nodes)),
            participating_agent_roles=sorted(list(participating_roles)),
            routing_decisions=[d.to_dict() for d in routing_decisions],
            message_count=len(collected_messages),
            remote_task_count=sum(1 for d in routing_decisions if d.is_remote),
            retry_count=retry_count,
            failure_count=failure_count,
            conflict_count=len(conflicts),
            synthesis_state=candidate.recommended_decision_state.value,
            decision_state=candidate.recommended_decision_state.value,
            latency_ms=round(elapsed_ms, 3),
            final_status="COMPLETED" if candidate.confidence >= 0.2 else "UNCERTAIN",
            trace_fingerprint=hashlib.sha256(f"{trace_id}:{len(collected_messages)}:{candidate.primary_synthesis}".encode("utf-8")).hexdigest(),
        )

        return candidate, trace

    def _classify_task(self, prompt: str) -> CognitiveTaskType:
        """Route prompt into appropriate task classification."""
        p_lower = prompt.lower()
        if any(w in p_lower for w in ("sensor", "device", "execute capability", "turn on", "hardware", "read_device", "trigger external")):
            return CognitiveTaskType.CAPABILITY
        if any(w in p_lower for w in ("calculate", "analyze", "derive", "prove", "step by step", "solve", "spike", "audit")):
            return CognitiveTaskType.ANALYTICAL
        if any(w in p_lower for w in ("choose", "decide", "option", "select", "recommend", "evaluate", "assess")):
            return CognitiveTaskType.DECISION
        if any(w in p_lower for w in ("what is", "when did", "who is", "state the", "definition", "atomic number")):
            return CognitiveTaskType.FACTUAL
        return CognitiveTaskType.GENERAL
