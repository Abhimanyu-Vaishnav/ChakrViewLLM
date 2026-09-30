"""
Federated Cognitive Engine — Step 42 top-level orchestrator.

Bridges cognitive problem-solving (Steps 15, 19, 28) with the production
federated distributed runtime (Steps 36–41).

Responsibilities:
1. Plan a CognitiveTaskGraph (DAG) for a given objective.
2. Advance the DAG by dispatching ready steps as WorkUnits or local simulations.
3. Record results, propagate context updates, and detect failures.
4. Synthesize multi-node outputs with anti-majority minority preservation.
5. Propose BFT consensus finalization via Step 40 ConsensusEngine (if available).
6. Expose CognitiveEpisode status and history.

AXIOMS:
- ADVERTISEMENT != PERMISSION: Dispatching cognitive steps does not bypass CapabilityGate.
- LOCAL_POLICY > CONSENSUS_DECISION: Episode consensus confirms order; it never overrides
  local sovereign gate decisions.
- ZERO NEURAL WEIGHT MUTATION: ΔW = 0 is verified by FederatedNeuralCapability.
- DATA != AUTHORITY: Episode conclusions are data; they cannot authorize capabilities.
"""

import logging
import threading
import uuid
from typing import Any, Dict, List, Optional

from chakrview.cognition.federation.cognitive.errors import (
    FederatedCognitiveError,
    CognitiveEpisodeStateError,
)
from chakrview.cognition.federation.cognitive.models import (
    CAPABILITY_ANALYST,
    CAPABILITY_CRITIC,
    CAPABILITY_RESEARCHER,
    CAPABILITY_SYNTHESIZER,
    CAPABILITY_VERIFIER,
    CognitiveContextEnvelope,
    CognitiveEpisode,
    CognitiveEpisodeState,
    CognitiveRole,
    CognitiveStep,
    CognitiveTaskGraph,
)
from chakrview.cognition.federation.cognitive.bridge import FederatedReasoningBridge
from chakrview.cognition.federation.cognitive.episode import CognitiveEpisodeManager
from chakrview.cognition.federation.cognitive.synthesis import CognitiveSynthesisEngine
from chakrview.capability.contract import CapabilityRequest
from chakrview.cognition.federation.cognitive.memory import PersistentCognitiveMemoryAdapter
from chakrview.cognition.federation.cognitive.capabilities import GovernedKnowledgeRetrievalCapability
from chakrview.cognition.orchestration.planner import AdaptiveTaskPlanner
from chakrview.cognition.orchestration.classifier import DeterministicWorkloadClassifier
from chakrview.cognition.orchestration.models import WorkloadClass
from chakrview.cognition.federated.models import AgentRole

logger = logging.getLogger("chakrview.federation.cognitive.engine")

ROLE_TO_CAPABILITY: Dict[AgentRole, str] = {
    AgentRole.ANALYST: CAPABILITY_ANALYST,
    AgentRole.RESEARCHER: CAPABILITY_RESEARCHER,
    AgentRole.CRITIC: CAPABILITY_CRITIC,
    AgentRole.SYNTHESIZER: CAPABILITY_SYNTHESIZER,
    AgentRole.VERIFIER: CAPABILITY_VERIFIER,
    AgentRole.PLANNER: CAPABILITY_ANALYST,
}


class FederatedCognitiveEngine:
    """
    Top-level orchestrator bridging cognitive problem-solving with federated
    distributed execution.

    Usage (simulation mode — no live federation needed):
        engine = FederatedCognitiveEngine()
        episode = engine.plan_episode("Analyse X", tenant_id="t1", session_id="s1")
        episode = engine.execute_episode(episode.episode_id)
        assert episode.state == CognitiveEpisodeState.COMMITTED

    Usage (live federation mode):
        engine = FederatedCognitiveEngine(federated_node=node)
        episode = engine.plan_episode("Analyse X", tenant_id="t1", session_id="s1")
        episode = engine.execute_episode(episode.episode_id)
    """

    # Standard pipeline step IDs
    STEP_ANALYST = "step_analyst"
    STEP_RESEARCHER = "step_researcher"
    STEP_CRITIC = "step_critic"
    STEP_SYNTHESIZER = "step_synthesizer"
    STEP_VERIFIER = "step_verifier"

    def __init__(
        self,
        federated_node: Optional[Any] = None,
        memory_adapter: Optional[PersistentCognitiveMemoryAdapter] = None,
        knowledge_capability: Optional[GovernedKnowledgeRetrievalCapability] = None,
    ) -> None:
        """
        Args:
            federated_node: Optional FederatedNode (Step 41). When provided,
                cognitive steps are dispatched through live federation.
                When None, the engine operates in local simulation mode.
            memory_adapter: Optional PersistentCognitiveMemoryAdapter (Step 43).
                When provided, queries memory before planning and consolidates
                committed episodes into persistent memory.
            knowledge_capability: Optional GovernedKnowledgeRetrievalCapability (Step 43).
                When provided, handles the RESEARCHER role with real BM25 retrieval.
        """
        self.federated_node = federated_node
        self.memory_adapter = memory_adapter
        self.knowledge_capability = knowledge_capability
        self.adaptive_planner = AdaptiveTaskPlanner()
        self.workload_classifier = DeterministicWorkloadClassifier()
        self.episode_manager = CognitiveEpisodeManager()
        self.bridge = FederatedReasoningBridge()
        self.synthesis_engine = CognitiveSynthesisEngine()
        self._lock = threading.RLock()

    # ─────────────────────────────────────────────────────────────────────────
    # Planning
    # ─────────────────────────────────────────────────────────────────────────

    def plan_episode(
        self,
        objective: str,
        tenant_id: str,
        session_id: str,
        initial_context: Optional[List[str]] = None,
        workload_class: Optional[WorkloadClass] = None,
        use_adaptive_planner: bool = False,
    ) -> CognitiveEpisode:
        """
        Plan a cognitive episode for the given objective.

        If use_adaptive_planner is True (or workload_class is supplied), uses
        AdaptiveTaskPlanner to build a dynamic CognitiveTaskGraph.
        Otherwise, builds the standard 5-step pipeline.

        Pre-populates CognitiveContextEnvelope from persistent memory if
        PersistentCognitiveMemoryAdapter is configured.

        Returns:
            CognitiveEpisode in PLANNING state.
        """
        # 1. Create episode
        episode = self.episode_manager.create_episode(
            objective=objective,
            tenant_id=tenant_id,
            session_id=session_id,
        )

        # 2. Build the cognitive DAG (adaptive or standard)
        if use_adaptive_planner or workload_class is not None:
            if workload_class is not None:
                effective_wc = workload_class
            else:
                effective_wc, _ = self.workload_classifier.classify(objective)
            graph = self._build_adaptive_graph(
                episode_id=episode.episode_id,
                objective=objective,
                workload_class=effective_wc,
            )
        else:
            graph = self._build_standard_graph(
                episode_id=episode.episode_id,
                objective=objective,
            )

        # 3. Retrieve prior context from persistent memory if configured
        combined_context: List[str] = []
        if self.memory_adapter is not None:
            try:
                retrieved_mems = self.memory_adapter.retrieve_context(
                    objective=objective,
                    tenant_id=tenant_id,
                    session_id=session_id,
                    top_k=5,
                )
                combined_context.extend(retrieved_mems)
            except Exception as exc:
                logger.warning("Memory context retrieval failed: %s", exc)

        if initial_context:
            combined_context.extend(initial_context)

        # 4. Build initial context envelope
        context = CognitiveContextEnvelope(
            envelope_id=f"env_{uuid.uuid4().hex[:10]}",
            episode_id=episode.episode_id,
            tenant_id=tenant_id,
            session_id=session_id,
        )
        for item in combined_context[: CognitiveContextEnvelope.MAX_CONTEXT_ITEMS]:
            context.add_context_item(item)
        context.validate()

        # 5. Attach graph to episode: UNINITIALIZED -> PLANNING
        self.episode_manager.attach_graph(episode.episode_id, graph, context)

        logger.info(
            "Planned episode '%s' with %d steps for tenant '%s'",
            episode.episode_id,
            len(graph.steps),
            tenant_id,
        )
        return episode

    def plan_adaptive_episode(
        self,
        objective: str,
        tenant_id: str,
        session_id: str,
        initial_context: Optional[List[str]] = None,
        workload_class: Optional[WorkloadClass] = None,
    ) -> CognitiveEpisode:
        """
        Plan an adaptive cognitive episode using AdaptiveTaskPlanner to dynamically
        tailor the CognitiveTaskGraph to the objective's WorkloadClass.
        """
        return self.plan_episode(
            objective=objective,
            tenant_id=tenant_id,
            session_id=session_id,
            initial_context=initial_context,
            workload_class=workload_class,
            use_adaptive_planner=True,
        )

    def _build_adaptive_graph(
        self,
        episode_id: str,
        objective: str,
        workload_class: WorkloadClass,
    ) -> CognitiveTaskGraph:
        """
        Build a dynamic CognitiveTaskGraph using AdaptiveTaskPlanner tailored
        to the WorkloadClass.
        """
        task_plan = self.adaptive_planner.plan_task(
            task_id=episode_id,
            objective=objective,
            workload_class=workload_class,
        )

        graph = CognitiveTaskGraph(
            graph_id=f"gr_{uuid.uuid4().hex[:10]}",
            episode_id=episode_id,
        )

        role_step_ids: Dict[str, str] = {}
        for role in task_plan.required_roles:
            role_step_ids[role.value] = f"step_{role.value.lower()}"

        for role in task_plan.required_roles:
            step_id = role_step_ids[role.value]
            cap_id = ROLE_TO_CAPABILITY.get(role, CAPABILITY_ANALYST)
            try:
                role_enum = CognitiveRole(role.value.lower())
            except ValueError:
                role_enum = CognitiveRole.ANALYST

            raw_deps = task_plan.role_dependencies.get(role.value, [])
            step_deps = [role_step_ids[dep] for dep in raw_deps if dep in role_step_ids]

            step = CognitiveStep(
                step_id=step_id,
                role=role_enum,
                capability_id=cap_id,
                input_payload={"objective": objective},
                dependencies=step_deps,
            )
            graph.add_step(step)

        graph.validate()
        return graph

    # ─────────────────────────────────────────────────────────────────────────
    # Execution
    # ─────────────────────────────────────────────────────────────────────────

    def execute_episode(
        self,
        episode_id: str,
        simulate_results: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> CognitiveEpisode:
        """
        Execute a planned cognitive episode by advancing its DAG until complete.

        In simulation mode (simulate_results dict provided), uses supplied results
        directly — no federation required. In live mode, dispatches WorkUnits.

        Args:
            episode_id:       ID of the planned episode to execute.
            simulate_results: Optional dict of {step_id: result_dict}. When a step's
                              ID is present in this dict, the dict value is used as
                              the step result instead of dispatching to federation.

        Returns:
            CognitiveEpisode in a terminal state.
        """
        episode = self.episode_manager.get_episode(episode_id)

        # Transition PLANNING -> EXECUTING
        if episode.state == CognitiveEpisodeState.PLANNING:
            self.episode_manager.begin_execution(episode_id)

        graph = episode.task_graph
        if graph is None:
            self.episode_manager.mark_failed(episode_id, "No task graph attached")
            return episode

        # Drive the DAG until complete or stuck
        max_iterations = len(graph.steps) * 3
        iteration = 0
        while not graph.is_complete() and iteration < max_iterations:
            iteration += 1
            ready = graph.get_ready_steps()

            if not ready:
                if graph.has_failures():
                    self.episode_manager.mark_failed(
                        episode_id,
                        "DAG blocked: one or more steps failed with no remaining path",
                    )
                    return episode
                if not graph.is_complete():
                    break
                break

            for step in ready:
                result = self._execute_step(episode, step, simulate_results)
                self.episode_manager.record_step_result(episode_id, step.step_id, result)

        # Check outcome
        if graph.has_failures():
            self.episode_manager.mark_failed(
                episode_id,
                f"Graph '{graph.graph_id}' completed with step failures",
            )
            return episode

        # Synthesize
        if graph.is_complete():
            all_results = list(graph.completed_results().values())
            synthesis = self.episode_manager.synthesize_episode(
                episode_id=episode_id,
                step_id="final_synthesis",
                step_results=all_results,
            )

            # Propose BFT consensus finalization
            proposal_id = self._propose_consensus_finalization(episode, synthesis)
            self.episode_manager.mark_finalizing(
                episode_id, proposal_id or "local_only"
            )
            self.episode_manager.mark_committed(episode_id)

            # Consolidate into persistent memory if adapter configured
            if self.memory_adapter is not None:
                try:
                    self.memory_adapter.consolidate_episode(episode)
                except Exception as exc:
                    logger.warning("Episode memory consolidation error: %s", exc)

        return episode

    # ─────────────────────────────────────────────────────────────────────────
    # Status
    # ─────────────────────────────────────────────────────────────────────────

    def get_episode_status(self, episode_id: str) -> Dict[str, Any]:
        """Return a status snapshot for a cognitive episode."""
        episode = self.episode_manager.get_episode(episode_id)
        graph = episode.task_graph
        return {
            "episode_id": episode.episode_id,
            "state": episode.state.value,
            "objective": episode.objective,
            "tenant_id": episode.tenant_id,
            "session_id": episode.session_id,
            "graph_complete": graph.is_complete() if graph else None,
            "graph_failures": graph.has_failures() if graph else None,
            "steps_total": len(graph.steps) if graph else 0,
            "steps_completed": (
                sum(1 for s in graph.steps.values()
                    if s.state.value == "completed")
                if graph else 0
            ),
            "synthesis_result": (
                episode.synthesis_result.to_dict()
                if episode.synthesis_result else None
            ),
            "consensus_proposal_id": episode.consensus_proposal_id,
            "error": episode.error,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Private Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _build_standard_graph(
        self, episode_id: str, objective: str
    ) -> CognitiveTaskGraph:
        """Build the standard 5-step cognitive reasoning pipeline DAG."""
        graph = CognitiveTaskGraph(
            graph_id=f"gr_{uuid.uuid4().hex[:10]}",
            episode_id=episode_id,
        )
        steps = [
            CognitiveStep(
                step_id=self.STEP_ANALYST,
                role=CognitiveRole.ANALYST,
                capability_id=CAPABILITY_ANALYST,
                input_payload={"objective": objective},
                dependencies=[],
            ),
            CognitiveStep(
                step_id=self.STEP_RESEARCHER,
                role=CognitiveRole.RESEARCHER,
                capability_id=CAPABILITY_RESEARCHER,
                input_payload={"objective": objective},
                dependencies=[self.STEP_ANALYST],
            ),
            CognitiveStep(
                step_id=self.STEP_CRITIC,
                role=CognitiveRole.CRITIC,
                capability_id=CAPABILITY_CRITIC,
                input_payload={"objective": objective},
                dependencies=[self.STEP_ANALYST, self.STEP_RESEARCHER],
            ),
            CognitiveStep(
                step_id=self.STEP_SYNTHESIZER,
                role=CognitiveRole.SYNTHESIZER,
                capability_id=CAPABILITY_SYNTHESIZER,
                input_payload={"objective": objective},
                dependencies=[
                    self.STEP_ANALYST,
                    self.STEP_RESEARCHER,
                    self.STEP_CRITIC,
                ],
            ),
            CognitiveStep(
                step_id=self.STEP_VERIFIER,
                role=CognitiveRole.VERIFIER,
                capability_id=CAPABILITY_VERIFIER,
                input_payload={"objective": objective},
                dependencies=[self.STEP_SYNTHESIZER],
            ),
        ]
        for step in steps:
            graph.add_step(step)
        graph.validate()
        return graph

    def _execute_step(
        self,
        episode: CognitiveEpisode,
        step: CognitiveStep,
        simulate_results: Optional[Dict[str, Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Execute a single step via simulation, federation, or local fallback."""
        step_id = step.step_id

        # Use caller-supplied simulation result if provided
        if simulate_results and step_id in simulate_results:
            logger.debug("Simulation result used for step '%s'", step_id)
            return simulate_results[step_id]

        # Live federation dispatch
        if self.federated_node is not None:
            return self._dispatch_via_federation(episode, step)

        # Check if local GovernedKnowledgeRetrievalCapability can handle RESEARCHER role
        if step.role == CognitiveRole.RESEARCHER and self.knowledge_capability is not None:
            req = CapabilityRequest(
                capability_id=self.knowledge_capability.descriptor.capability_id,
                parameters={
                    "objective": step.input_payload.get("objective", ""),
                    "query": step.input_payload.get("objective", ""),
                    "step_id": step_id,
                },
            )
            cap_result = self.knowledge_capability.execute(req)
            if cap_result.success:
                return {
                    "step_id": step_id,
                    "success": True,
                    "node_id": "local",
                    "conclusion": cap_result.output.get("conclusion", ""),
                    "evidence": cap_result.output.get("evidence", []),
                    "evidence_records": cap_result.output.get("evidence_records", []),
                    "hypotheses": cap_result.output.get("hypotheses", []),
                }

        # Local fallback: minimal governed response
        return {
            "step_id": step_id,
            "success": True,
            "node_id": "local",
            "conclusion": (
                f"[{step.role.value.upper()}] Local execution of '{step_id}' "
                f"for: {step.input_payload.get('objective', '')[:80]}"
            ),
            "evidence": [f"Local evidence from {step.role.value} step '{step_id}'"],
            "hypotheses": [
                {
                    "claim": f"{step.role.value} hypothesis for '{step_id}'",
                    "confidence": 0.8,
                    "step_id": step_id,
                }
            ],
        }

    def _dispatch_via_federation(
        self,
        episode: CognitiveEpisode,
        step: CognitiveStep,
    ) -> Dict[str, Any]:
        """Dispatch a cognitive step as a WorkUnit through Step 38 coordinator."""
        if self.federated_node is None:
            raise FederatedCognitiveError("No federated node available for dispatch")

        try:
            coord = self.federated_node.engine.task_coordinator
            task = coord.create_task(
                name=f"cog_{step.step_id}_{episode.episode_id[:8]}",
                tenant_id=episode.tenant_id,
            )
            unit_spec = {
                "capability_id": step.capability_id,
                "input_payload": {
                    "step_id": step.step_id,
                    "role": step.role.value,
                    "objective": step.input_payload.get("objective", ""),
                    "input_payload": step.input_payload,
                    "context_envelope": (
                        episode.context.to_dict() if episode.context else {}
                    ),
                },
                "min_cpu_cores": 1.0,
                "min_memory_mb": 256,
            }
            coord.decompose_task(task.task_id, [unit_spec])
            coord.schedule_and_dispatch_task(task.task_id)

            completed = coord.get_task(task.task_id)
            if completed and completed.final_result:
                return {
                    "step_id": step.step_id,
                    "success": True,
                    "node_id": self.federated_node.config.node_id,
                    "conclusion": str(
                        completed.final_result.get("result", "Federation dispatch complete")
                    ),
                    "evidence": [],
                    "hypotheses": [],
                    "raw_output": completed.final_result,
                }
            return {
                "step_id": step.step_id,
                "success": True,
                "node_id": self.federated_node.config.node_id,
                "conclusion": f"Federated step '{step.step_id}' dispatched and accepted",
                "evidence": [],
                "hypotheses": [],
            }
        except Exception as exc:
            logger.error("Federation dispatch failed for step '%s': %s", step.step_id, exc)
            return {
                "step_id": step.step_id,
                "success": False,
                "node_id": "unknown",
                "conclusion": "",
                "evidence": [],
                "hypotheses": [],
                "error": str(exc),
            }

    def _propose_consensus_finalization(
        self,
        episode: CognitiveEpisode,
        synthesis: Any,
    ) -> Optional[str]:
        """Propose BFT consensus finalization for a completed episode if available."""
        if self.federated_node is None:
            return None
        try:
            from chakrview.cognition.federation.consensus.models import (
                ConsensusTransitionType,
            )
            proposal = self.federated_node.propose_consensus(
                transition_type=ConsensusTransitionType.TASK_COMMIT_FINALIZATION,
                payload={
                    "episode_id": episode.episode_id,
                    "synthesis_id": synthesis.synthesis_id,
                    "conclusion": synthesis.synthesized_conclusion,
                    "conflict_count": len(synthesis.conflict_records),
                    "participating_nodes": synthesis.participating_nodes,
                },
            )
            return proposal.proposal_id
        except Exception as exc:
            logger.warning(
                "Consensus finalization failed for episode '%s': %s",
                episode.episode_id,
                exc,
            )
            return None
