"""
Cognitive Episode Manager for Step 42.

Manages the full lifecycle of CognitiveEpisode instances:
1. Creating episodes and tracking them by ID.
2. Attaching planned CognitiveTaskGraphs and initial context envelopes.
3. Recording individual step results and propagating context updates.
4. Synthesizing multi-node outputs via CognitiveSynthesisEngine.
5. Managing terminal state transitions (COMMITTED, FAILED, REJECTED).

AXIOM: LOCAL_POLICY > CONSENSUS_DECISION
  Episode finalization proposals are submitted to BFT consensus (Step 40),
  but each local node retains sovereign discretion over any resulting
  local state changes (e.g., memory promotion) independent of consensus.
"""

import logging
import threading
import uuid
from typing import Any, Dict, List, Optional

from chakrview.cognition.federation.cognitive.errors import (
    CognitiveEpisodeNotFoundError,
    CognitiveEpisodeStateError,
)
from chakrview.cognition.federation.cognitive.models import (
    CognitiveEpisode,
    CognitiveEpisodeState,
    CognitiveContextEnvelope,
    CognitiveTaskGraph,
    SynthesisResult,
)
from chakrview.cognition.federation.cognitive.synthesis import CognitiveSynthesisEngine

logger = logging.getLogger("chakrview.federation.cognitive.episode")


class CognitiveEpisodeManager:
    """
    Thread-safe manager for all active and historical CognitiveEpisode instances.
    """

    def __init__(self) -> None:
        self._episodes: Dict[str, CognitiveEpisode] = {}
        self._synthesis_engine = CognitiveSynthesisEngine()
        self._lock = threading.RLock()

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle Creation
    # ─────────────────────────────────────────────────────────────────────────

    def create_episode(
        self,
        objective: str,
        tenant_id: str,
        session_id: str,
    ) -> CognitiveEpisode:
        """Create a new CognitiveEpisode in UNINITIALIZED state."""
        episode_id = f"ep_{uuid.uuid4().hex[:12]}"
        episode = CognitiveEpisode(
            episode_id=episode_id,
            tenant_id=tenant_id,
            session_id=session_id,
            objective=objective,
        )
        with self._lock:
            self._episodes[episode_id] = episode
        logger.info("Created episode '%s' for tenant '%s'", episode_id, tenant_id)
        return episode

    def get_episode(self, episode_id: str) -> CognitiveEpisode:
        """Retrieve a tracked episode by ID. Raises CognitiveEpisodeNotFoundError."""
        with self._lock:
            if episode_id not in self._episodes:
                raise CognitiveEpisodeNotFoundError(f"Episode '{episode_id}' not found")
            return self._episodes[episode_id]

    # ─────────────────────────────────────────────────────────────────────────
    # State Transitions
    # ─────────────────────────────────────────────────────────────────────────

    def attach_graph(
        self,
        episode_id: str,
        graph: CognitiveTaskGraph,
        context: CognitiveContextEnvelope,
    ) -> None:
        """Attach a planned CognitiveTaskGraph. Transitions UNINITIALIZED -> PLANNING."""
        episode = self.get_episode(episode_id)
        with self._lock:
            episode.task_graph = graph
            episode.context = context
            episode.transition_to(CognitiveEpisodeState.PLANNING)
        logger.info("Attached graph '%s' to episode '%s'", graph.graph_id, episode_id)

    def begin_execution(self, episode_id: str) -> None:
        """Transition episode PLANNING -> EXECUTING."""
        episode = self.get_episode(episode_id)
        with self._lock:
            episode.transition_to(CognitiveEpisodeState.EXECUTING)

    def record_step_result(
        self,
        episode_id: str,
        step_id: str,
        result: Dict[str, Any],
    ) -> None:
        """
        Record a completed step result and propagate context updates.

        On success, evidence and conclusions from the result are added to
        the episode's CognitiveContextEnvelope for subsequent steps.
        """
        episode = self.get_episode(episode_id)
        with self._lock:
            if episode.task_graph is None:
                raise CognitiveEpisodeStateError(
                    f"Episode '{episode_id}' has no task graph attached"
                )
            success = result.get("success", False)
            if success:
                episode.task_graph.mark_step_completed(step_id, result)
                # Propagate conclusions and evidence into context envelope
                if episode.context is not None:
                    conclusion = result.get("conclusion", "")
                    if conclusion:
                        episode.context.add_context_item(f"[{step_id}] {conclusion}")
                    for ev in result.get("evidence", []):
                        episode.context.add_evidence(ev)
                    node_id = result.get("node_id", "")
                    if node_id:
                        episode.context.add_provenance(node_id)
            else:
                episode.task_graph.mark_step_failed(
                    step_id, result.get("error", "Unknown error")
                )
        logger.debug(
            "Recorded step '%s' in episode '%s' (success=%s)", step_id, episode_id, success
        )

    def synthesize_episode(
        self,
        episode_id: str,
        step_id: str,
        step_results: List[Dict[str, Any]],
    ) -> SynthesisResult:
        """
        Run synthesis over multi-node results. Transitions EXECUTING -> SYNTHESIZING.
        """
        episode = self.get_episode(episode_id)
        with self._lock:
            episode.transition_to(CognitiveEpisodeState.SYNTHESIZING)

        synthesis = self._synthesis_engine.synthesize(
            episode_id=episode_id,
            step_id=step_id,
            step_results=step_results,
        )

        with self._lock:
            episode.synthesis_result = synthesis

        logger.info(
            "Synthesis complete for episode '%s' step '%s': conflicts=%d",
            episode_id,
            step_id,
            len(synthesis.conflict_records),
        )
        return synthesis

    def mark_finalizing(self, episode_id: str, proposal_id: str) -> None:
        """Transition SYNTHESIZING -> FINALIZING with consensus proposal ID."""
        episode = self.get_episode(episode_id)
        with self._lock:
            episode.transition_to(CognitiveEpisodeState.FINALIZING)
            episode.consensus_proposal_id = proposal_id

    def mark_committed(self, episode_id: str) -> None:
        """Transition FINALIZING -> COMMITTED (terminal success)."""
        episode = self.get_episode(episode_id)
        with self._lock:
            episode.transition_to(CognitiveEpisodeState.COMMITTED)
        logger.info("Episode '%s' committed successfully", episode_id)

    def mark_failed(self, episode_id: str, error: str) -> None:
        """Transition episode to FAILED terminal state (fail-closed)."""
        episode = self.get_episode(episode_id)
        with self._lock:
            try:
                episode.transition_to(CognitiveEpisodeState.FAILED)
            except CognitiveEpisodeStateError:
                pass  # Already in a terminal state
            episode.error = error
        logger.error("Episode '%s' failed: %s", episode_id, error)

    # ─────────────────────────────────────────────────────────────────────────
    # Query
    # ─────────────────────────────────────────────────────────────────────────

    def list_episodes(
        self,
        tenant_id: Optional[str] = None,
        state: Optional[CognitiveEpisodeState] = None,
    ) -> List[CognitiveEpisode]:
        """List episodes optionally filtered by tenant_id and/or state."""
        with self._lock:
            episodes = list(self._episodes.values())
        if tenant_id is not None:
            episodes = [ep for ep in episodes if ep.tenant_id == tenant_id]
        if state is not None:
            episodes = [ep for ep in episodes if ep.state == state]
        return episodes

    def episode_count(self) -> int:
        """Return total number of tracked episodes."""
        with self._lock:
            return len(self._episodes)
