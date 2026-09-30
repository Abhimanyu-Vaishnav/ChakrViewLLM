"""
ChakrView Step 58: Canonical Cognitive Workspace.

The primary orchestration layer for multi-turn feedback, external execution in ChakrKshetra,
and multi-tier memory retrieval, consolidation, and reflection.

Target Loop:
UNDERSTAND -> PLAN -> ACT -> OBSERVE -> DIAGNOSE -> CORRECT -> VERIFY -> REMEMBER -> CONSOLIDATE -> REUSE -> REFLECT

Strict separation of concerns:
CHAKRVIEW NEURAL CORE ≠ CHAKRKSHETRA ≠ WORKING MEMORY ≠ EPISODIC MEMORY ≠ SEMANTIC MEMORY ≠ RIL ≠ ADAPTER ≠ EVALUATOR ≠ HOST APPLICATION
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from chakrview.arena.models import ProjectSpecification, ProjectManifest, SourceFile, FileRole, FailureCategory
from chakrview.arena.evaluator import ArenaEvaluator
from chakrview.learning.episode import Attempt, LearningEpisode
from chakrview.learning.experience import ExperienceExtractor, ExperienceRecord
from chakrview.runtime.cortex_context import CognitiveContext
from chakrview.cognition.workspace.state import CognitiveWorkingState, CognitiveStatePhase
from chakrview.cognition.workspace.consolidation import MemoryConsolidator, SemanticMemoryEntry
from chakrview.cognition.workspace.retrieval import ExplainableMemoryRetriever, MemoryRetrievalResult
from chakrview.cognition.workspace.reflection import StructuredReflector, EpisodeReflection


class CognitiveWorkspace:
    """
    Canonical Cognitive Workspace orchestrator for ChakrView.
    Coordinates multi-turn problem solving, sandboxed execution, and memory consolidation.
    """

    def __init__(
        self,
        evaluator: Optional[ArenaEvaluator] = None,
        consolidator: Optional[MemoryConsolidator] = None,
        retriever: Optional[ExplainableMemoryRetriever] = None,
        max_attempts: int = 3,
    ) -> None:
        self.evaluator = evaluator or ArenaEvaluator()
        self.consolidator = consolidator or MemoryConsolidator()
        self.retriever = retriever or ExplainableMemoryRetriever()
        self.max_attempts = max_attempts

        # Current active working state
        self.working_state: Optional[CognitiveWorkingState] = None

        # Auditable execution history
        self.completed_episodes: List[LearningEpisode] = []
        self.reflections: List[EpisodeReflection] = []
        self.trajectory_logs: List[str] = []

    def reset(self) -> None:
        """Clear active working state and turn-level scratchpads."""
        self.working_state = None

    def initialize_task(
        self,
        task_id: str,
        task_spec: Dict[str, Any],
        initial_action: str,
        active_adapter: Optional[str] = None,
    ) -> CognitiveWorkingState:
        """
        Stage 1: UNDERSTAND & Initialize working memory.
        """
        family = task_spec.get("category", "general")
        desc = task_spec.get("description", task_id)

        self.working_state = CognitiveWorkingState(
            task_id=task_id,
            task_family=family,
            objective=desc,
            current_phase=CognitiveStatePhase.INITIALIZING,
            current_action=initial_action,
            max_attempts=self.max_attempts,
            active_adapter=active_adapter,
        )
        return self.working_state

    def plan_task(self) -> List[str]:
        """
        Stage 2: PLAN initial execution strategy.
        """
        if not self.working_state:
            raise RuntimeError("Cannot plan: Workspace not initialized with a task.")

        self.working_state.transition_to(CognitiveStatePhase.PLANNING)
        plan = [
            f"1. Dispatch initial proposal to ChakrKshetra sandbox for {self.working_state.task_id}",
            "2. Capture test assertions and runtime observations",
            "3. If failed: diagnose fault, query explainable memory, synthesize correction",
            "4. Verify clean pass and consolidate verified experience",
        ]
        self.working_state.current_plan = plan
        return plan

    def retrieve_relevant_memories(
        self,
        current_diagnosis: Optional[str] = None,
        top_k: int = 2,
    ) -> List[MemoryRetrievalResult]:
        """
        Stage 10 (Preview / Inter-turn): REUSE relevant memory patterns.
        """
        if not self.working_state:
            return []

        results = self.retriever.retrieve(
            task_family=self.working_state.task_family,
            objective=self.working_state.objective,
            current_diagnosis=current_diagnosis,
            task_id=self.working_state.task_id,
            consolidator=self.consolidator,
            top_k=top_k,
        )

        # Store in working state
        self.working_state.relevant_memories = [r.to_dict() for r in results]
        return results


    def build_cognitive_context(
        self,
        retrieved_results: Optional[List[MemoryRetrievalResult]] = None,
        diagnosis_str: Optional[str] = None,
    ) -> CognitiveContext:
        """
        Construct bounded CognitiveContext (< 800 chars / ~192 tokens) for neural inference.
        """
        if not self.working_state:
            return CognitiveContext()

        mem_summary = ""
        if retrieved_results:
            # Use top memory content bounded in length
            top_mem = retrieved_results[0]
            mem_summary = f"[{top_mem.memory_type}] {top_mem.reusable_content}"[:120]

        ctx = CognitiveContext(
            task_mode="REPAIR" if diagnosis_str else "SOLVE",
            goal=self.working_state.objective[:100],
            state=f"Attempt {self.working_state.attempt_index}/{self.max_attempts}",
            memory_context=mem_summary,
            reasoning_state=(diagnosis_str or "Initial Attempt")[:100],
            constraints="Sandboxed ChakrKshetra Execution",
        )
        ctx.validate()
        return ctx

    def execute_turn(
        self,
        action: str,
        task_spec: Dict[str, Any],
        entrypoint: str,
        test_code: str,
    ) -> Tuple[bool, str, Optional[str], float]:
        """
        Stages 3, 4, 5: ACT -> OBSERVE -> DIAGNOSE
        Executes action in ChakrKshetra sandbox and returns (passed, observation, diagnosis, duration).
        """
        if not self.working_state:
            raise RuntimeError("Cannot execute turn: Workspace not initialized.")

        self.working_state.transition_to(CognitiveStatePhase.EXECUTING)
        self.working_state.current_action = action
        start_t = time.perf_counter()

        source_file = SourceFile(path=entrypoint, content=action, role=FileRole.SOURCE)
        test_file = SourceFile(path="test_solution.py", content=test_code, role=FileRole.TEST)
        spec = ProjectSpecification(
            project_id=f"proj_{self.working_state.task_id}_{self.working_state.attempt_index}",
            project_name=self.working_state.task_id,
            description=self.working_state.objective,
            entrypoint=entrypoint,
        )
        manifest = ProjectManifest(specification=spec, files=[source_file, test_file])

        # OBSERVE in ChakrKshetra
        self.working_state.transition_to(CognitiveStatePhase.OBSERVING)
        exec_res = self.evaluator.evaluate_manifest(manifest)
        duration = time.perf_counter() - start_t

        # EVALUATE & DIAGNOSE
        self.working_state.transition_to(CognitiveStatePhase.EVALUATING)
        passed = (exec_res.failure_category == FailureCategory.SUCCESS and exec_res.test_result.passed > 0)

        observation = exec_res.test_result.stderr or ("PASSED" if passed else "Execution Failed")
        self.working_state.last_observation = observation

        diagnosis = None
        if not passed:
            self.working_state.transition_to(CognitiveStatePhase.DIAGNOSING)
            err_msg = exec_res.syntax_error_message or exec_res.test_result.stderr
            diagnosis = f"Error in execution: {err_msg.strip()}"[:100]
            self.working_state.record_failure(
                action=action,
                observation=observation,
                diagnosis=diagnosis,
                duration=duration,
            )

        return passed, observation, diagnosis, duration

    def run_task(
        self,
        task_id: str,
        task_spec: Dict[str, Any],
        initial_action: str,
        neural_proposer: Optional[Callable[[str, CognitiveContext], str]] = None,
        active_adapter: Optional[str] = None,
        enable_memory_retrieval: bool = True,
        enable_consolidation: bool = True,
    ) -> LearningEpisode:
        """
        Execute full 11-stage cognitive loop across multiple turns / attempts.
        """
        # 1. UNDERSTAND
        state = self.initialize_task(task_id, task_spec, initial_action, active_adapter=active_adapter)
        
        # 2. PLAN
        self.plan_task()

        episode_id = f"ep_{task_id}_{int(time.time() * 1000)}"
        episode = LearningEpisode(
            episode_id=episode_id,
            task_id=task_id,
            task_spec=task_spec,
            initial_context={"adapter": active_adapter},
        )
        episode.add_stage("UNDERSTAND")
        episode.add_stage("PLAN")

        test_code = task_spec.get("test_code", "")
        entrypoint = task_spec.get("entrypoint", "solution.py")

        current_action = initial_action

        # Initial memory check before attempt 1
        initial_memories = []
        if enable_memory_retrieval:
            initial_memories = self.retrieve_relevant_memories()
            if initial_memories:
                episode.initial_context["retrieved_memories"] = [m.to_dict() for m in initial_memories]
                # If we have a direct reusable pattern from memory, check if it suggests an action
                top_m = initial_memories[0]
                if top_m.score >= 0.70:
                    verified_act = top_m.raw_item.get("what_worked")
                    if verified_act and verified_act != "Did not converge to verified solution":
                        current_action = verified_act
                    elif "Successful Action:" in top_m.reusable_content:
                        extracted_act = top_m.reusable_content.split("Successful Action:", 1)[1].strip()
                        if extracted_act:
                            current_action = extracted_act
                    elif "Solution:" in top_m.reusable_content:
                        extracted_act = top_m.reusable_content.split("Solution:", 1)[1].strip()
                        if extracted_act and "def " in extracted_act:
                            current_action = extracted_act


        for attempt_idx in range(1, self.max_attempts + 1):
            state.attempt_index = attempt_idx
            episode.add_stage("ACT")

            # 3, 4, 5: ACT -> OBSERVE -> EVALUATE / DIAGNOSE
            passed, obs, diag, duration = self.execute_turn(
                action=current_action,
                task_spec=task_spec,
                entrypoint=entrypoint,
                test_code=test_code,
            )
            episode.add_stage("OBSERVE")
            episode.add_stage("EVALUATE")

            correction_str = None
            if not passed:
                episode.add_stage("DIAGNOSE")
                # 6. CORRECT
                if attempt_idx < self.max_attempts:
                    episode.add_stage("CORRECT")
                    state.transition_to(CognitiveStatePhase.REPAIRING)

                    # Retrieve targeted memories based on diagnosis
                    targeted_mems = []
                    if enable_memory_retrieval:
                        targeted_mems = self.retrieve_relevant_memories(current_diagnosis=diag)

                    # Build context
                    cortex_ctx = self.build_cognitive_context(
                        retrieved_results=targeted_mems,
                        diagnosis_str=diag,
                    )

                    # Correction synthesis
                    if task_spec.get("correction_rule"):
                        correction_str = task_spec["correction_rule"](current_action, diag)
                    elif neural_proposer:
                        correction_str = neural_proposer(current_action, cortex_ctx)

            # Record attempt
            attempt = Attempt(
                attempt_id=attempt_idx,
                state="EXECUTED",
                action=current_action,
                observation=obs,
                evaluation="PASS" if passed else "FAIL",
                diagnosis=diag,
                correction=correction_str,
                duration_seconds=duration,
            )
            episode.add_attempt(attempt)

            # Format and append trajectory log (Step 54 format)
            traj_entry = (
                f"<TRAJECTORY>\n"
                f"<SPEC>\n{task_id}: {state.objective}\n</SPEC>\n"
                f"<STATE>\nAttempt {attempt_idx} / {self.max_attempts}\n</STATE>\n"
                f"<ACTION>\n{current_action.strip()}\n</ACTION>\n"
                f"<OBSERVATION>\n{obs.strip()}\n</OBSERVATION>\n"
                f"<DIAGNOSIS>\n{diag or 'None'}\n</DIAGNOSIS>\n"
                f"<NEXT_ACTION>\n{correction_str.strip() if correction_str else 'None'}\n</NEXT_ACTION>\n"
                f"<RESULT>\n{'SUCCESS' if passed else 'FAILURE'}\n</RESULT>\n"
                f"</TRAJECTORY>"
            )
            self.trajectory_logs.append(traj_entry)

            # 7. VERIFY
            if passed:
                episode.add_stage("VERIFY")
                state.transition_to(CognitiveStatePhase.VERIFYING)
                state.verified = True
                state.completed = True
                episode.final_result = "SUCCESS"
                break
            elif correction_str:
                current_action = correction_str
            else:
                break

        if not state.verified:
            state.transition_to(CognitiveStatePhase.FAILED)
            state.completed = True
            episode.final_result = "FAILURE"

        # 8, 9, 10: REMEMBER -> CONSOLIDATE -> REUSE
        if episode.final_result == "SUCCESS":
            episode.add_stage("REMEMBER")
            exp = ExperienceExtractor.extract_from_episode(episode)
            if exp and exp.verified:
                episode.experience_record = exp.to_dict()
                # Admit into consolidator
                admitted = self.consolidator.admit_experience(exp)
                if admitted and enable_consolidation:
                    episode.add_stage("CONSOLIDATE")
                    self.consolidator.consolidate()
                episode.learning_status = "VERIFIED_LEARNING"

        # 11. REFLECT
        episode.add_stage("REFLECT")
        reflection = StructuredReflector.reflect(episode)
        self.reflections.append(reflection)
        self.completed_episodes.append(episode)

        return episode
