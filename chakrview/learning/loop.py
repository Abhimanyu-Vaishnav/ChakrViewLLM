"""
ChakrView Step 57: Cognitive Learning Loop Orchestrator.

Implements the executable loop:
UNDERSTAND -> PLAN -> ACT -> OBSERVE -> EVALUATE -> DIAGNOSE -> CORRECT -> RETRY -> VERIFY -> REMEMBER -> LEARN

Decoupled components:
- ChakrMicro neural core (generates actions)
- ChakrKshetra (executes and isolates code)
- Evaluator (objective syntax/test pass/fail)
- EpisodicMemoryStore (persists validated experience)
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from chakrview.arena.models import ProjectSpecification, ProjectManifest, SourceFile, FileRole, FailureCategory
from chakrview.arena.evaluator import ArenaEvaluator
from chakrview.learning.episode import Attempt, LearningEpisode
from chakrview.learning.experience import ExperienceExtractor, ExperienceRecord
from chakrview.runtime.cortex_context import CognitiveContext


class CognitiveLearningLoop:
    """
    Orchestrates problem-solving episodes with failure diagnosis and experiential memory.
    """

    def __init__(
        self,
        evaluator: Optional[ArenaEvaluator] = None,
        max_attempts: int = 3,
    ) -> None:
        self.evaluator = evaluator or ArenaEvaluator()
        self.max_attempts = max_attempts
        self.memory_store: Dict[str, ExperienceRecord] = {}

    def store_experience(self, record: ExperienceRecord) -> None:
        """Store verified experience record indexed by task family and task id."""
        if record.verified:
            self.memory_store[record.task_id] = record
            # Also index by family
            self.memory_store[f"family:{record.task_family}"] = record

    def retrieve_experience(self, task_id: str, task_family: Optional[str] = None) -> Optional[ExperienceRecord]:
        """Retrieve matching experience by task ID or task family."""
        if task_id in self.memory_store:
            return self.memory_store[task_id]
        if task_family and f"family:{task_family}" in self.memory_store:
            return self.memory_store[f"family:{task_family}"]
        return None

    def execute_episode(
        self,
        task_id: str,
        task_spec: Dict[str, Any],
        initial_action: str,
        neural_proposer: Optional[Callable[[str, Optional[CognitiveContext]], str]] = None,
        retrieved_experience: Optional[ExperienceRecord] = None,
    ) -> LearningEpisode:
        """
        Execute an end-to-end learning episode:
        1. UNDERSTAND
        2. PLAN
        3. ACT
        4. OBSERVE (via ArenaEvaluator / ChakrKshetra)
        5. EVALUATE
        6. DIAGNOSE / CORRECT / RETRY if failed
        7. VERIFY
        8. REMEMBER & LEARN
        """
        episode_id = f"ep_{task_id}_{int(time.time() * 1000)}"
        episode = LearningEpisode(
            episode_id=episode_id,
            task_id=task_id,
            task_spec=task_spec,
            initial_context={
                "retrieved_experience": retrieved_experience.to_dict() if retrieved_experience else None,
            },
        )

        episode.add_stage("UNDERSTAND")
        episode.add_stage("PLAN")

        current_action = initial_action
        test_code = task_spec.get("test_code", "")
        entrypoint = task_spec.get("entrypoint", "solution.py")

        for attempt_idx in range(1, self.max_attempts + 1):
            episode.add_stage("ACT")
            start_t = time.perf_counter()

            # Build synthetic manifest for evaluation
            source_file = SourceFile(path=entrypoint, content=current_action, role=FileRole.SOURCE)
            test_file = SourceFile(path="test_solution.py", content=test_code, role=FileRole.TEST)
            spec = ProjectSpecification(
                project_id=f"proj_{task_id}_{attempt_idx}",
                project_name=task_id,
                description=task_spec.get("description", ""),
                entrypoint=entrypoint,
            )
            manifest = ProjectManifest(specification=spec, files=[source_file, test_file])

            # OBSERVE in ChakrKshetra
            episode.add_stage("OBSERVE")
            exec_res = self.evaluator.evaluate_manifest(manifest)
            duration = time.perf_counter() - start_t

            # EVALUATE
            episode.add_stage("EVALUATE")
            passed = (exec_res.failure_category == FailureCategory.SUCCESS and exec_res.test_result.passed > 0)
            eval_str = "PASS" if passed else "FAIL"

            diagnosis_str = None
            correction_str = None

            if not passed:
                episode.add_stage("DIAGNOSE")
                # Extract diagnosis from stderr or syntax error message
                err_msg = exec_res.syntax_error_message or exec_res.test_result.stderr
                diagnosis_str = f"Error in execution: {err_msg.strip()}"[:100]

                # If neural proposer or rule-based correction available
                if attempt_idx < self.max_attempts:
                    episode.add_stage("CORRECT")
                    episode.add_stage("RETRY")
                    # Synthesize correction
                    if task_spec.get("correction_rule"):
                        correction_str = task_spec["correction_rule"](current_action, diagnosis_str)
                    elif neural_proposer:
                        ctx = CognitiveContext(
                            task_mode="REPAIR",
                            goal=task_spec.get("description", ""),
                            memory_context=retrieved_experience.reusable_pattern if retrieved_experience else "",
                            reasoning_state=diagnosis_str,
                        )
                        correction_str = neural_proposer(current_action, ctx)

            attempt = Attempt(
                attempt_id=attempt_idx,
                state="EXECUTED",
                action=current_action,
                observation=exec_res.test_result.stderr or "PASSED",
                evaluation=eval_str,
                diagnosis=diagnosis_str,
                correction=correction_str,
                duration_seconds=duration,
            )
            episode.add_attempt(attempt)

            if passed:
                episode.add_stage("VERIFY")
                episode.final_result = "SUCCESS"
                break
            elif correction_str:
                current_action = correction_str
            else:
                break

        # REMEMBER & LEARN
        if episode.final_result == "SUCCESS":
            episode.add_stage("REMEMBER")
            episode.add_stage("LEARN")
            exp = ExperienceExtractor.extract_from_episode(episode)
            if exp and exp.verified:
                episode.experience_record = exp.to_dict()
                self.store_experience(exp)
                episode.learning_status = "NEW_LEARNING" if not retrieved_experience else "REPLAY_REUSE"

        return episode
