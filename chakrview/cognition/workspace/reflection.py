"""
ChakrView Step 58: Structured Deterministic Reflection Layer.

Defines:
- EpisodeReflection: Strictly typed dataclass containing the post-episode reflection.
- StructuredReflector: Deterministic analyzer answering:
  - What happened? (Trajectory summary)
  - What failed? (Failed actions & causes)
  - What worked? (Verified resolution)
  - Why did it work? (Structural rationale)
  - What can transfer? (Task-family pattern)
  - What should NOT be generalized? (Boundary conditions)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import time
from typing import Any, Dict, List, Optional

from chakrview.learning.episode import LearningEpisode


@dataclass
class EpisodeReflection:
    """
    Structured, deterministic reflection artifact for a completed episode.
    """
    episode_id: str
    task_id: str
    task_family: str
    what_happened: str
    what_failed: List[str]
    what_worked: str
    why_it_worked: str
    what_can_transfer: str
    boundary_limits: List[str]
    transfer_confidence: float = 0.0
    timestamp_utc: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StructuredReflector:
    """
    Lightweight deterministic reflection analyzer.
    Extracts structured cognitive insights from an executed LearningEpisode.
    """

    @staticmethod
    def reflect(episode: LearningEpisode) -> EpisodeReflection:
        """Analyze episode attempts and synthesize deterministic reflection."""
        task_id = episode.task_id
        spec = episode.task_spec
        family = spec.get("category", "general")

        total_attempts = len(episode.attempts)
        is_success = episode.final_result == "SUCCESS"

        # What happened summary
        summary = (
            f"Episode '{episode.episode_id}' for task '{task_id}' executed {total_attempts} attempts "
            f"reaching final outcome: {episode.final_result}."
        )

        # What failed
        failures = []
        for att in episode.attempts:
            if att.evaluation == "FAIL":
                failures.append(
                    f"Attempt {att.attempt_id} failed with observation: {att.observation.strip()[:80]} "
                    f"(Diagnosis: {att.diagnosis or 'None'})"
                )
        if not failures:
            failures.append("No failures recorded; solved on first attempt.")

        # What worked
        if is_success and episode.attempts:
            last_att = episode.attempts[-1]
            what_worked = f"Verified action: {last_att.action.strip()[:100]}"
            why_it_worked = (
                f"Action satisfied all test assertions in ChakrKshetra with zero syntax/runtime errors."
            )
            confidence = 1.0 if total_attempts == 1 else 0.85
        else:
            what_worked = "Did not converge to verified solution."
            why_it_worked = "Remaining unresolved failures in execution environment."
            confidence = 0.0

        # Transfer insight
        if is_success:
            what_can_transfer = (
                f"Defect repair patterns and state alignments within task family '{family}'."
            )
        else:
            what_can_transfer = "No transferable pattern; episode unverified."

        # Boundary limits
        boundaries = [
            f"Do not generalize across different task families (e.g. from {family} to unrelated domains).",
            "Requires matching entrypoint signatures and test assertions.",
            "Valid only within deterministic, sandboxed execution contexts.",
        ]

        return EpisodeReflection(
            episode_id=episode.episode_id,
            task_id=task_id,
            task_family=family,
            what_happened=summary,
            what_failed=failures,
            what_worked=what_worked,
            why_it_worked=why_it_worked,
            what_can_transfer=what_can_transfer,
            boundary_limits=boundaries,
            transfer_confidence=confidence,
            timestamp_utc=time.time(),
        )
