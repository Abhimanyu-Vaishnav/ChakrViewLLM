"""
ChakrView Step 57: Canonical Experience Record & Extractor.

Defines:
- ExperienceRecord: Serializable dataclass for validated experience.
- ExperienceExtractor: Extracts structured experience from a LearningEpisode.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

from chakrview.learning.episode import LearningEpisode


@dataclass
class ExperienceRecord:
    experience_id: str
    task_id: str
    task_family: str
    task_description: str
    context: Dict[str, Any]
    attempt_count: int
    initial_action: str
    failure_observation: Optional[str]
    diagnosis: Optional[str]
    correction: Optional[str]
    verified_result: str
    what_worked: str
    what_failed: str
    reusable_pattern: str
    verified: bool
    timestamp_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExperienceExtractor:
    """
    Extracts canonical ExperienceRecords from completed LearningEpisodes.
    Enforces that only verified successes become training candidates.
    """

    @staticmethod
    def extract_from_episode(episode: LearningEpisode) -> Optional[ExperienceRecord]:
        """Extract experience if episode produced a verified resolution."""
        if not episode.attempts:
            return None

        first_att = episode.attempts[0]
        last_att = episode.attempts[-1]

        is_success = (episode.final_result == "SUCCESS" and last_att.evaluation == "PASS")

        failure_obs = None
        diagnosis_str = None
        correction_str = None

        for att in episode.attempts:
            if att.evaluation == "FAIL":
                failure_obs = att.observation
                diagnosis_str = att.diagnosis
            if att.correction:
                correction_str = att.correction

        what_failed = failure_obs or "No recorded failure"
        what_worked = last_att.action if is_success else "Did not converge to verified solution"

        # Reusable pattern synthesis (bounded length)
        if is_success and diagnosis_str and correction_str:
            reusable_pattern = f"Diagnosis: {diagnosis_str}; Solution: {correction_str.strip()}"[:100]
        elif is_success:
            reusable_pattern = f"Successful Action: {last_att.action.strip()}"[:100]
        else:
            reusable_pattern = "Unresolved failure"

        spec = episode.task_spec
        family = spec.get("category", "general")
        desc = spec.get("description", episode.task_id)

        return ExperienceRecord(
            experience_id=f"exp_cog_{episode.task_id}_{int(time.time())}",
            task_id=episode.task_id,
            task_family=family,
            task_description=desc,
            context=dict(episode.initial_context),
            attempt_count=len(episode.attempts),
            initial_action=first_att.action,
            failure_observation=failure_obs,
            diagnosis=diagnosis_str,
            correction=correction_str,
            verified_result=episode.final_result,
            what_worked=what_worked,
            what_failed=what_failed,
            reusable_pattern=reusable_pattern,
            verified=is_success,
            timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
