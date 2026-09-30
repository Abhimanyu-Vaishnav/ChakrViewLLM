"""
ChakrView Step 57: Learning Episode & Attempt Abstractions.

Defines:
- Attempt: Snapshot of a single action-observation-diagnostic turn.
- LearningEpisode: End-to-end container tracking problem-solving trajectory,
  executed stages, outcome, and promotion status.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class Attempt:
    attempt_id: int
    state: str
    action: str
    observation: str
    evaluation: str  # "PASS" | "FAIL"
    diagnosis: Optional[str] = None
    correction: Optional[str] = None
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LearningEpisode:
    episode_id: str
    task_id: str
    task_spec: Dict[str, Any]
    initial_context: Dict[str, Any]
    attempts: List[Attempt] = field(default_factory=list)
    executed_stages: List[str] = field(default_factory=list)
    final_result: str = "FAILURE"  # "SUCCESS" | "FAILURE"
    learning_status: str = "NO_CHANGE"  # "NEW_LEARNING" | "REPLAY_REUSE" | "NO_CHANGE"
    experience_record: Optional[Dict[str, Any]] = None
    candidate_update: Optional[Dict[str, Any]] = None
    verification_result: Dict[str, Any] = field(default_factory=dict)
    promotion_status: str = "NOT_PROMOTED"  # "NOT_PROMOTED" | "PROMOTED" | "QUARANTINED"
    timestamp_utc: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    def add_stage(self, stage_name: str) -> None:
        """Record an executed cognitive stage."""
        if stage_name not in self.executed_stages:
            self.executed_stages.append(stage_name)

    def add_attempt(self, attempt: Attempt) -> None:
        """Add an attempt snapshot to the episode."""
        self.attempts.append(attempt)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LearningEpisode:
        attempts_data = data.pop("attempts", [])
        episode = cls(**data)
        for att in attempts_data:
            episode.attempts.append(Attempt(**att))
        return episode
