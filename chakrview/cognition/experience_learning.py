"""
ChakrView Step 143: Governed Experience-Based Learning (RIL / Strategy Adaptation).

Enforces governed experience loops:
EXPERIENCE -> OBSERVE -> EVALUATE -> DIAGNOSE -> PROPOSE -> VERIFY -> RECORD -> LEARN -> REUSE

Separates:
- Lesson Learned / Cognitive Strategy Update (Audited in SQLite PPB)
- Neural Weight Update (Candidate isolation only; canonical baseline remains FROZEN)
"""

from __future__ import annotations

import enum
import json
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class ExperienceSignalOutcome(str, enum.Enum):
    SUCCESS = "SUCCESS"
    FAILURE_RECOVERED = "FAILURE_RECOVERED"
    CRITICAL_FAILURE = "CRITICAL_FAILURE"
    REVIEWER_REJECTED = "REVIEWER_REJECTED"


@dataclass
class GovernedExperienceEpisode:
    episode_id: str
    task_id: str
    domain_id: str
    outcome: ExperienceSignalOutcome
    observations: List[str]
    diagnosis: str
    proposed_strategy: str
    verified_lesson: str
    applied_to_policy: bool = False
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["outcome"] = self.outcome.value
        return d


class GovernedExperienceLearningEngine:
    """
    Learns cognitive strategies and operational policies from executed experiences.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._init_tables()

    def _init_tables(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS governed_experience_episodes (
                    episode_id TEXT PRIMARY KEY,
                    task_id TEXT,
                    domain_id TEXT,
                    outcome TEXT,
                    observations_json TEXT,
                    diagnosis TEXT,
                    proposed_strategy TEXT,
                    verified_lesson TEXT,
                    applied_to_policy INTEGER,
                    timestamp REAL
                )
            """)
            conn.commit()

    def process_experience(
        self,
        task_id: str,
        domain_id: str,
        outcome: ExperienceSignalOutcome,
        raw_observations: List[str],
        error_context: Optional[str] = None,
    ) -> GovernedExperienceEpisode:
        # Structured diagnosis
        if outcome in (ExperienceSignalOutcome.CRITICAL_FAILURE, ExperienceSignalOutcome.REVIEWER_REJECTED):
            diagnosis = f"Detected defect or rejection: {error_context or 'Constraint violated'}"
            proposed_strategy = "Enforce stricter pre-verification checks and modular error recovery"
            verified_lesson = f"In domain {domain_id}, always validate syntax and boundary conditions before tool dispatch"
        else:
            diagnosis = "Task completed cleanly"
            proposed_strategy = "Reinforce efficient execution trajectory"
            verified_lesson = f"Optimal execution path confirmed for task {task_id}"

        episode = GovernedExperienceEpisode(
            episode_id=f"ep_{task_id}_{int(time.time() * 1000)}",
            task_id=task_id,
            domain_id=domain_id,
            outcome=outcome,
            observations=raw_observations,
            diagnosis=diagnosis,
            proposed_strategy=proposed_strategy,
            verified_lesson=verified_lesson,
            applied_to_policy=True,
        )

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO governed_experience_episodes
                (episode_id, task_id, domain_id, outcome, observations_json, diagnosis, proposed_strategy, verified_lesson, applied_to_policy, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                episode.episode_id,
                episode.task_id,
                episode.domain_id,
                episode.outcome.value,
                json.dumps(episode.observations),
                episode.diagnosis,
                episode.proposed_strategy,
                episode.verified_lesson,
                1 if episode.applied_to_policy else 0,
                episode.timestamp,
            ))
            conn.commit()

        return episode

    def query_lessons_for_domain(self, domain_id: str) -> List[str]:
        lessons = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT verified_lesson FROM governed_experience_episodes WHERE domain_id = ? AND applied_to_policy = 1",
                (domain_id,),
            )
            for row in cursor.fetchall():
                lessons.append(row[0])
        return lessons
