from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional


@dataclass
class RepositorySemanticRecord:
    """Semantic representation of a repository repair experience.

    Mirrors the STEP‑57 ExperienceRecord but is specialized for repository‑level
    patterns. All fields are deliberately simple and serializable.
    """

    memory_id: str
    task_family: str
    language: str
    framework: str
    repository_pattern: str
    symptom_signature: str
    root_cause_signature: str
    dependency_signature: str
    affected_modules: List[str]
    solution_pattern: str
    verification_requirements: List[str]
    known_boundaries: List[str]
    confidence: float  # 0.0 – 1.0 deterministic confidence
    evidence_count: int
    successful_episodes: int
    failed_episodes: int
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    updated_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    version: int = 1
    source_episode_ids: List[str] = field(default_factory=list)
    active_version: bool = True
    superseded_by: Optional[str] = None
    supersedes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RepositorySemanticRecord":
        return cls(**data)
