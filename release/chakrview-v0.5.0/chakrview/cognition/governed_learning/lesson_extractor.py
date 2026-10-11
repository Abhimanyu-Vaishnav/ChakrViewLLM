"""
ChakrView Step 96: Cognitive Lesson Extraction & Conflict Resolution.

Extracts structured, provenance-tracked lessons from experiences and audits.
Enforces:
- Categorization of lessons: REASONING_PATTERN, FAILURE_AVOIDANCE, CONVENTION,
  DEPENDENCY_RELATION, RESOURCE_CONSTRAINT, TOOL_LIMITATION, VERIFICATION_STRATEGY.
- Epistemic authority ranking during conflict resolution:
  FACT (100) > OBSERVATION (80) > LESSON (60) > INFERENCE (40) > HYPOTHESIS (20) > UNKNOWN (0).
- A lesson or inference NEVER overrides a stronger verified FACT.
- Historical version lineage tracking for lessons.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence, Tuple

from chakrview.cognition.governed_learning.experience_models import (
    EPISTEMIC_AUTHORITY_RANK,
    ExperienceEpistemicCategory,
    ExperienceProvenance,
    GovernedExperienceRecord,
)
from chakrview.cognition.governed_learning.self_evaluator import TaskAuditReport, FailureClass


class LessonCategory(str, Enum):
    """Classification of cognitive lessons extracted from operational work."""
    REASONING_PATTERN = "REASONING_PATTERN"
    FAILURE_AVOIDANCE = "FAILURE_AVOIDANCE"
    CONVENTION = "CONVENTION"
    DEPENDENCY_RELATION = "DEPENDENCY_RELATION"
    RESOURCE_CONSTRAINT = "RESOURCE_CONSTRAINT"
    TOOL_LIMITATION = "TOOL_LIMITATION"
    VERIFICATION_STRATEGY = "VERIFICATION_STRATEGY"


@dataclass
class CognitiveLesson:
    """
    Step 96: Structured cognitive lesson extracted from experience.
    """
    lesson_id: str
    project_id: str
    title: str
    summary: str
    category: LessonCategory
    applicable_modules: List[str]
    trigger_condition: str
    recommended_action: str
    confidence: float
    epistemic_category: ExperienceEpistemicCategory = ExperienceEpistemicCategory.LESSON
    parent_experience_id: Optional[str] = None
    version: int = 1
    previous_version_id: Optional[str] = None
    created_at_utc: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lesson_id": self.lesson_id,
            "project_id": self.project_id,
            "title": self.title,
            "summary": self.summary,
            "category": self.category.value,
            "applicable_modules": list(self.applicable_modules),
            "trigger_condition": self.trigger_condition,
            "recommended_action": self.recommended_action,
            "confidence": self.confidence,
            "epistemic_category": self.epistemic_category.value,
            "parent_experience_id": self.parent_experience_id,
            "version": self.version,
            "previous_version_id": self.previous_version_id,
            "created_at_utc": self.created_at_utc,
        }


class ConflictResolutionOutcome(Enum):
    PRESERVE_EXISTING = "PRESERVE_EXISTING"  # Existing record has higher or equal authority
    SUPERSEDE = "SUPERSEDE"                  # New record has strictly higher authority
    CONTESTED = "CONTESTED"                  # Same authority rank but conflicting assertions


class CognitiveLessonExtractor:
    """
    Step 96: Extracts lessons and resolves conflicts according to epistemic ranks.
    """

    @staticmethod
    def extract_from_audit(audit: TaskAuditReport, project_id: str) -> Optional[CognitiveLesson]:
        """Synthesizes a lesson from an audit report."""
        if audit.is_success:
            # Positive operational lesson
            lid = f"lsn_succ_{audit.task_id}_{int(time.time())}"
            return CognitiveLesson(
                lesson_id=lid,
                project_id=project_id,
                title=f"Effective execution pattern for {audit.stated_goal[:40]}",
                summary=audit.extracted_lesson or "Task succeeded under current execution strategy",
                category=LessonCategory.REASONING_PATTERN,
                applicable_modules=list(audit.affected_modules),
                trigger_condition=f"When executing task: {audit.stated_goal[:40]}",
                recommended_action="Reuse successful execution path and verification checks",
                confidence=0.85,
                parent_experience_id=f"exp_{audit.audit_id}",
            )
        else:
            # Failure avoidance lesson
            f = audit.failure_analysis
            cat = LessonCategory.FAILURE_AVOIDANCE
            if f and f.failure_class == FailureClass.RESOURCE_EXHAUSTION:
                cat = LessonCategory.RESOURCE_CONSTRAINT
            elif f and f.failure_class == FailureClass.DEPENDENCY_MISMATCH:
                cat = LessonCategory.DEPENDENCY_RELATION

            lid = f"lsn_fail_{audit.task_id}_{int(time.time())}"
            rule = f.reusable_avoidance_rule if f else "Verify dependencies and prerequisites before execution"
            return CognitiveLesson(
                lesson_id=lid,
                project_id=project_id,
                title=f"Failure avoidance for {f.failure_class.value if f else 'unknown failure'}",
                summary=audit.extracted_lesson or "Avoid execution path that resulted in verified failure",
                category=cat,
                applicable_modules=list(audit.affected_modules),
                trigger_condition=f"When encountering {f.failure_class.value if f else 'similar task'}",
                recommended_action=rule,
                confidence=0.90,
                parent_experience_id=f"exp_{audit.audit_id}",
            )

    @staticmethod
    def resolve_epistemic_conflict(
        existing_category: ExperienceEpistemicCategory,
        new_category: ExperienceEpistemicCategory,
    ) -> ConflictResolutionOutcome:
        """
        Enforces strict authority hierarchy:
        FACT (100) > OBSERVATION (80) > LESSON (60) > INFERENCE (40) > HYPOTHESIS (20) > UNKNOWN (0).
        A lesson or inference can NEVER supersede a verified FACT.
        """
        existing_rank = EPISTEMIC_AUTHORITY_RANK.get(existing_category, 0)
        new_rank = EPISTEMIC_AUTHORITY_RANK.get(new_category, 0)

        if new_rank > existing_rank:
            return ConflictResolutionOutcome.SUPERSEDE
        elif new_rank < existing_rank:
            return ConflictResolutionOutcome.PRESERVE_EXISTING
        else:
            return ConflictResolutionOutcome.CONTESTED
