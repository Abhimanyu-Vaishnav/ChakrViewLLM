"""
Deterministic Memory Importance & Confidence Scoring for ChakrView (Step 16).

Separates two orthogonal dimensions:
1. Importance: "How useful and significant is this memory for future interactions?"
2. Confidence: "How certain are we that the information is factually accurate?"

Never collapses importance and confidence into a single scalar.
Modular design allows deterministic heuristic scoring now and future learned models later.
"""

from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Any

from chakrview.memory.record import MemoryType, MemoryProvenance


@dataclass
class MemoryScoreResult:
    """Calculated dual-metric score result."""
    importance: float
    confidence: float
    importance_reasons: List[str]
    confidence_reasons: List[str]


class MemoryScorer:
    """
    Deterministic rule-based scorer for memory importance and confidence.
    """

    # Markers of explicit user directive indicating high importance
    EXPLICIT_IMPORTANCE_MARKERS = [
        "remember",
        "always",
        "never",
        "my preference",
        "important",
        "note that",
        "critical",
        "mandatory",
        "rule",
        "client requirement",
    ]

    # Source confidence baselines
    SOURCE_CONFIDENCE_TABLE = {
        "explicit_user": 0.95,      # Direct user assertion
        "verified_task_result": 0.90, # Result passed Step 15 verifier
        "uploaded_document": 0.85,  # Extracted from source documents
        "task_result": 0.80,        # Standard tool/task execution
        "conversation": 0.65,       # Dialogue extraction
        "inferred": 0.50,           # Deduced or synthesized
    }

    @classmethod
    def calculate_importance(
        cls,
        content: str,
        memory_type: MemoryType,
        provenance: Optional[MemoryProvenance] = None,
        occurrence_count: int = 1,
    ) -> float:
        """
        Calculate deterministic importance score in [0.0, 1.0].
        """
        score = 0.4  # baseline moderate importance
        text = content.lower()

        # 1. Type-based baseline
        if memory_type == MemoryType.USER_PROFILE:
            score += 0.25  # Durable user preferences have high long-term utility
        elif memory_type == MemoryType.SEMANTIC:
            score += 0.20  # Stable domain facts
        elif memory_type == MemoryType.EPISODIC:
            score += 0.10  # Interaction transcripts

        # 2. Explicit user importance cues
        if any(marker in text for marker in cls.EXPLICIT_IMPORTANCE_MARKERS):
            score += 0.20

        # 3. Frequency / repetition boost
        if occurrence_count > 1:
            repetition_boost = min(0.15, (occurrence_count - 1) * 0.05)
            score += repetition_boost

        # 4. Specificity boost (e.g. structured entities, numbers, percentages)
        if re.search(r"\b\d+(\.\d+)?\b", content):
            score += 0.05  # Contains concrete numeric data

        return min(1.0, max(0.1, score))

    @classmethod
    def calculate_confidence(
        cls,
        content: str,
        provenance: Optional[MemoryProvenance] = None,
        verification_passed: bool = True,
    ) -> float:
        """
        Calculate deterministic confidence score in [0.0, 1.0].
        """
        source_type = provenance.source_type if provenance else "conversation"
        base_confidence = cls.SOURCE_CONFIDENCE_TABLE.get(source_type, 0.60)

        # Penalty if verification failed or was ambiguous
        if not verification_passed:
            base_confidence -= 0.30

        # Confidence modifiers based on uncertainty phrases
        text = content.lower()
        if any(w in text for w in ["maybe", "perhaps", "might be", "possibly", "i guess", "not sure"]):
            base_confidence -= 0.20

        return min(1.0, max(0.1, base_confidence))

    @classmethod
    def score_memory(
        cls,
        content: str,
        memory_type: MemoryType,
        provenance: Optional[MemoryProvenance] = None,
        occurrence_count: int = 1,
        verification_passed: bool = True,
    ) -> MemoryScoreResult:
        """
        Compute dual-metric score with diagnostic audit reasons.
        """
        imp_reasons = []
        conf_reasons = []

        # Importance computation
        imp = cls.calculate_importance(content, memory_type, provenance, occurrence_count)
        imp_reasons.append(f"MemoryType: {memory_type.value}")
        if any(m in content.lower() for m in cls.EXPLICIT_IMPORTANCE_MARKERS):
            imp_reasons.append("Contains explicit user importance marker")
        if occurrence_count > 1:
            imp_reasons.append(f"Repetition count: {occurrence_count}")

        # Confidence computation
        conf = cls.calculate_confidence(content, provenance, verification_passed)
        src = provenance.source_type if provenance else "conversation"
        conf_reasons.append(f"Source reliability: {src}")
        if not verification_passed:
            conf_reasons.append("Verification assertion failed")

        return MemoryScoreResult(
            importance=imp,
            confidence=conf,
            importance_reasons=imp_reasons,
            confidence_reasons=conf_reasons,
        )
