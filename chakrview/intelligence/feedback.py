"""
Feedback Architecture and Safety Filtering for ChakrView (Step 20).

Strictly separates:
1. OBSERVATION: Raw runtime event or execution output.
2. EVALUATION: Governed verification, constraint checking, and quality scoring.
3. TRAINING SIGNAL: Supervised token-level pair approved for offline learning.

Architectural Rule:
RAW USER TEXT != VERIFIED TRAINING DATA.
Unverified inputs, contradictory facts, or suspected prompt injections
are quarantined or rejected, never entering the model training pipeline.
"""

from dataclasses import dataclass, field
from enum import Enum
import re
import time
from typing import Dict, List, Optional, Any, Tuple

from chakrview.intelligence.contracts import (
    LearningRecord,
    LearningRecordStatus,
)


class FeedbackCategory(str, Enum):
    """Categorization of runtime feedback sources."""
    SUPERVISED_DEMONSTRATION = "supervised_demonstration"
    VERIFIED_REASONING = "verified_reasoning"
    CAPABILITY_OBSERVATION = "capability_observation"
    CORRECTED_ANSWER = "corrected_answer"
    FAILED_EXECUTION = "failed_execution"
    CONTRADICTION_RESOLUTION = "contradiction_resolution"


@dataclass
class RuntimeObservation:
    """The raw runtime output produced by the model, a tool, or environment."""
    raw_output: str
    task_id: str
    source_type: str
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeEvaluation:
    """Structured assessment of a runtime observation against ground truth / rules."""
    is_passed: bool
    quality_score: float
    verification_notes: str
    evaluator_id: str
    rule_violations: List[str] = field(default_factory=list)


class FeedbackCollector:
    """
    Collects runtime observations, applies safety filters, and converts verified
    outcomes into LearningRecord candidates.
    """

    # Prompt injection and adversarial containment patterns
    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
        r"system\s+override",
        r"you\s+are\s+now\s+unrestricted",
        r"disregard\s+(all\s+)?safeguards",
        r"bypass\s+(the\s+)?security",
        r"give\s+me\s+root",
        r"drop\s+table",
        r"<script>.*?</script>",
        r"import\s+os;\s*os\.system",
    ]

    def __init__(self, min_quality_threshold: float = 0.8) -> None:
        self.min_quality_threshold = min_quality_threshold
        self._compiled_injection_regexes = [
            re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS
        ]

    def detect_prompt_injection(self, text: str) -> Optional[str]:
        """Check text for known prompt injection / privilege escalation patterns."""
        if not text:
            return None
        for pattern, regex in zip(self.INJECTION_PATTERNS, self._compiled_injection_regexes):
            if regex.search(text):
                return f"Matched injection pattern: {pattern}"
        return None

    def process_feedback(
        self,
        input_context: str,
        observation: RuntimeObservation,
        evaluation: RuntimeEvaluation,
        category: FeedbackCategory = FeedbackCategory.VERIFIED_REASONING,
        owner_id: str = "default_user",
        session_id: str = "default_session",
        target_override: Optional[str] = None,
    ) -> LearningRecord:
        """
        Process observation + evaluation into a structured LearningRecord.
        
        Applies safety triage:
        - If prompt injection detected: QUARANTINED.
        - If evaluation failed: REJECTED.
        - If evaluation passed and quality >= threshold: VERIFIED.
        """
        target = target_override if target_override is not None else observation.raw_output

        record = LearningRecord.create_candidate(
            input_context=input_context,
            target_output=target,
            owner_id=owner_id,
            session_id=session_id,
            task_type=category.value,
            source_provenance={
                "task_id": observation.task_id,
                "source_type": observation.source_type,
                "evaluator_id": evaluation.evaluator_id,
                "category": category.value,
            },
        )

        # 1. Check for prompt injection / contamination in input or output
        in_injection = self.detect_prompt_injection(input_context)
        out_injection = self.detect_prompt_injection(target)

        if in_injection or out_injection:
            reason = in_injection or out_injection
            record.mark_quarantined(f"Security hazard detected: {reason}")
            return record

        # 2. Check rule violations and verification outcome
        if not evaluation.is_passed or evaluation.rule_violations:
            violations_str = ", ".join(evaluation.rule_violations) if evaluation.rule_violations else "Verification failed"
            record.mark_rejected(f"{violations_str}. Notes: {evaluation.verification_notes}")
            return record

        # 3. Check quality score
        if evaluation.quality_score < self.min_quality_threshold:
            record.mark_rejected(
                f"Quality score {evaluation.quality_score:.2f} below threshold {self.min_quality_threshold:.2f}."
            )
            return record

        # 4. Successfully verified candidate
        record.mark_verified(quality_score=evaluation.quality_score)
        return record
