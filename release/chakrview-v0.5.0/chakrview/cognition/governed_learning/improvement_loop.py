"""
ChakrView Step 97: Governed Cognitive Improvement Loop.

Implements the bounded, closed-loop learning cycle:
TASK
  ↓
PLAN
  ↓
EXECUTE
  ↓
OBSERVE
  ↓
VERIFY
  ↓
SELF-EVALUATE (Step 95)
  ↓
EXTRACT LESSON (Step 96)
  ↓
STORE EXPERIENCE (Step 94)
  ↓
UPDATE STRATEGY (Step 98)
  ↓
NEXT TASK

CRITICAL INVARIANT:
Cognitive self-improvement improves persistent strategies, lessons, and project knowledge.
It strictly NEVER mutates the frozen ChakrMicro neural core weights (ΔW = 0).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from chakrview.cognition.ppb.task_models import PersistentTaskNode, TaskNodeStatus
from chakrview.cognition.governed_learning.experience_models import (
    GovernedExperienceRecord,
    GovernedExperienceStore,
)
from chakrview.cognition.governed_learning.self_evaluator import (
    GovernedSelfEvaluator,
    TaskAuditReport,
)
from chakrview.cognition.governed_learning.lesson_extractor import (
    CognitiveLesson,
    CognitiveLessonExtractor,
)
from chakrview.cognition.governed_learning.strategy_registry import (
    CognitiveStrategy,
    CognitiveStrategyRegistry,
    StrategyStatus,
)


@dataclass
class ImprovementCycleResult:
    """
    Structured outcome of an executed cognitive improvement cycle.
    """
    cycle_id: str
    task_id: str
    is_success: bool
    audit_report: TaskAuditReport
    experience_record: GovernedExperienceRecord
    extracted_lesson: Optional[CognitiveLesson]
    updated_strategy: Optional[CognitiveStrategy]
    neural_invariant_preserved: bool = True
    execution_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "task_id": self.task_id,
            "is_success": self.is_success,
            "audit_verdict": self.audit_report.verdict.value,
            "experience_id": self.experience_record.record_id,
            "lesson_id": self.extracted_lesson.lesson_id if self.extracted_lesson else None,
            "strategy_id": self.updated_strategy.strategy_id if self.updated_strategy else None,
            "neural_invariant_preserved": self.neural_invariant_preserved,
            "execution_time_ms": self.execution_time_ms,
        }


class GovernedCognitiveImprovementLoop:
    """
    Step 97: Bounded cognitive improvement loop.
    Extracts lessons, stores experiences, and refines strategies without neural modification.
    """

    def __init__(
        self,
        experience_store: GovernedExperienceStore,
        strategy_registry: CognitiveStrategyRegistry,
        self_evaluator: Optional[GovernedSelfEvaluator] = None,
    ) -> None:
        self.experience_store = experience_store
        self.strategy_registry = strategy_registry
        self.evaluator = self_evaluator or GovernedSelfEvaluator()

    def run_cycle_on_task_completion(
        self,
        project_id: str,
        node: PersistentTaskNode,
        execution_result: Dict[str, Any],
        error_message: Optional[str] = None,
        associated_strategy_id: Optional[str] = None,
    ) -> ImprovementCycleResult:
        """
        Executes complete learning cycle after a subtask finishes (success or failure).
        """
        t0 = time.perf_counter()
        cid = f"cycle_{node.node_id}_{int(time.time())}"

        # 1. Self-evaluate outcome
        audit = self.evaluator.audit_node_execution(
            node=node,
            execution_result=execution_result,
            error_message=error_message,
        )

        # 2. Store governed experience record
        exp_record = self.evaluator.convert_audit_to_experience(audit, project_id=project_id)
        self.experience_store.store_experience(exp_record)

        # 3. Extract cognitive lesson
        lesson = CognitiveLessonExtractor.extract_from_audit(audit, project_id=project_id)

        # 4. Update or register strategy
        updated_strategy: Optional[CognitiveStrategy] = None
        if associated_strategy_id:
            strat = self.strategy_registry.get_strategy(associated_strategy_id)
            if strat:
                strat.record_outcome(audit.is_success)
                self.strategy_registry.store_strategy(strat)
                updated_strategy = strat
        elif not audit.is_success and lesson:
            # Create candidate failure avoidance strategy
            new_strat_id = f"strat_avoid_{node.node_id}_{int(time.time())}"
            new_strat = CognitiveStrategy(
                strategy_id=new_strat_id,
                name=f"Avoidance Strategy for {node.title[:30]}",
                description=lesson.recommended_action,
                applicable_conditions=list(lesson.applicable_modules),
                expected_benefit="Prevents repeating detected execution defect",
                success_count=1,
                failure_count=0,
                confidence=0.6,
                status=StrategyStatus.CANDIDATE,
                provenance_lesson_id=lesson.lesson_id,
            )
            self.strategy_registry.store_strategy(new_strat)
            updated_strategy = new_strat

        elapsed = (time.perf_counter() - t0) * 1000.0

        return ImprovementCycleResult(
            cycle_id=cid,
            task_id=node.node_id,
            is_success=audit.is_success,
            audit_report=audit,
            experience_record=exp_record,
            extracted_lesson=lesson,
            updated_strategy=updated_strategy,
            neural_invariant_preserved=True,
            execution_time_ms=elapsed,
        )
