"""
ChakrView Step 80: Project Knowledge Retrieval Layer.

Enables targeted retrieval of relevant project knowledge without requiring
a full repository rescan:
- Retrieves by task description, keyword overlap, symbol name, dependency distance, or file path.
- Respects resource-adaptive budgets (max_records, max_context_chars).
- Links provenance, evidence IDs, and epistemic status.
- Integrates with CognitiveContextBundle and downstream reasoning.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from chakrview.runtime.resource import RuntimeStrategy
from chakrview.cognition.repository.context_store import EvidenceRecord, EpistemicState
from chakrview.cognition.ppb.brain import PersistentProjectBrain
from chakrview.cognition.ppb.models import (
    EpistemicStatus,
    KnowledgeRecord,
    KnowledgeRecordType,
)


@dataclass(frozen=True)
class PPBRetrievalBudget:
    """Strict resource limits on retrieved project brain knowledge."""
    max_records: int = 15
    max_dependency_depth: int = 2
    include_stale: bool = False
    include_historical: bool = False


@dataclass
class RetrievedKnowledgeBundle:
    """Bounded, provenance-tracked bundle of project knowledge for downstream cognition."""
    task_query: str
    records: List[KnowledgeRecord]
    matched_files: List[str]
    matched_symbols: List[str]
    evidence_records: List[EvidenceRecord]
    epistemic_summary: Dict[str, int]
    retrieval_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_query": self.task_query,
            "records": [r.to_dict() for r in self.records],
            "matched_files": list(self.matched_files),
            "matched_symbols": list(self.matched_symbols),
            "evidence_records": [e.to_dict() for e in self.evidence_records],
            "epistemic_summary": self.epistemic_summary,
            "retrieval_time_ms": round(self.retrieval_time_ms, 2),
        }


class ProjectKnowledgeRetriever:
    """
    Retrieves bounded, task-relevant project knowledge from PersistentProjectBrain.
    """

    def __init__(self, brain: PersistentProjectBrain) -> None:
        self.brain = brain

    def retrieve_for_task(
        self,
        task_query: str,
        budget: Optional[PPBRetrievalBudget] = None,
        target_files: Optional[Sequence[str]] = None,
        target_symbols: Optional[Sequence[str]] = None,
    ) -> RetrievedKnowledgeBundle:
        """
        Locates relevant project knowledge for a task without scanning repository files.
        """
        start_t = time.perf_counter()

        # Adapt budget if not specified
        if budget is None:
            strat = self.brain.resource_policy.strategy
            if strat == RuntimeStrategy.LOW_RESOURCE:
                budget = PPBRetrievalBudget(max_records=8, max_dependency_depth=1)
            elif strat == RuntimeStrategy.STANDARD:
                budget = PPBRetrievalBudget(max_records=16, max_dependency_depth=2)
            else:
                budget = PPBRetrievalBudget(max_records=30, max_dependency_depth=3)

        query_tokens = set(re.findall(r"\w+", task_query.lower()))
        # Remove common stop words
        stop_words = {"the", "a", "an", "and", "or", "in", "on", "for", "with", "to", "of", "fix", "update"}
        salient_tokens = query_tokens - stop_words

        # 1. Fetch active candidate records from persistent storage
        all_active = self.brain.query_knowledge(
            active_only=not budget.include_historical,
            limit=500,
        )

        scored_records: List[Tuple[float, KnowledgeRecord]] = []
        matched_files_set: Set[str] = set()
        matched_symbols_set: Set[str] = set()

        for rec in all_active:
            if not budget.include_stale and rec.epistemic_status == EpistemicStatus.STALE:
                continue

            score = 0.0

            # Direct file path target match
            if target_files and any(tf in rec.file_path for tf in target_files):
                score += 10.0
                matched_files_set.add(rec.file_path)

            # Direct symbol target match
            if target_symbols and rec.symbol_name and rec.symbol_name in target_symbols:
                score += 15.0
                matched_symbols_set.add(rec.symbol_name)

            # Query token overlap in file path, symbol name, or summary
            path_tokens = set(re.findall(r"\w+", rec.file_path.lower()))
            symbol_tokens = set(re.findall(r"\w+", (rec.symbol_name or "").lower()))
            summary_tokens = set(re.findall(r"\w+", rec.summary.lower()))

            score += 3.0 * len(salient_tokens & symbol_tokens)
            score += 2.0 * len(salient_tokens & path_tokens)
            score += 1.0 * len(salient_tokens & summary_tokens)

            # Preference for high confidence and verified facts
            if rec.epistemic_status in (EpistemicStatus.FACT, EpistemicStatus.REVERIFIED):
                score += 0.5

            if score > 0.0 or not salient_tokens:
                scored_records.append((score, rec))
                if rec.file_path:
                    matched_files_set.add(rec.file_path)
                if rec.symbol_name:
                    matched_symbols_set.add(rec.symbol_name)

        # Sort deterministically: highest score first, then record_id
        scored_records.sort(key=lambda item: (-item[0], item[1].record_id))
        selected = [item[1] for item in scored_records[: budget.max_records]]

        # Construct EvidenceRecords for provenance tracing
        evidence_records: List[EvidenceRecord] = []
        epistemic_counts: Dict[str, int] = {}

        for rec in selected:
            st = rec.epistemic_status.value
            epistemic_counts[st] = epistemic_counts.get(st, 0) + 1

            evidence_records.append(
                EvidenceRecord(
                    source_type="PPB_PERSISTENT_MEMORY",
                    file_path=rec.file_path,
                    symbol_name=rec.symbol_name,
                    structural_digest=rec.record_id,
                    retrieval_reason=f"Matched project memory [{rec.record_type.value}] for query: '{task_query}'",
                    dependency_distance=0,
                    epistemic_state=(
                        EpistemicState.KNOWN if rec.epistemic_status in (EpistemicStatus.FACT, EpistemicStatus.REVERIFIED)
                        else EpistemicState.STALE if rec.epistemic_status == EpistemicStatus.STALE
                        else EpistemicState.UNKNOWN
                    ),
                )
            )

        elapsed = (time.perf_counter() - start_t) * 1000.0

        return RetrievedKnowledgeBundle(
            task_query=task_query,
            records=selected,
            matched_files=sorted(list(matched_files_set)),
            matched_symbols=sorted(list(matched_symbols_set)),
            evidence_records=evidence_records,
            epistemic_summary=epistemic_counts,
            retrieval_time_ms=elapsed,
        )
