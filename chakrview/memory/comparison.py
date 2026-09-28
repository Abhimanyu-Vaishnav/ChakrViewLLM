"""
Generic Case & Document Comparison Foundation for ChakrView (Step 16).

Provides structured comparison between current case/document/task and
historical persistent memories:
- Entity and attribute divergence
- Commonalities and differences
- Temporal progression
- Domain-agnostic comparison representations for CA, Doctor, Legal, or Business workflows
"""

from dataclasses import dataclass, field
import re
import time
from typing import Dict, List, Optional, Any, Set, Tuple
import uuid

from chakrview.memory.record import MemoryRecord


@dataclass
class ComparisonFacet:
    """Individual attribute or claim comparison."""
    facet_name: str
    current_value: str
    historical_value: str
    status: str  # "identical", "modified", "added", "removed"


@dataclass
class CaseComparisonResult:
    """
    Structured outcome of comparing current context against a historical memory.
    """
    comparison_id: str
    current_context: str
    historical_memory_id: str
    historical_content: str
    facets: List[ComparisonFacet]
    differences: List[str]
    commonalities: List[str]
    temporal_delta_days: float
    confidence: float
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "comparison_id": self.comparison_id,
            "current_context": self.current_context,
            "historical_memory_id": self.historical_memory_id,
            "historical_content": self.historical_content,
            "facets": [
                {
                    "facet_name": f.facet_name,
                    "current_value": f.current_value,
                    "historical_value": f.historical_value,
                    "status": f.status,
                }
                for f in self.facets
            ],
            "differences": self.differences,
            "commonalities": self.commonalities,
            "temporal_delta_days": self.temporal_delta_days,
            "confidence": self.confidence,
            "summary": self.summary,
        }


class CaseComparator:
    """
    Domain-agnostic case, document, and task comparator.
    """

    @staticmethod
    def _extract_clauses(text: str) -> List[str]:
        return [c.strip() for c in re.split(r"[;.\n]+", text) if len(c.strip()) > 3]

    @staticmethod
    def _extract_key_values(text: str) -> Dict[str, str]:
        kv: Dict[str, str] = {}
        matches = re.findall(
            r"([a-zA-Z\s]{3,25})\s*(?:=|\bis\b|:)\s*([0-9a-zA-Z\s₹$%.,]+)",
            text,
            re.IGNORECASE,
        )
        for k, v in matches:
            kv[k.strip().lower()] = v.strip().lower()
        return kv

    @classmethod
    def compare(
        cls,
        current_text: str,
        historical_record: MemoryRecord,
        current_timestamp: Optional[float] = None,
    ) -> CaseComparisonResult:
        """
        Compare current text against a historical memory record.
        """
        now = time.time() if current_timestamp is None else current_timestamp
        delta_days = max(0.0, (now - historical_record.temporal.created_at) / 86400.0)

        curr_kv = cls._extract_key_values(current_text)
        hist_kv = cls._extract_key_values(historical_record.content)

        facets: List[ComparisonFacet] = []
        differences: List[str] = []
        commonalities: List[str] = []

        all_keys = set(curr_kv.keys()).union(set(hist_kv.keys()))

        for k in sorted(all_keys):
            in_curr = k in curr_kv
            in_hist = k in hist_kv

            if in_curr and in_hist:
                v_curr = curr_kv[k]
                v_hist = hist_kv[k]
                if v_curr == v_hist:
                    facets.append(ComparisonFacet(k, v_curr, v_hist, "identical"))
                    commonalities.append(f"{k}: '{v_curr}' remained unchanged.")
                else:
                    facets.append(ComparisonFacet(k, v_curr, v_hist, "modified"))
                    differences.append(f"{k}: changed from '{v_hist}' to '{v_curr}'.")
            elif in_curr and not in_hist:
                facets.append(ComparisonFacet(k, curr_kv[k], "", "added"))
                differences.append(f"{k}: newly introduced ('{curr_kv[k]}').")
            elif not in_curr and in_hist:
                facets.append(ComparisonFacet(k, "", hist_kv[k], "removed"))
                differences.append(f"{k}: previously present ('{hist_kv[k]}'), absent currently.")

        # Fallback clause-level matching if no structured key-values found
        if not facets:
            curr_words = set(re.findall(r"\b\w{4,}\b", current_text.lower()))
            hist_words = set(re.findall(r"\b\w{4,}\b", historical_record.content.lower()))
            shared = curr_words.intersection(hist_words)
            diff_curr = curr_words - hist_words

            if shared:
                commonalities.append(f"Shared topics: {list(shared)[:5]}")
            if diff_curr:
                differences.append(f"Distinct in current context: {list(diff_curr)[:5]}")

        summary = (
            f"Compared current context with historical memory '{historical_record.memory_id}' "
            f"({delta_days:.1f} days ago). Found {len(commonalities)} commonalities and {len(differences)} differences."
        )

        return CaseComparisonResult(
            comparison_id=f"cmp_{uuid.uuid4().hex[:8]}",
            current_context=current_text,
            historical_memory_id=historical_record.memory_id,
            historical_content=historical_record.content,
            facets=facets,
            differences=differences,
            commonalities=commonalities,
            temporal_delta_days=delta_days,
            confidence=min(1.0, (historical_record.confidence + 0.9) / 2.0),
            summary=summary,
        )
