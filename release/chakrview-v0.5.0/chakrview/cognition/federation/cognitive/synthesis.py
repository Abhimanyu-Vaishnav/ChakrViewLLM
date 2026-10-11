"""
Cognitive Synthesis Engine for Step 42.

Aggregates multi-node cognitive reasoning results with:
- Anti-majority evidence preservation (CT-12): minority views are preserved
  as formal ConflictRecords rather than silently overridden by majority vote.
- Deduplication of evidence items across workers.
- Deterministic conflict detection and recording.

AXIOM: Anti-Majority Evidence Principle
  When distributed workers disagree, both the majority conclusion AND all
  minority conclusions are documented in ConflictRecords. Neither is discarded.
"""

import logging
import uuid
from typing import Any, Dict, List

from chakrview.cognition.federation.cognitive.errors import CognitiveSynthesisError
from chakrview.cognition.federation.cognitive.models import (
    ConflictRecord,
    SynthesisResult,
)

logger = logging.getLogger("chakrview.federation.cognitive.synthesis")


class CognitiveSynthesisEngine:
    """
    Multi-node cognitive evidence aggregator.

    Principle: Preserve minority evidence.
    When workers disagree, both majority and minority conclusions are documented
    in formal ConflictRecords — never silently discarded.
    """

    def synthesize(
        self,
        episode_id: str,
        step_id: str,
        step_results: List[Dict[str, Any]],
    ) -> SynthesisResult:
        """
        Aggregate results from multiple cognitive workers for a given step.

        Each result dict is expected to contain:
            node_id     str          Worker node that produced the result
            conclusion  str          Worker's primary conclusion text
            evidence    List[str]    Supporting evidence strings
            hypotheses  List[Dict]   Active hypotheses

        Returns:
            SynthesisResult with synthesized_conclusion, aggregated evidence,
            and ConflictRecords for any minority-opinion disagreements.

        Raises:
            CognitiveSynthesisError if step_results is empty.
        """
        if not step_results:
            raise CognitiveSynthesisError(
                f"No results to synthesize for step '{step_id}' in episode '{episode_id}'"
            )

        conclusions: List[str] = []
        all_evidence: List[str] = []
        all_hypotheses: List[Dict[str, Any]] = []
        node_ids: List[str] = []
        conflict_records: List[ConflictRecord] = []

        # Collect all outputs
        for result in step_results:
            node_id = result.get("node_id", "unknown")
            conclusion = result.get("conclusion", "")
            evidence = result.get("evidence", [])
            hypotheses = result.get("hypotheses", [])

            if node_id not in node_ids:
                node_ids.append(node_id)
            if conclusion:
                conclusions.append(conclusion)
            all_evidence.extend(evidence)
            all_hypotheses.extend(hypotheses)

        # Determine unique conclusions (insertion-order preserved)
        seen: dict = {}
        unique_conclusions: List[str] = []
        for c in conclusions:
            if c not in seen:
                seen[c] = True
                unique_conclusions.append(c)

        majority_conclusion = unique_conclusions[0] if unique_conclusions else ""

        # CT-12: Preserve minority conclusions as ConflictRecords
        if len(unique_conclusions) > 1:
            minority_node_id = "unknown"
            for result in step_results:
                c = result.get("conclusion", "")
                nid = result.get("node_id", "unknown")
                if c and c != majority_conclusion:
                    minority_node_id = nid
                    break

            for minority_conclusion in unique_conclusions[1:]:
                conflict_records.append(
                    ConflictRecord(
                        record_id=f"cr_{uuid.uuid4().hex[:10]}",
                        episode_id=episode_id,
                        step_id=step_id,
                        dissenting_node_id=minority_node_id,
                        majority_conclusion=majority_conclusion,
                        minority_conclusion=minority_conclusion,
                        contradiction_description=(
                            f"Worker disagreement at step '{step_id}': "
                            f"majority='{majority_conclusion[:80]}'; "
                            f"minority='{minority_conclusion[:80]}'"
                        ),
                    )
                )

        # Deduplicate evidence
        seen_ev: set = set()
        unique_evidence: List[str] = []
        for ev in all_evidence:
            if ev not in seen_ev:
                seen_ev.add(ev)
                unique_evidence.append(ev)

        # Build synthesized conclusion
        if len(unique_conclusions) == 1:
            synthesized = unique_conclusions[0]
        elif unique_conclusions:
            synthesized = (
                f"[Majority: {majority_conclusion}] "
                f"[Minority views preserved in {len(conflict_records)} conflict record(s)]"
            )
        else:
            synthesized = "[No conclusive output from distributed workers]"

        synthesis = SynthesisResult(
            synthesis_id=f"syn_{uuid.uuid4().hex[:10]}",
            episode_id=episode_id,
            synthesized_conclusion=synthesized,
            evidence_count=len(unique_evidence),
            hypothesis_count=len(all_hypotheses),
            conflict_records=conflict_records,
            participating_nodes=node_ids,
            has_minority_evidence=len(conflict_records) > 0,
        )

        logger.info(
            "Synthesis complete: episode='%s' step='%s' workers=%d conflicts=%d minority=%s",
            episode_id,
            step_id,
            len(node_ids),
            len(conflict_records),
            synthesis.has_minority_evidence,
        )
        return synthesis
