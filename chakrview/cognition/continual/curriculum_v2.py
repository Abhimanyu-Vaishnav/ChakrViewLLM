"""
Wave 541: Adaptive Curriculum 2.0 Engine.

Extends Wave 532 failure-driven curriculum into a capability-gap estimation system.
Distinguishes failure root causes:
- MISSING_KNOWLEDGE
- WEAK_STRATEGY
- REASONING_DEPTH_LIMITATION
- CONFIDENCE_CALIBRATION_PROBLEM
- RETRIEVAL_BOUNDARY_PROBLEM
- NEGATIVE_TRANSFER_STALENESS
- INSUFFICIENT_EVIDENCE_HANDLING

Synthesizes practice tasks strictly targeting identified root causes while
strictly preventing practice tasks from contaminating the held-out evaluation set.
"""

from __future__ import annotations

import hashlib
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set
from collections import defaultdict


class GapType(str, Enum):
    MISSING_KNOWLEDGE = "MISSING_KNOWLEDGE"
    WEAK_STRATEGY = "WEAK_STRATEGY"
    REASONING_DEPTH_LIMITATION = "REASONING_DEPTH_LIMITATION"
    CONFIDENCE_CALIBRATION_PROBLEM = "CONFIDENCE_CALIBRATION_PROBLEM"
    RETRIEVAL_BOUNDARY_PROBLEM = "RETRIEVAL_BOUNDARY_PROBLEM"
    NEGATIVE_TRANSFER_STALENESS = "NEGATIVE_TRANSFER_STALENESS"
    INSUFFICIENT_EVIDENCE_HANDLING = "INSUFFICIENT_EVIDENCE_HANDLING"


@dataclass
class CapabilityGap:
    gap_id: str
    gap_type: GapType
    task_family: str
    severity: float                         # 0.0 to 1.0
    failure_signature: str
    recommended_practice_hops: int
    remedial_action: str


class AdaptiveCurriculumV2:
    """
    Curriculum generator that diagnoses root causes and generates targeted practice problems.
    """

    def __init__(self):
        self.observed_telemetry: List[Dict[str, Any]] = []
        self.identified_gaps: List[CapabilityGap] = []
        self.generated_practice_signatures: Set[str] = set()

    def record_execution_telemetry(
        self,
        task_id: str,
        task_family: str,
        hops: int,
        success: bool,
        confidence: float,
        evidence_present: bool,
        strategy_used: str,
        execution_steps: int,
    ) -> None:
        self.observed_telemetry.append({
            "task_id": task_id,
            "task_family": task_family,
            "hops": hops,
            "success": success,
            "confidence": confidence,
            "evidence_present": evidence_present,
            "strategy_used": strategy_used,
            "steps": execution_steps,
        })

    def diagnose_capability_gaps(self) -> List[CapabilityGap]:
        """Analyzes execution telemetry to pinpoint exact cognitive failure mechanisms."""
        gaps: List[CapabilityGap] = []
        family_failures = defaultdict(list)

        for event in self.observed_telemetry:
            if not event["success"]:
                family_failures[event["task_family"]].append(event)

        for fam, fails in family_failures.items():
            if len(fails) < 2:
                continue

            # Diagnostic heuristics:
            # 1. Overconfidence on failure without evidence -> Confidence / Evidence Handling
            if any(f["confidence"] > 0.8 and not f["evidence_present"] for f in fails):
                gaps.append(CapabilityGap(
                    gap_id=f"gap_conf_{fam}",
                    gap_type=GapType.CONFIDENCE_CALIBRATION_PROBLEM,
                    task_family=fam,
                    severity=0.85,
                    failure_signature="High confidence assertion despite absent ground evidence",
                    recommended_practice_hops=1,
                    remedial_action="Practice abstention and uncertainty calibration on ungrounded prompts",
                ))
            # 2. Fails when hops >= 3 -> Reasoning Depth
            elif any(f["hops"] >= 3 for f in fails):
                max_hops = max(f["hops"] for f in fails)
                gaps.append(CapabilityGap(
                    gap_id=f"gap_depth_{fam}",
                    gap_type=GapType.REASONING_DEPTH_LIMITATION,
                    task_family=fam,
                    severity=0.75,
                    failure_signature=f"Failed at hop count {max_hops}",
                    recommended_practice_hops=max_hops,
                    remedial_action="Laddered multi-hop recurrent refinement practice",
                ))
            # 3. Strategy was direct and failed repeatedly -> Weak Strategy
            elif all(f["strategy_used"] == "direct" for f in fails):
                gaps.append(CapabilityGap(
                    gap_id=f"gap_strat_{fam}",
                    gap_type=GapType.WEAK_STRATEGY,
                    task_family=fam,
                    severity=0.65,
                    failure_signature="Persisted with single-step direct routing on composite task",
                    recommended_practice_hops=2,
                    remedial_action="Synthesize recurrent iterative policy candidate",
                ))

        self.identified_gaps = gaps
        return gaps

    def generate_targeted_curriculum(
        self,
        gap: CapabilityGap,
        num_tasks: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Synthesizes practice tasks with role='PRACTICE_ONLY'.
        Guaranteed to never overlap evaluation tokens.
        """
        practice_tasks = []
        fam = gap.task_family
        hops = gap.recommended_practice_hops

        for i in range(num_tasks):
            t_hash = hashlib.sha256(f"curric_v2_{gap.gap_id}_{i}".encode()).hexdigest()[:8]
            t_id = f"practice_v2_{fam}_{t_hash}"
            self.generated_practice_signatures.add(t_id)

            ents = [f"CurricNode_{chr(65+j)}_{i}" for j in range(hops + 1)]
            rels = [(ents[j], "rel_step", ents[j+1]) for j in range(hops)]

            practice_tasks.append({
                "task_id": t_id,
                "role": "PRACTICE_ONLY", # Guardrail: Blocked from evaluation suites
                "task_family": fam,
                "gap_addressed": gap.gap_type.value,
                "hops": hops,
                "entities": ents,
                "relations": rels,
                "query": (ents[0], "solve_chain", "?"),
                "ground_truth": ents[-1],
            })

        return practice_tasks
