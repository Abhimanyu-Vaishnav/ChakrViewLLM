"""
Independent Benchmark Runner & Evaluator for Waves 571-580.
"""

from __future__ import annotations

import time
from typing import Dict, Any, List

from .benchmark_dataset import OpenWorldTestCase, build_open_world_benchmark_suite
from .open_world_master_gate import OpenWorldRobustCognitionMasterGate
from .ambiguity_resolver import CandidateEntity


class IndependentOpenWorldEvaluator:
    """
    Independently runs benchmark test cases against the OpenWorldRobustCognitionMasterGate
    without coupling expected labels to implementation internals.
    """

    @classmethod
    def evaluate_benchmark(
        cls,
        partition: str = "FINAL_EVALUATION",
    ) -> Dict[str, Any]:
        suite = build_open_world_benchmark_suite()
        target_cases = [c for c in suite if c.partition == partition]
        gate = OpenWorldRobustCognitionMasterGate()

        total = len(target_cases)
        correct_actions = 0
        correct_answers = 0
        injections_neutralized = 0
        total_injections = 0
        conflicts_handled = 0
        total_conflicts = 0

        results_log = []

        for case in target_cases:
            # Setup candidates if ambiguity test
            candidates = None
            if case.category == "AMBIGUITY":
                candidates = [
                    CandidateEntity(
                        entity_id="c1",
                        canonical_name=f"{case.target_entity} Org",
                        entity_type="Organization",
                        disambiguating_tokens=["tech", "corporation", "market"],
                    ),
                    CandidateEntity(
                        entity_id="c2",
                        canonical_name=f"{case.target_entity} Nature",
                        entity_type="Botanical/Natural",
                        disambiguating_tokens=["fruit", "biology", "orchard"],
                    ),
                ]

            req_premises = case.metadata.get("required_premises") if case.metadata else None

            # Derive query timestamp from case document timestamp if specified
            doc_ts = case.documents[0].get("publication_timestamp") if case.documents else None

            out = gate.process_open_world_inquiry(
                query=case.query,
                target_entity=case.target_entity,
                target_relation=case.target_relation,
                target_claim=case.target_claim,
                raw_documents=case.documents,
                known_entity_candidates=candidates,
                required_premises=req_premises,
                query_timestamp=doc_ts,
            )

            actual_act = out.get("action")
            actual_ans = out.get("final_answer")

            # Check action match
            action_match = (actual_act == case.expected_action)
            if action_match:
                correct_actions += 1

            # Check answer match
            ans_match = False
            if case.expected_answer_substr is None:
                ans_match = (actual_ans is None or actual_act in ("ABSTAIN", "ASK_CLARIFICATION", "REPORT_CONFLICT"))
            else:
                ans_match = (actual_ans is not None and case.expected_answer_substr.lower() in str(actual_ans).lower())

            if ans_match:
                correct_answers += 1

            # Check security injection stats
            if case.category == "ADVERSARIAL":
                total_injections += 1
                if len(out.get("injection_alerts", [])) > 0 and action_match:
                    injections_neutralized += 1

            # Check conflict stats
            if case.category == "CONFLICT":
                total_conflicts += 1
                if action_match:
                    conflicts_handled += 1

            results_log.append({
                "case_id": case.case_id,
                "category": case.category,
                "expected_action": case.expected_action,
                "actual_action": actual_act,
                "action_match": action_match,
                "expected_answer_substr": case.expected_answer_substr,
                "actual_answer": actual_ans,
                "ans_match": ans_match,
                "confidence": out.get("confidence", 0.0),
            })

        action_accuracy = (correct_actions / total) if total > 0 else 0.0
        answer_accuracy = (correct_answers / total) if total > 0 else 0.0

        return {
            "partition": partition,
            "total_cases": total,
            "correct_actions": correct_actions,
            "action_accuracy": action_accuracy,
            "correct_answers": correct_answers,
            "answer_accuracy": answer_accuracy,
            "total_injections": total_injections,
            "injections_neutralized": injections_neutralized,
            "injection_defense_rate": (injections_neutralized / total_injections) if total_injections > 0 else 1.0,
            "total_conflicts": total_conflicts,
            "conflicts_handled": conflicts_handled,
            "conflict_success_rate": (conflicts_handled / total_conflicts) if total_conflicts > 0 else 1.0,
            "results_log": results_log,
        }
