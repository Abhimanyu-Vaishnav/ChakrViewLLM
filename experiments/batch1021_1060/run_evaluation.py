"""
Benchmark execution script for WP 1048–1052: Frozen 50-Item Evaluation.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict, List

from chakrview.runtime.local_knowledge_assistant import LocalKnowledgeAssistant


def run_benchmark():
    corpus_dir = Path("experiments/batch1021_1060/data/sample_corpus")
    bench_file = Path("experiments/batch1021_1060/data/held_out_50_benchmark.json")

    with open(bench_file, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    assistant = LocalKnowledgeAssistant()
    total_chunks = assistant.ingest_directory(corpus_dir)
    print(f"Indexed {total_chunks} chunks from {corpus_dir}")

    results = []
    total_q = len(benchmark)
    correct_count = 0
    citation_valid_count = 0
    citation_support_count = 0
    total_citations_evaluated = 0
    abstention_correct = 0
    total_abstention_targets = 0
    followup_correct = 0
    total_followup_targets = 0
    refutation_correct = 0
    total_refutation_targets = 0
    latencies = []

    for item in benchmark:
        qid = item["id"]
        cat = item["category"]
        q = item["query"]
        expected_type = item["expected_type"]
        expected_source = item.get("expected_source")
        expected_substring = item.get("expected_substring")

        assistant.reset_conversation()

        # Handle multi-turn context priming
        if item.get("context_query"):
            _ = assistant.ask(item["context_query"])

        resp = assistant.ask(q)
        latencies.append(resp.latency_ms)

        is_type_match = (resp.decision == expected_type)
        is_content_match = True
        cit_valid = True
        cit_supported = True

        if expected_type == "ANSWER":
            # Check content
            if expected_substring and expected_substring.lower() not in resp.answer.lower():
                is_content_match = False

            # Check citation
            if not resp.citations:
                cit_valid = False
                cit_supported = False
            else:
                for c in resp.citations:
                    total_citations_evaluated += 1
                    # Check document exists
                    if expected_source and c.get("doc_id") != expected_source:
                        cit_valid = False
                    # Check text exists in answer
                    if c.get("exact_text") and c.get("exact_text") not in resp.answer:
                        cit_supported = False

                if cit_valid:
                    citation_valid_count += 1
                if cit_supported:
                    citation_support_count += 1

        elif expected_type == "ABSTAIN":
            total_abstention_targets += 1
            if is_type_match:
                abstention_correct += 1

        elif expected_type == "REFUTE":
            total_refutation_targets += 1
            if is_type_match and (not expected_substring or expected_substring.lower() in resp.answer.lower()):
                refutation_correct += 1

        if cat == "follow_up":
            total_followup_targets += 1
            if is_type_match and is_content_match:
                followup_correct += 1

        item_passed = is_type_match and is_content_match and (cit_valid if expected_type == "ANSWER" else True)
        if item_passed:
            correct_count += 1

        results.append({
            "id": qid,
            "category": cat,
            "query": q,
            "expected_type": expected_type,
            "decision": resp.decision,
            "passed": item_passed,
            "citations": resp.citations,
            "answer": resp.answer,
            "latency_ms": resp.latency_ms,
        })

    latencies_sorted = sorted(latencies)
    median_lat = latencies_sorted[len(latencies_sorted) // 2]
    p95_lat = latencies_sorted[int(len(latencies_sorted) * 0.95)]

    metrics = {
        "total_questions": total_q,
        "correct_answers_or_abstentions": correct_count,
        "accuracy_pct": round(correct_count / total_q * 100, 2),
        "citation_validity_count": citation_valid_count,
        "citation_support_count": citation_support_count,
        "total_citations_evaluated": total_citations_evaluated,
        "correct_abstentions": f"{abstention_correct}/{total_abstention_targets}",
        "correct_refutations": f"{refutation_correct}/{total_refutation_targets}",
        "correct_followups": f"{followup_correct}/{total_followup_targets}",
        "latency_median_ms": median_lat,
        "latency_p95_ms": p95_lat,
        "details": results,
    }

    out_file = Path("experiments/batch1021_1060/reports/FINAL_METRICS.json")
    out_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\n================ BENCHMARK EVALUATION FINISHED ================")
    print(f"Total Questions: {total_q}")
    print(f"Passed: {correct_count}/{total_q} ({round(correct_count / total_q * 100, 1)}%)")
    print(f"Abstentions: {abstention_correct}/{total_abstention_targets}")
    print(f"Refutations: {refutation_correct}/{total_refutation_targets}")
    print(f"Follow-ups: {followup_correct}/{total_followup_targets}")
    print(f"Median Latency: {median_lat} ms | p95 Latency: {p95_lat} ms")
    print(f"Report written to: {out_file}")


if __name__ == "__main__":
    run_benchmark()
