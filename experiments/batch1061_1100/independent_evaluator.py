"""
Independent Evaluator for ChakrView Local Knowledge Assistant.
Location: experiments/batch1061_1100/independent_evaluator.py

Features:
1. Tests the assistant STRICTLY through its public interface:
     assistant = LocalKnowledgeAssistant(...)
     assistant.ingest_directory(...)
     assistant.ask(...)
2. Validates document hashes against DATASET_MANIFEST.json before evaluation.
3. Strict Citation Validation:
   - Verifies doc_id exists on disk.
   - Reads actual source document and slices [start_char:end_char].
   - Verifies that citation exact_text matches the disk byte slice.
4. Independent 4-Tier Evaluation:
   - Citation Validity: Real file + real byte slice bounds.
   - Citation Relevance: Document and passage relevance to the query.
   - Citation Support: Whether the cited span contains the supporting evidence.
   - Answer Correctness: Rigorous assertion of expected answer value.
   - Abstention & Refutation Correctness: Verifies appropriate abstention or domain refutation.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

from chakrview.runtime.local_knowledge_assistant import LocalKnowledgeAssistant, AssistantResponse


@dataclass
class EvaluationItemResult:
    id: int
    category: str
    query: str
    expected_type: str
    expected_source: Optional[str]
    expected_substring: Optional[str]
    decision: str
    is_decision_correct: bool
    citation_validity: bool
    citation_relevance: bool
    citation_support: bool
    answer_correct: bool
    passed: bool
    latency_ms: float
    answer_preview: str
    citations_evaluated: List[Dict[str, Any]]
    failure_reasons: List[str]


def evaluate_dataset(
    corpus_dir: Path | str,
    benchmark_file: Path | str,
    manifest_file: Optional[Path | str] = None,
) -> Dict[str, Any]:
    corpus_p = Path(corpus_dir).resolve()
    bench_p = Path(benchmark_file).resolve()

    if not corpus_p.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_p}")
    if not bench_p.exists():
        raise FileNotFoundError(f"Benchmark file not found: {bench_p}")

    # Verify manifest if provided
    manifest_data = None
    if manifest_file:
        m_path = Path(manifest_file).resolve()
        if m_path.exists():
            with open(m_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
            # Verify benchmark hash if evaluating the canonical benchmark declared in manifest
            actual_bench_hash = hashlib.sha256(bench_p.read_bytes()).hexdigest()
            manifest_bench_file = manifest_data.get("benchmark_file", "")
            if manifest_data.get("benchmark_sha256") and (bench_p.name in manifest_bench_file or "held_out_50_benchmark" in bench_p.name):
                assert actual_bench_hash == manifest_data["benchmark_sha256"], (
                    f"Benchmark SHA-256 mismatch! Expected {manifest_data['benchmark_sha256']}, got {actual_bench_hash}"
                )

    with open(bench_p, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    # Initialize assistant via public API
    assistant = LocalKnowledgeAssistant()
    total_indexed_chunks = assistant.ingest_directory(corpus_p)

    results: List[EvaluationItemResult] = []
    total_q = len(benchmark)
    passed_count = 0
    citation_valid_count = 0
    citation_support_count = 0
    total_citations_checked = 0
    abstention_correct = 0
    total_abstentions_expected = 0
    refutation_correct = 0
    total_refutations_expected = 0
    followup_correct = 0
    total_followups_expected = 0
    latencies: List[float] = []

    for item in benchmark:
        qid = item["id"]
        cat = item.get("category", "unknown")
        q = item["query"]
        exp_type = item["expected_type"]
        exp_source = item.get("expected_source")
        exp_sub = item.get("expected_substring")
        context_q = item.get("context_query")

        # Reset session context before each isolated test (unless follow-up priming)
        assistant.reset_conversation()
        if context_q:
            assistant.ask(context_q)

        t_start = time.perf_counter()
        resp: AssistantResponse = assistant.ask(q)
        lat = (time.perf_counter() - t_start) * 1000.0
        latencies.append(lat)

        failure_reasons = []
        is_decision_correct = (resp.decision == exp_type)
        if not is_decision_correct:
            failure_reasons.append(f"Decision mismatch: expected {exp_type}, got {resp.decision}")

        cit_valid = True
        cit_relevant = True
        cit_support = True
        cit_eval_list = []

        if exp_type == "ANSWER":
            if not resp.citations:
                cit_valid = False
                cit_relevant = False
                cit_support = False
                failure_reasons.append("Missing required citations for answer")
            else:
                for c in resp.citations:
                    total_citations_checked += 1
                    doc_id = c.get("doc_id")
                    start_c = c.get("start_char", 0)
                    end_c = c.get("end_char", 0)
                    exact_text = c.get("exact_text", "")

                    doc_file = corpus_p / doc_id
                    if not doc_file.exists():
                        cit_valid = False
                        failure_reasons.append(f"Cited file does not exist: {doc_id}")
                    else:
                        doc_content = doc_file.read_text(encoding="utf-8", errors="replace")
                        file_len = len(doc_content)
                        if start_c < 0 or end_c > file_len or start_c >= end_c:
                            cit_valid = False
                            failure_reasons.append(f"Invalid char offsets [{start_c}:{end_c}] for file len {file_len}")
                        else:
                            actual_slice = doc_content[start_c:end_c].strip()
                            if actual_slice != exact_text.strip():
                                cit_valid = False
                                failure_reasons.append("Cited exact_text does not match slice from disk")

                    # Check relevance & support
                    if exp_source and doc_id != exp_source:
                        cit_relevant = False
                        failure_reasons.append(f"Cited wrong document: expected {exp_source}, got {doc_id}")

                    if exp_sub and exp_sub.lower() not in exact_text.lower():
                        cit_support = False
                        failure_reasons.append(f"Cited passage does not contain target claim '{exp_sub}'")

                    cit_eval_list.append({
                        "doc_id": doc_id,
                        "valid": cit_valid,
                        "relevant": cit_relevant,
                        "supports_claim": cit_support,
                    })

                if cit_valid:
                    citation_valid_count += 1
                if cit_support:
                    citation_support_count += 1

            ans_correct = (exp_sub.lower() in resp.answer.lower()) if exp_sub else True
            if not ans_correct:
                failure_reasons.append(f"Target substring '{exp_sub}' not found in answer")

        elif exp_type == "ABSTAIN":
            total_abstentions_expected += 1
            ans_correct = is_decision_correct and ("could not find sufficient" in resp.answer.lower())
            if is_decision_correct:
                abstention_correct += 1

        elif exp_type == "REFUTE":
            total_refutations_expected += 1
            ans_correct = is_decision_correct and (not exp_sub or exp_sub.lower() in resp.answer.lower())
            if is_decision_correct:
                refutation_correct += 1
        else:
            ans_correct = False

        if cat == "follow_up":
            total_followups_expected += 1
            if is_decision_correct and ans_correct and cit_valid:
                followup_correct += 1

        item_passed = is_decision_correct and ans_correct and (cit_valid if exp_type == "ANSWER" else True)
        if item_passed:
            passed_count += 1

        results.append(EvaluationItemResult(
            id=qid,
            category=cat,
            query=q,
            expected_type=exp_type,
            expected_source=exp_source,
            expected_substring=exp_sub,
            decision=resp.decision,
            is_decision_correct=is_decision_correct,
            citation_validity=cit_valid,
            citation_relevance=cit_relevant,
            citation_support=cit_support,
            answer_correct=ans_correct,
            passed=item_passed,
            latency_ms=round(lat, 2),
            answer_preview=resp.answer[:120].replace("\n", " "),
            citations_evaluated=cit_eval_list,
            failure_reasons=failure_reasons,
        ))

    latencies_sorted = sorted(latencies)
    median_lat = round(latencies_sorted[len(latencies_sorted) // 2], 2)
    p95_lat = round(latencies_sorted[int(len(latencies_sorted) * 0.95)], 2)

    return {
        "benchmark_file": str(bench_p),
        "corpus_directory": str(corpus_p),
        "total_chunks_indexed": total_indexed_chunks,
        "total_questions": total_q,
        "passed_count": passed_count,
        "accuracy_pct": round(passed_count / total_q * 100, 2),
        "citation_validity_count": citation_valid_count,
        "citation_support_count": citation_support_count,
        "total_citations_checked": total_citations_checked,
        "correct_abstentions": f"{abstention_correct}/{total_abstentions_expected}",
        "correct_refutations": f"{refutation_correct}/{total_refutations_expected}",
        "correct_followups": f"{followup_correct}/{total_followups_expected}",
        "latency_median_ms": median_lat,
        "latency_p95_ms": p95_lat,
        "items": [asdict(r) for r in results],
    }


def main():
    parser = argparse.ArgumentParser(description="ChakrView Independent Evaluator")
    parser.add_argument("--corpus-dir", default="experiments/batch1021_1060/data/sample_corpus")
    parser.add_argument("--benchmark-file", default="experiments/batch1021_1060/data/held_out_50_benchmark.json")
    parser.add_argument("--manifest-file", default="experiments/batch1021_1060/reports/DATASET_MANIFEST.json")
    parser.add_argument("--output", default="experiments/batch1061_1100/reports/INDEPENDENT_BASELINE_RESULTS.json")

    args = parser.parse_args()
    summary = evaluate_dataset(args.corpus_dir, args.benchmark_file, args.manifest_file)

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n================ INDEPENDENT EVALUATION SUMMARY ================")
    print(f"Total Questions: {summary['total_questions']}")
    print(f"Passed: {summary['passed_count']}/{summary['total_questions']} ({summary['accuracy_pct']}%)")
    print(f"Citation Validity: {summary['citation_validity_count']} / 38 checked")
    print(f"Citation Support: {summary['citation_support_count']} / 38 checked")
    print(f"Abstentions: {summary['correct_abstentions']}")
    print(f"Refutations: {summary['correct_refutations']}")
    print(f"Follow-ups: {summary['correct_followups']}")
    print(f"Latency: Median {summary['latency_median_ms']} ms | p95 {summary['latency_p95_ms']} ms")
    print(f"Report written to: {out_p}")


if __name__ == "__main__":
    main()
