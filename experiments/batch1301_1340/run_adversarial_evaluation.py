"""
Batch 1301–1340 Adversarial and Frozen Benchmark Evaluation Runner.
Location: experiments/batch1301_1340/run_adversarial_evaluation.py

Evaluates:
1. Frozen 70-Item Benchmark (batch 1141_1180 data)
2. Adversarial 100-Item Decision Benchmark (batch 1301_1340 data)
3. Computes:
   - Decision Accuracy (overall & per-category)
   - False-Refutation Rate
   - Missed-Refutation Rate
   - Unsupported-Answer Rate
   - Abstention Precision and Recall
   - Citation & Evidence Validity
   - Retrieval Relevance vs Decision Correctness

Generates machine-readable output under experiments/batch1301_1340/results/ADVERSARIAL_EVALUATION_RESULTS.json
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.runtime.local_knowledge_assistant import LocalKnowledgeAssistant

CORPUS_DIR = Path("experiments/batch1141_1180/data/corpus")
FROZEN_70_PATH = Path("experiments/batch1141_1180/data/held_out_70_benchmark.json")
ADVERSARIAL_100_PATH = Path("experiments/batch1301_1340/data/adversarial_decision_100.json")
OUT_DIR = Path("experiments/batch1301_1340/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_benchmark(asst: LocalKnowledgeAssistant, benchmark_path: Path, benchmark_name: str) -> dict:
    items = json.loads(benchmark_path.read_text(encoding="utf-8"))
    
    total = len(items)
    correct_decisions = 0
    cat_stats = {}
    
    # Detailed metrics
    # Decision confusion:
    # Expected vs Actual
    # Labels: ANSWER, REFUTE, ABSTAIN
    confusion = {
        "ANSWER": {"ANSWER": 0, "REFUTE": 0, "ABSTAIN": 0},
        "REFUTE": {"ANSWER": 0, "REFUTE": 0, "ABSTAIN": 0},
        "ABSTAIN": {"ANSWER": 0, "REFUTE": 0, "ABSTAIN": 0},
    }
    
    citation_valid = 0
    citation_total = 0
    per_item_results = []
    latencies = []
    
    for item in items:
        cat = item.get("category", "uncategorized")
        if cat not in cat_stats:
            cat_stats[cat] = {"total": 0, "correct": 0}
        cat_stats[cat]["total"] += 1
        
        asst.reset_conversation()
        if item.get("context_query"):
            asst.ask(item["context_query"])
            
        t0 = time.perf_counter()
        resp = asst.ask(item["query"])
        latency_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(latency_ms)
        
        expected_type = item.get("expected_decision") or item.get("expected_type")
        actual_type = resp.decision
        
        is_correct = (actual_type == expected_type)
        if is_correct:
            correct_decisions += 1
            cat_stats[cat]["correct"] += 1
            
        if expected_type in confusion and actual_type in confusion[expected_type]:
            confusion[expected_type][actual_type] += 1
            
        # Check citation validity for ANSWER and REFUTE
        cites_valid = False
        if actual_type in ("ANSWER", "REFUTE"):
            citation_total += 1
            # Must have non-empty citations and cite an existing doc
            valid_corpus_names = [p.name for p in CORPUS_DIR.glob("*.md")]
            def extract_doc_name(c):
                if isinstance(c, dict):
                    return Path(c.get("doc_id", c.get("path", ""))).name
                return Path(str(c)).name

            if resp.citations and all(extract_doc_name(c) in valid_corpus_names for c in resp.citations):
                citation_valid += 1
                cites_valid = True
        else:
            cites_valid = True  # Not applicable for abstain
            
        per_item_results.append({
            "id": item["id"],
            "category": cat,
            "query": item["query"],
            "expected": expected_type,
            "actual": actual_type,
            "is_correct": is_correct,
            "citations": resp.citations,
            "cites_valid": cites_valid,
            "evidence_span": resp.citations[0].get("evidence_span", "")[:100] if resp.citations and isinstance(resp.citations[0], dict) else None,
            "answer_preview": resp.answer[:120],
            "latency_ms": round(latency_ms, 2)
        })

    # Metric computations:
    # 1. False-Refutation Rate: Expected ANSWER or ABSTAIN, but got REFUTE
    non_refute_expected = confusion["ANSWER"]["REFUTE"] + confusion["ABSTAIN"]["REFUTE"]
    total_non_refute = (sum(confusion["ANSWER"].values()) + sum(confusion["ABSTAIN"].values()))
    false_refutation_rate = non_refute_expected / total_non_refute if total_non_refute > 0 else 0.0

    # 2. Missed-Refutation Rate: Expected REFUTE, but got ANSWER or ABSTAIN
    total_refute_expected = sum(confusion["REFUTE"].values())
    missed_refutations = confusion["REFUTE"]["ANSWER"] + confusion["REFUTE"]["ABSTAIN"]
    missed_refutation_rate = missed_refutations / total_refute_expected if total_refute_expected > 0 else 0.0

    # 3. Unsupported-Answer Rate: Expected ABSTAIN or REFUTE, but got ANSWER
    unsupported_answers = confusion["ABSTAIN"]["ANSWER"] + confusion["REFUTE"]["ANSWER"]
    total_unsupported_candidate = sum(confusion["ABSTAIN"].values()) + sum(confusion["REFUTE"].values())
    unsupported_answer_rate = unsupported_answers / total_unsupported_candidate if total_unsupported_candidate > 0 else 0.0

    # 4. Abstention Precision and Recall
    total_abstain_predicted = confusion["ANSWER"]["ABSTAIN"] + confusion["REFUTE"]["ABSTAIN"] + confusion["ABSTAIN"]["ABSTAIN"]
    total_abstain_expected = sum(confusion["ABSTAIN"].values())
    abstain_precision = confusion["ABSTAIN"]["ABSTAIN"] / total_abstain_predicted if total_abstain_predicted > 0 else 0.0
    abstain_recall = confusion["ABSTAIN"]["ABSTAIN"] / total_abstain_expected if total_abstain_expected > 0 else 0.0

    citation_validity_rate = citation_valid / citation_total if citation_total > 0 else 1.0

    print(f"\n==========================================")
    print(f"Results for Benchmark: {benchmark_name} (N={total})")
    print(f"==========================================")
    print(f"Overall Decision Accuracy: {correct_decisions}/{total} ({correct_decisions/total*100:.2f}%)")
    print(f"False-Refutation Rate: {non_refute_expected}/{total_non_refute} ({false_refutation_rate*100:.2f}%)")
    print(f"Missed-Refutation Rate: {missed_refutations}/{total_refute_expected} ({missed_refutation_rate*100:.2f}%)")
    print(f"Unsupported-Answer Rate: {unsupported_answers}/{total_unsupported_candidate} ({unsupported_answer_rate*100:.2f}%)")
    print(f"Abstention Precision: {abstain_precision*100:.2f}% | Recall: {abstain_recall*100:.2f}%")
    print(f"Citation Validity Rate: {citation_valid}/{citation_total} ({citation_validity_rate*100:.2f}%)")
    print(f"Median Latency: {statistics.median(latencies):.2f} ms")
    print("\nConfusion Matrix (Expected \\ Actual):")
    for exp in ["ANSWER", "REFUTE", "ABSTAIN"]:
        print(f"  Exp {exp:7s} -> ANS: {confusion[exp]['ANSWER']:2d}, REF: {confusion[exp]['REFUTE']:2d}, ABS: {confusion[exp]['ABSTAIN']:2d}")

    print("\nCategory Breakdown:")
    for cat, v in sorted(cat_stats.items()):
        acc = v["correct"] / v["total"] * 100.0 if v["total"] > 0 else 0.0
        print(f"  {cat:32s}: {v['correct']:2d}/{v['total']:2d} ({acc:5.1f}%)")

    return {
        "benchmark_name": benchmark_name,
        "total_queries": total,
        "correct_decisions": correct_decisions,
        "accuracy": round(correct_decisions / total, 4),
        "false_refutation_rate": round(false_refutation_rate, 4),
        "false_refutations_count": non_refute_expected,
        "total_non_refute_expected": total_non_refute,
        "missed_refutation_rate": round(missed_refutation_rate, 4),
        "missed_refutations_count": missed_refutations,
        "total_refute_expected": total_refute_expected,
        "unsupported_answer_rate": round(unsupported_answer_rate, 4),
        "unsupported_answers_count": unsupported_answers,
        "total_unanswerable_expected": total_unsupported_candidate,
        "abstention_precision": round(abstain_precision, 4),
        "abstention_recall": round(abstain_recall, 4),
        "citation_validity_rate": round(citation_validity_rate, 4),
        "citation_valid_count": citation_valid,
        "citation_total_count": citation_total,
        "median_latency_ms": round(statistics.median(latencies), 2),
        "confusion_matrix": confusion,
        "category_breakdown": cat_stats,
        "per_item_results": per_item_results
    }


def main():
    # Build Assistant with BM25
    asst = LocalKnowledgeAssistant(
        storage_dir=None,
        chunk_size=400,
        chunk_overlap=100,
        embedding_provider=None,
        extract_concise_span=True
    )
    for f in sorted(CORPUS_DIR.glob("*.md")):
        asst.ingest_document(f)

    # 1. Evaluate Frozen 70 Benchmark
    frozen_res = evaluate_benchmark(asst, FROZEN_70_PATH, "Frozen_70_Benchmark")

    # 2. Evaluate Adversarial 100 Benchmark
    adv_res = evaluate_benchmark(asst, ADVERSARIAL_100_PATH, "Adversarial_100_Benchmark")

    combined = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evaluator": "experiments/batch1301_1340/run_adversarial_evaluation.py",
        "frozen_70_benchmark": frozen_res,
        "adversarial_100_benchmark": adv_res
    }

    out_file = OUT_DIR / "ADVERSARIAL_EVALUATION_RESULTS.json"
    out_file.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(f"\nSaved combined benchmark results to: {out_file}")


if __name__ == "__main__":
    main()
