"""
Stage F: Comprehensive Baseline Evaluation of Canonical Checkpoint (Step 10,000)
Evaluates canonical checkpoint step_10000 on the 160 frozen held-out Exp18 test records.
Saves:
  - BASELINE_EVALUATION.json
"""

import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path("d:/Project/ChakrView").resolve()
MISSION_DIR = REPO_ROOT / "experiments" / "overnight_runs" / "mission_v08_overnight_20261023_232924"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.runtime.neural_runtime_adapter import NeuralRuntimeAdapter
from chakrview.evaluation.task_evaluator import TaskEvaluator, EvaluationOutcome, ScoringStatus
from experiments.research_v09_conversation_sft.exp18_scaled_instruction_format_v1.generate_exp18_datasets import Exp18DatasetGenerator

def evaluate_baseline():
    print("Initializing NeuralRuntimeAdapter on Canonical Step 10,000 Checkpoint...")
    adapter = NeuralRuntimeAdapter(
        checkpoint_path="step_10000",
        max_new_tokens=45,
        decoding="greedy",
    )
    adapter.initialize()
    print(f"Adapter initialized. Checkpoint: {adapter._checkpoint_label} ({adapter._checkpoint_sha256})")

    gen = Exp18DatasetGenerator()
    splits = gen.generate_all_splits()
    test_records = splits["test"]
    assert len(test_records) == 160, f"Expected 160 test records, got {len(test_records)}"

    evaluator = TaskEvaluator()

    results = []
    category_metrics = defaultdict(lambda: Counter())
    total_tokens_generated = 0
    total_time_ms = 0.0

    print("Beginning generation and evaluation across 160 test records...")
    for idx, rec in enumerate(test_records):
        prompt_mode = rec.get("prompt_mode", "plain_instruction")
        prompt = rec["prompt"]
        context = rec.get("context", "")

        t0 = time.perf_counter()
        gen_res = adapter.generate(
            prompt=prompt,
            prompt_mode=prompt_mode,
            context=context,
            max_new_tokens=45,
            decoding="greedy",
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        total_time_ms += elapsed_ms
        total_tokens_generated += gen_res["output_token_count"]

        eval_res = evaluator.evaluate_record(
            rec,
            gen_res["generated_text"],
            stop_reason=gen_res["stop_reason"],
        )

        category_metrics[rec["category"]]["total"] += 1
        category_metrics[rec["category"]][eval_res.outcome.value] += 1
        if eval_res.scoring_status == ScoringStatus.SCORED:
            category_metrics[rec["category"]]["scorable"] += 1
        elif eval_res.scoring_status == ScoringStatus.NEEDS_REVIEW:
            category_metrics[rec["category"]]["needs_review"] += 1

        if gen_res["stop_reason"] == "STOP_TOKEN":
            category_metrics[rec["category"]]["eos_completed"] += 1
        if eval_res.format_valid:
            category_metrics[rec["category"]]["valid_json"] += 1

        results.append({
            "record_id": rec["id"],
            "category": rec["category"],
            "prompt": prompt,
            "expected": rec["response"],
            "generated": gen_res["generated_text"],
            "stop_reason": gen_res["stop_reason"],
            "tokens_generated": gen_res["output_token_count"],
            "latency_ms": elapsed_ms,
            "outcome": eval_res.outcome.value,
            "scoring_status": eval_res.scoring_status.value,
            "reasons": eval_res.reason_codes,
            "format_valid": eval_res.format_valid,
        })

        if (idx + 1) % 40 == 0:
            print(f"  Processed {idx + 1} / 160 test records...")

    # Summary statistics
    total_scorable = sum(category_metrics[cat]["scorable"] for cat in category_metrics)
    total_correct = sum(category_metrics[cat][EvaluationOutcome.CORRECT.value] for cat in category_metrics)
    total_eos = sum(category_metrics[cat]["eos_completed"] for cat in category_metrics)
    
    json_recs = category_metrics["structured_formatting"]
    json_validity_rate = (json_recs["valid_json"] / json_recs["total"]) if json_recs["total"] > 0 else 0.0

    baseline_manifest = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": "step_10000",
        "checkpoint_sha256": adapter._checkpoint_sha256,
        "total_test_records": len(test_records),
        "total_scorable_records": total_scorable,
        "total_correct": total_correct,
        "scorable_task_accuracy": round((total_correct / total_scorable) * 100.0, 2) if total_scorable > 0 else 0.0,
        "total_eos_completions": total_eos,
        "eos_completion_rate": round((total_eos / len(test_records)) * 100.0, 2),
        "structured_formatting_json_validity_pct": round(json_validity_rate * 100.0, 2),
        "avg_latency_ms": round(total_time_ms / len(test_records), 2),
        "total_tokens_generated": total_tokens_generated,
        "category_metrics": {cat: dict(counts) for cat, counts in category_metrics.items()},
        "records_sample": results[:10],
    }

    with open(MISSION_DIR / "BASELINE_EVALUATION.json", "w", encoding="utf-8") as f:
        json.dump(baseline_manifest, f, indent=2)

    print("\n=== BASELINE EVALUATION REPORT ===")
    print(f"Scorable Records: {total_scorable} (140 scorable, 20 short_explanation in needs_review)")
    print(f"Scorable Task Accuracy: {baseline_manifest['scorable_task_accuracy']}% ({total_correct}/{total_scorable})")
    print(f"EOS Completion Rate: {baseline_manifest['eos_completion_rate']}% ({total_eos}/{len(test_records)})")
    print(f"Structured JSON Validity: {baseline_manifest['structured_formatting_json_validity_pct']}%")
    print(f"Average CPU Inference Latency: {baseline_manifest['avg_latency_ms']} ms/record")

if __name__ == "__main__":
    evaluate_baseline()
