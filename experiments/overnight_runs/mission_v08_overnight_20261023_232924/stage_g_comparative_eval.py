"""
Stage G.4: Comparative Evaluation of SFT Best Checkpoint vs Canonical Baseline.
Evaluates sft_best_checkpoint.pt on the EXACT identical 160 frozen held-out Exp18 test records.
Produces:
  - EXPERIMENT_COMPARISON.json
"""

import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import torch

REPO_ROOT = Path("d:/Project/ChakrView").resolve()
MISSION_DIR = REPO_ROOT / "experiments" / "overnight_runs" / "mission_v08_overnight_20261023_232924"
BEST_CKPT_PATH = MISSION_DIR / "checkpoints" / "sft_exp18_run" / "sft_best_checkpoint.pt"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.runtime.neural_inference_contract import NeuralInferenceContract
from chakrview.runtime.sampling import Sampler, SamplingConfig
from chakrview.runtime.prompt_conditioning import format_conditioned_prompt
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.evaluation.task_evaluator import TaskEvaluator, EvaluationOutcome, ScoringStatus
from experiments.research_v09_conversation_sft.exp18_scaled_instruction_format_v1.generate_exp18_datasets import Exp18DatasetGenerator

def run_comparative_eval():
    print(f"Loading SFT Best Checkpoint: {BEST_CKPT_PATH}...")
    tokenizer_dir = REPO_ROOT / "data" / "experiments" / "vocab_4096"
    tokenizer, _ = load_tokenizer_artifacts(str(tokenizer_dir))

    contract = NeuralInferenceContract.from_checkpoint(
        checkpoint_path=BEST_CKPT_PATH,
        tokenizer=tokenizer,
        config=ModelConfig(),
        device=torch.device("cpu"),
    )
    sampler = Sampler()

    # Load frozen test set
    gen = Exp18DatasetGenerator()
    splits = gen.generate_all_splits()
    test_records = splits["test"]
    assert len(test_records) == 160

    evaluator = TaskEvaluator()

    results = []
    category_metrics = defaultdict(lambda: Counter())
    total_tokens_generated = 0
    total_time_ms = 0.0

    print("Evaluating SFT Best Checkpoint across 160 test records...")
    for idx, rec in enumerate(test_records):
        prompt_mode = rec.get("prompt_mode", "plain_instruction")
        prompt = rec["prompt"]
        context = rec.get("context", "")

        t0 = time.perf_counter()
        formatted_prompt = format_conditioned_prompt(prompt, prompt_mode=prompt_mode, context=context)
        prompt_tokens = contract.encode(formatted_prompt, add_bos=True, add_eos=False)

        generated_tokens = []
        curr_tokens = list(prompt_tokens)
        stop_reason = "MAX_TOKENS"
        sampling_cfg = SamplingConfig(temperature=0.0, seed=42)

        with torch.no_grad():
            for step in range(45):
                tensor_in = torch.tensor([curr_tokens], dtype=torch.long, device=contract.device)
                logits = contract.model(tensor_in)
                next_token_logits = logits[0, -1, :]
                next_token = sampler.sample(
                    next_token_logits,
                    generated_tokens=generated_tokens,
                    config=sampling_cfg,
                    step=step,
                )
                generated_tokens.append(next_token)
                curr_tokens.append(next_token)
                if next_token == 1:  # EOS_ID
                    stop_reason = "STOP_TOKEN"
                    break

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        total_time_ms += elapsed_ms
        total_tokens_generated += len(generated_tokens)

        gen_text = contract.decode(generated_tokens)
        eval_res = evaluator.evaluate_record(
            rec,
            gen_text,
            stop_reason=stop_reason,
        )

        category_metrics[rec["category"]]["total"] += 1
        category_metrics[rec["category"]][eval_res.outcome.value] += 1
        if eval_res.scoring_status == ScoringStatus.SCORED:
            category_metrics[rec["category"]]["scorable"] += 1
        elif eval_res.scoring_status == ScoringStatus.NEEDS_REVIEW:
            category_metrics[rec["category"]]["needs_review"] += 1

        if stop_reason == "STOP_TOKEN":
            category_metrics[rec["category"]]["eos_completed"] += 1
        if eval_res.format_valid:
            category_metrics[rec["category"]]["valid_json"] += 1

        results.append({
            "record_id": rec["id"],
            "category": rec["category"],
            "prompt": prompt,
            "expected": rec["response"],
            "generated": gen_text,
            "stop_reason": stop_reason,
            "tokens_generated": len(generated_tokens),
            "latency_ms": elapsed_ms,
            "outcome": eval_res.outcome.value,
            "scoring_status": eval_res.scoring_status.value,
            "reasons": eval_res.reason_codes,
            "format_valid": eval_res.format_valid,
        })

        if (idx + 1) % 40 == 0:
            print(f"  Processed {idx + 1} / 160 test records...")

    # Load baseline evaluation for side-by-side comparison
    with open(MISSION_DIR / "BASELINE_EVALUATION.json", "r", encoding="utf-8") as f:
        baseline = json.load(f)

    total_scorable = sum(category_metrics[cat]["scorable"] for cat in category_metrics)
    total_correct = sum(category_metrics[cat][EvaluationOutcome.CORRECT.value] for cat in category_metrics)
    total_eos = sum(category_metrics[cat]["eos_completed"] for cat in category_metrics)
    
    json_recs = category_metrics["structured_formatting"]
    json_validity_rate = (json_recs["valid_json"] / json_recs["total"]) if json_recs["total"] > 0 else 0.0

    sft_scorable_acc = round((total_correct / total_scorable) * 100.0, 2) if total_scorable > 0 else 0.0
    sft_eos_rate = round((total_eos / len(test_records)) * 100.0, 2)
    sft_json_validity = round(json_validity_rate * 100.0, 2)

    comparison = {
        "timestamp": datetime.now().isoformat(),
        "evaluation_protocol": "Frozen 160-item Exp18 held-out test split, greedy decoding, max_new_tokens=45",
        "baseline": {
            "checkpoint": baseline["checkpoint"],
            "scorable_task_accuracy": baseline["scorable_task_accuracy"],
            "correct_scorable_count": f"{baseline['total_correct']}/{baseline['total_scorable_records']}",
            "eos_completion_rate": baseline["eos_completion_rate"],
            "structured_formatting_json_validity_pct": baseline["structured_formatting_json_validity_pct"],
            "avg_latency_ms": baseline["avg_latency_ms"],
        },
        "sft_experiment": {
            "checkpoint": str(BEST_CKPT_PATH),
            "scorable_task_accuracy": sft_scorable_acc,
            "correct_scorable_count": f"{total_correct}/{total_scorable}",
            "eos_completion_rate": sft_eos_rate,
            "structured_formatting_json_validity_pct": sft_json_validity,
            "avg_latency_ms": round(total_time_ms / len(test_records), 2),
        },
        "delta": {
            "scorable_task_accuracy_delta": round(sft_scorable_acc - baseline["scorable_task_accuracy"], 2),
            "eos_completion_rate_delta": round(sft_eos_rate - baseline["eos_completion_rate"], 2),
            "structured_formatting_json_validity_delta": round(sft_json_validity - baseline["structured_formatting_json_validity_pct"], 2),
        },
        "category_breakdown": {
            cat: {
                "baseline_correct": baseline["category_metrics"].get(cat, {}).get("CORRECT", 0),
                "sft_correct": category_metrics[cat][EvaluationOutcome.CORRECT.value],
                "total": category_metrics[cat]["total"],
                "sft_eos": category_metrics[cat]["eos_completed"],
            }
            for cat in category_metrics
        },
        "records_sample": results[:10],
    }

    with open(MISSION_DIR / "EXPERIMENT_COMPARISON.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print("\n=== EXPERIMENTAL COMPARISON RESULTS ===")
    print(f"Scorable Task Accuracy: Baseline {baseline['scorable_task_accuracy']}% -> SFT {sft_scorable_acc}% (Delta: {comparison['delta']['scorable_task_accuracy_delta']}%)")
    print(f"EOS Completion Rate:    Baseline {baseline['eos_completion_rate']}% -> SFT {sft_eos_rate}% (Delta: {comparison['delta']['eos_completion_rate_delta']}%)")
    print(f"Structured JSON Validity: Baseline {baseline['structured_formatting_json_validity_pct']}% -> SFT {sft_json_validity}% (Delta: {comparison['delta']['structured_formatting_json_validity_delta']}%)")

if __name__ == "__main__":
    run_comparative_eval()
