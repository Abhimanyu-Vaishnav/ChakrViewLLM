"""
Exp18 Aligned SFT Experiment Execution (300 Steps on CPU).
Trains exclusively on an isolated copy of canonical step 10,000 checkpoint.
Uses in-memory generated Exp18 dataset splits (isolated JSONL files in experiment folder).
"""

import csv
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
import torch

REPO_ROOT = Path("d:/Project/ChakrView").resolve()
EXP_DIR = REPO_ROOT / "experiments" / "overnight_runs" / "exp18_aligned_sft_cpu_300"
DATA_DIR = EXP_DIR / "data"
CKPT_OUTPUT_DIR = EXP_DIR / "checkpoints"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CKPT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.training.sft_engine import SFTConfig, SFTExperimentEngine
from experiments.research_v09_conversation_sft.exp18_scaled_instruction_format_v1.generate_exp18_datasets import Exp18DatasetGenerator
from chakrview.runtime.neural_inference_contract import NeuralInferenceContract
from chakrview.runtime.sampling import Sampler, SamplingConfig
from chakrview.runtime.prompt_conditioning import format_conditioned_prompt
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.evaluation.task_evaluator import TaskEvaluator, EvaluationOutcome, ScoringStatus

def main():
    print(f"=== Starting ChakrView Exp18 Aligned SFT (300 Steps CPU) ===")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Output directory: {EXP_DIR}")

    # 1. Prepare in-memory dataset splits
    gen = Exp18DatasetGenerator()
    splits = gen.generate_all_splits()
    
    train_file = DATA_DIR / "exp18_train_isolated.jsonl"
    val_file = DATA_DIR / "exp18_val_isolated.jsonl"
    
    with open(train_file, "w", encoding="utf-8") as f:
        for r in splits["train"]:
            f.write(json.dumps(r) + "\n")
            
    with open(val_file, "w", encoding="utf-8") as f:
        for r in splits["val"]:
            f.write(json.dumps(r) + "\n")
            
    base_checkpoint = CKPT_OUTPUT_DIR / "training_checkpoint_start.pt"
    assert base_checkpoint.exists(), f"Starting checkpoint missing: {base_checkpoint}"

    # 2. Configure training
    config = SFTConfig(
        repo_root=REPO_ROOT,
        train_file=train_file,
        val_file=val_file,
        output_dir=CKPT_OUTPUT_DIR,
        base_checkpoint=base_checkpoint,
        steps=300,
        batch_size=2,
        max_seq_len=512,
        learning_rate=5e-5,
        min_learning_rate=5e-6,
        weight_decay=0.01,
        grad_clip=1.0,
        val_interval=20,
        save_interval=50,
        seed=42,
        device="cpu",
        tokenizer_dir=str(REPO_ROOT / "data" / "experiments" / "vocab_4096"),
        allow_exploratory_opt_in=True,
        enforce_trust_gate=True,
        strip_prompt_delimiters=False,
        save_best_checkpoint=True,
    )

    training_config_payload = {
        "experiment_id": "exp18_aligned_sft_cpu_300",
        "timestamp": datetime.now().isoformat(),
        "base_checkpoint": str(base_checkpoint),
        "steps": config.steps,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "min_learning_rate": config.min_learning_rate,
        "weight_decay": config.weight_decay,
        "grad_clip": config.grad_clip,
        "val_interval": config.val_interval,
        "save_interval": config.save_interval,
        "seed": config.seed,
        "device": config.device,
        "tokenizer_dir": config.tokenizer_dir,
    }
    with open(EXP_DIR / "TRAINING_CONFIG.json", "w", encoding="utf-8") as f:
        json.dump(training_config_payload, f, indent=2)

    # 3. Train
    print("Initializing SFTExperimentEngine...")
    engine = SFTExperimentEngine(config)
    t0 = time.perf_counter()
    res = engine.train()
    train_duration = time.perf_counter() - t0
    print(f"Training completed in {train_duration:.2f} seconds ({train_duration/60:.2f} minutes).")

    # 4. Save training log
    with open(EXP_DIR / "TRAINING_LOG.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "train_loss", "active_tokens", "val_loss", "val_perplexity", "learning_rate", "grad_norm", "step_time_sec", "is_best"])
        for h in res["history"]:
            writer.writerow([
                h.get("step"),
                h.get("train_loss"),
                h.get("active_tokens", ""),
                h.get("val_loss", ""),
                h.get("val_perplexity", ""),
                h.get("learning_rate"),
                h.get("grad_norm"),
                h.get("step_time_sec"),
                h.get("is_best", False),
            ])

    # 5. Evaluate best checkpoint on held-out 160 test records
    best_ckpt_path = CKPT_OUTPUT_DIR / "sft_best_checkpoint.pt"
    if not best_ckpt_path.exists():
        best_ckpt_path = CKPT_OUTPUT_DIR / f"sft_checkpoint_step_{config.steps}.pt"
    print(f"Evaluating best trained checkpoint: {best_ckpt_path} on held-out test split...")

    tokenizer, _ = load_tokenizer_artifacts(config.tokenizer_dir)
    contract = NeuralInferenceContract.from_checkpoint(
        checkpoint_path=best_ckpt_path,
        tokenizer=tokenizer,
        config=ModelConfig(),
        device=torch.device("cpu"),
    )
    sampler = Sampler()
    test_records = splits["test"]
    assert len(test_records) == 160
    evaluator = TaskEvaluator()

    post_results = []
    category_metrics = {}
    total_tokens_generated = 0
    total_eval_time_ms = 0.0

    from collections import Counter, defaultdict
    cat_counts = defaultdict(lambda: Counter())

    for idx, rec in enumerate(test_records):
        prompt_mode = rec.get("prompt_mode", "plain_instruction")
        prompt = rec["prompt"]
        context = rec.get("context", "")

        t_start = time.perf_counter()
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
                if next_token == 1:
                    stop_reason = "STOP_TOKEN"
                    break

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        total_eval_time_ms += elapsed_ms
        total_tokens_generated += len(generated_tokens)

        gen_text = contract.decode(generated_tokens)
        eval_res = evaluator.evaluate_record(
            rec,
            gen_text,
            stop_reason=stop_reason,
        )

        cat = rec["category"]
        cat_counts[cat]["total"] += 1
        cat_counts[cat][eval_res.outcome.value] += 1
        if eval_res.scoring_status == ScoringStatus.SCORED:
            cat_counts[cat]["scorable"] += 1
        elif eval_res.scoring_status == ScoringStatus.NEEDS_REVIEW:
            cat_counts[cat]["needs_review"] += 1

        if stop_reason == "STOP_TOKEN":
            cat_counts[cat]["eos_completed"] += 1
        if eval_res.format_valid:
            cat_counts[cat]["valid_json"] += 1

        post_results.append({
            "record_id": rec["id"],
            "category": cat,
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

    total_scorable = sum(cat_counts[c]["scorable"] for c in cat_counts)
    total_correct = sum(cat_counts[c][EvaluationOutcome.CORRECT.value] for c in cat_counts)
    total_eos = sum(cat_counts[c]["eos_completed"] for c in cat_counts)

    json_recs = cat_counts["structured_formatting"]
    json_validity_rate = (json_recs["valid_json"] / json_recs["total"]) if json_recs["total"] > 0 else 0.0

    sft_scorable_acc = round((total_correct / total_scorable) * 100.0, 2) if total_scorable > 0 else 0.0
    sft_eos_rate = round((total_eos / len(test_records)) * 100.0, 2)
    sft_json_validity = round(json_validity_rate * 100.0, 2)
    avg_latency = round(total_eval_time_ms / len(test_records), 2)

    # 6. Load preflight baseline for comparative reconciliation
    baseline_path = REPO_ROOT / "experiments" / "overnight_runs" / "preflight_exp18_20261024_114600" / "BASELINE_EVALUATION.json"
    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    comparison = {
        "timestamp": datetime.now().isoformat(),
        "experiment_id": "exp18_aligned_sft_cpu_300",
        "training": {
            "steps": config.steps,
            "initial_val_loss": res["initial_val_loss"],
            "final_train_loss": res["final_train_loss"],
            "final_val_loss": res["final_val_loss"],
            "best_val_loss": res["best_val_loss"],
            "best_val_step": res["best_val_step"],
            "elapsed_seconds": res["elapsed_seconds"],
        },
        "baseline": {
            "checkpoint": baseline["checkpoint"],
            "scorable_task_accuracy": baseline["scorable_task_accuracy"],
            "correct_scorable_count": f"{baseline['total_correct']}/{baseline['total_scorable_records']}",
            "eos_completion_rate": baseline["eos_completion_rate"],
            "structured_formatting_json_validity_pct": baseline["structured_formatting_json_validity_pct"],
            "avg_latency_ms": baseline["avg_latency_ms"],
        },
        "sft_experiment": {
            "checkpoint": str(best_ckpt_path),
            "scorable_task_accuracy": sft_scorable_acc,
            "correct_scorable_count": f"{total_correct}/{total_scorable}",
            "eos_completion_rate": sft_eos_rate,
            "structured_formatting_json_validity_pct": sft_json_validity,
            "avg_latency_ms": avg_latency,
        },
        "delta": {
            "scorable_task_accuracy_delta": round(sft_scorable_acc - baseline["scorable_task_accuracy"], 2),
            "eos_completion_rate_delta": round(sft_eos_rate - baseline["eos_completion_rate"], 2),
            "structured_formatting_json_validity_delta": round(sft_json_validity - baseline["structured_formatting_json_validity_pct"], 2),
            "avg_latency_delta_ms": round(avg_latency - baseline["avg_latency_ms"], 2),
        },
        "category_metrics": {c: dict(cat_counts[c]) for c in cat_counts},
        "sample_records": post_results[:10],
    }

    with open(EXP_DIR / "EXPERIMENT_COMPARISON.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    with open(EXP_DIR / "POST_TRAINING_EVALUATION.json", "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "checkpoint": str(best_ckpt_path),
            "total_test_records": len(test_records),
            "total_scorable_records": total_scorable,
            "total_correct": total_correct,
            "scorable_task_accuracy": sft_scorable_acc,
            "total_eos_completions": total_eos,
            "eos_completion_rate": sft_eos_rate,
            "structured_formatting_json_validity_pct": sft_json_validity,
            "avg_latency_ms": avg_latency,
            "category_metrics": {c: dict(cat_counts[c]) for c in cat_counts},
            "all_records": post_results,
        }, f, indent=2)

    print("\n=== EXPERIMENT COMPARISON SUMMARY ===")
    print(f"Scorable Task Accuracy: Baseline {baseline['scorable_task_accuracy']}% -> SFT {sft_scorable_acc}% (Delta: {comparison['delta']['scorable_task_accuracy_delta']}%)")
    print(f"EOS Completion Rate:    Baseline {baseline['eos_completion_rate']}% -> SFT {sft_eos_rate}% (Delta: {comparison['delta']['eos_completion_rate_delta']}%)")
    print(f"Structured JSON Validity: Baseline {baseline['structured_formatting_json_validity_pct']}% -> SFT {sft_json_validity}% (Delta: {comparison['delta']['structured_formatting_json_validity_delta']}%)")
    print("Done!")

if __name__ == "__main__":
    main()
