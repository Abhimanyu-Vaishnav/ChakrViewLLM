"""
Exp19 Debiased SFT Training & Evaluation Execution Script (600 Steps on CPU).
"""

import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
import torch

REPO_ROOT = Path("d:/Project/ChakrView").resolve()
EXP_DIR = REPO_ROOT / "experiments" / "overnight_runs" / "exp19_debiased_sft_cpu_600_20261024_125700"
DATA_DIR = EXP_DIR / "data"
CKPT_DIR = EXP_DIR / "checkpoints"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.training.sft_engine import SFTConfig, SFTExperimentEngine
from chakrview.runtime.neural_inference_contract import NeuralInferenceContract
from chakrview.runtime.sampling import Sampler, SamplingConfig
from chakrview.runtime.prompt_conditioning import format_conditioned_prompt
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.evaluation.task_evaluator import TaskEvaluator, EvaluationOutcome, ScoringStatus

def main():
    print("=== Launching Authorized Exp19 Debiased SFT (600 Steps CPU) ===")
    start_iso = datetime.now().isoformat()
    print(f"Start Timestamp: {start_iso}")
    print(f"Output Directory: {EXP_DIR}")

    train_file = DATA_DIR / "exp19_train_debiased_isolated.jsonl"
    val_file = DATA_DIR / "exp19_val_isolated.jsonl"
    base_checkpoint = CKPT_DIR / "training_checkpoint_start.pt"

    assert train_file.exists(), f"Train file missing: {train_file}"
    assert val_file.exists(), f"Val file missing: {val_file}"
    assert base_checkpoint.exists(), f"Base checkpoint missing: {base_checkpoint}"

    config = SFTConfig(
        repo_root=REPO_ROOT,
        train_file=train_file,
        val_file=val_file,
        output_dir=CKPT_DIR,
        base_checkpoint=base_checkpoint,
        steps=600,
        batch_size=2,
        max_seq_len=512,
        learning_rate=5e-5,
        min_learning_rate=5e-6,
        weight_decay=0.01,
        grad_clip=1.0,
        val_interval=20,
        save_interval=20,
        seed=42,
        device="cpu",
        tokenizer_dir=str(REPO_ROOT / "data" / "experiments" / "vocab_4096"),
        allow_exploratory_opt_in=True,
        enforce_trust_gate=True,
        strip_prompt_delimiters=False,
        save_best_checkpoint=True,
    )

    print("Initializing SFTExperimentEngine...")
    engine = SFTExperimentEngine(config)

    t0 = time.perf_counter()
    train_results = engine.train()
    elapsed_total = time.perf_counter() - t0
    print(f"Training completed in {elapsed_total:.2f} seconds ({elapsed_total/60:.2f} minutes).")

    # Write TRAINING_LOG.jsonl
    with open(EXP_DIR / "TRAINING_LOG.jsonl", "w", encoding="utf-8") as f:
        for entry in train_results["history"]:
            f.write(json.dumps(entry) + "\n")
    print(f"Wrote {len(train_results['history'])} log entries to TRAINING_LOG.jsonl")

    # Evaluate best checkpoint
    best_ckpt = CKPT_DIR / "sft_best_checkpoint.pt"
    if not best_ckpt.exists():
        best_ckpt = CKPT_DIR / f"sft_checkpoint_step_{config.steps}.pt"
    print(f"Evaluating best checkpoint: {best_ckpt} (best_val_step={train_results.get('best_val_step')})...")

    # Load frozen test set
    test_file = DATA_DIR / "exp19_test_frozen_isolated.jsonl"
    with open(test_file, "r", encoding="utf-8") as f:
        test_records = [json.loads(line) for line in f if line.strip()]
    assert len(test_records) == 160

    tokenizer, _ = load_tokenizer_artifacts(config.tokenizer_dir)
    contract = NeuralInferenceContract.from_checkpoint(
        checkpoint_path=best_ckpt,
        tokenizer=tokenizer,
        config=ModelConfig(),
        device=torch.device("cpu"),
    )
    sampler = Sampler()
    evaluator = TaskEvaluator()

    eval_records = []
    total_tokens_gen = 0
    total_latency_ms = 0.0

    from collections import Counter, defaultdict
    cat_counts = defaultdict(lambda: Counter())

    print("Evaluating across 160 frozen test records...")
    for idx, rec in enumerate(test_records):
        p_mode = rec.get("prompt_mode", "plain_instruction")
        prompt = rec["prompt"]
        context = rec.get("context", "")

        t_start = time.perf_counter()
        f_prompt = format_conditioned_prompt(prompt, prompt_mode=p_mode, context=context)
        p_tokens = contract.encode(f_prompt, add_bos=True, add_eos=False)

        gen_tokens = []
        curr = list(p_tokens)
        stop_reason = "MAX_TOKENS"
        s_cfg = SamplingConfig(temperature=0.0, seed=42)

        with torch.no_grad():
            for step in range(45):
                t_in = torch.tensor([curr], dtype=torch.long, device=contract.device)
                logits = contract.model(t_in)
                next_tok_logits = logits[0, -1, :]
                next_tok = sampler.sample(
                    next_tok_logits,
                    generated_tokens=gen_tokens,
                    config=s_cfg,
                    step=step,
                )
                gen_tokens.append(next_tok)
                curr.append(next_tok)
                if next_tok == 1:
                    stop_reason = "STOP_TOKEN"
                    break

        lat_ms = (time.perf_counter() - t_start) * 1000.0
        total_latency_ms += lat_ms
        total_tokens_gen += len(gen_tokens)

        gen_str = contract.decode(gen_tokens)
        eval_res = evaluator.evaluate_record(
            rec,
            gen_str,
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

        eval_records.append({
            "record_id": rec["id"],
            "category": cat,
            "prompt": prompt,
            "expected": rec["response"],
            "generated": gen_str,
            "stop_reason": stop_reason,
            "tokens_generated": len(gen_tokens),
            "latency_ms": lat_ms,
            "outcome": eval_res.outcome.value,
            "scoring_status": eval_res.scoring_status.value,
            "reasons": eval_res.reason_codes,
            "format_valid": eval_res.format_valid,
        })

    # Save per-record output
    with open(EXP_DIR / "EVAL_EXP19_PER_RECORD.jsonl", "w", encoding="utf-8") as f:
        for r in eval_records:
            f.write(json.dumps(r) + "\n")

    # Aggregate metrics
    tot_scorable = sum(cat_counts[c]["scorable"] for c in cat_counts)
    tot_correct = sum(cat_counts[c][EvaluationOutcome.CORRECT.value] for c in cat_counts)
    tot_eos = sum(cat_counts[c]["eos_completed"] for c in cat_counts)

    json_recs = cat_counts["structured_formatting"]
    json_validity_pct = round((json_recs["valid_json"] / json_recs["total"] * 100.0), 2) if json_recs["total"] > 0 else 0.0

    task_acc_pct = round((tot_correct / tot_scorable * 100.0), 2) if tot_scorable > 0 else 0.0
    eos_rate_pct = round((tot_eos / len(test_records) * 100.0), 2)
    avg_latency = round(total_latency_ms / len(test_records), 2)
    avg_tokens = round(total_tokens_gen / len(test_records), 2)

    # Prefix concentration analysis on test outputs
    test_first_two = Counter(" ".join(r["generated"].split()[:2]) for r in eval_records if len(r["generated"].split()) >= 2)
    max_prefix, max_cnt = test_first_two.most_common(1)[0] if test_first_two else ("NONE", 0)
    max_prefix_pct = round((max_cnt / len(test_records) * 100.0), 2)

    eval_summary = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": str(best_ckpt),
        "total_test_records": len(test_records),
        "total_scorable_records": tot_scorable,
        "total_correct": tot_correct,
        "scorable_task_accuracy_pct": task_acc_pct,
        "total_eos_completions": tot_eos,
        "eos_completion_rate_pct": eos_rate_pct,
        "structured_formatting_json_validity_pct": json_validity_pct,
        "avg_tokens_generated": avg_tokens,
        "avg_latency_ms": avg_latency,
        "max_output_prefix": max_prefix,
        "max_output_prefix_count": max_cnt,
        "max_output_prefix_pct": max_prefix_pct,
        "top_output_prefixes": test_first_two.most_common(5),
        "category_metrics": {c: dict(cat_counts[c]) for c in cat_counts},
    }

    with open(EXP_DIR / "EVAL_EXP19_RESULTS.json", "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    print("\n=== EXP19 POST-TRAINING EVALUATION SUMMARY ===")
    print(f"Scorable Task Accuracy: {task_acc_pct}% ({tot_correct}/{tot_scorable})")
    print(f"EOS Termination Rate:   {eos_rate_pct}% ({tot_eos}/{len(test_records)})")
    print(f"JSON Validity Rate:     {json_validity_pct}%")
    print(f"Max Prefix Concentration: {max_prefix} = {max_prefix_pct}%")
    print(f"Avg Generated Tokens:   {avg_tokens}")
    print(f"Avg Query Latency:      {avg_latency} ms")
    print("Execution completed successfully!")

if __name__ == "__main__":
    main()
