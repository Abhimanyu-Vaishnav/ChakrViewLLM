"""
Exp20 Single-Task Context Extraction SFT Training & Evaluation Execution Script (400 Steps on CPU).
"""

import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
import torch

REPO_ROOT = Path("D:/Project/ChakrView").resolve()
EXP_DIR = REPO_ROOT / "experiments" / "overnight_runs" / "exp20_single_task_contextual_binding_cpu_400_20261024_133947"
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
    print("=== Launching Authorized Exp20 Single-Task Context Extraction SFT (400 Steps CPU) ===")
    start_iso = datetime.now().isoformat()
    print(f"Start Timestamp: {start_iso}")
    print(f"Output Directory: {EXP_DIR}")

    train_file = DATA_DIR / "exp20_train_context_extraction.jsonl"
    val_file = DATA_DIR / "exp20_val_context_extraction.jsonl"
    test_file = DATA_DIR / "exp20_test_frozen.jsonl"
    base_checkpoint = CKPT_DIR / "training_checkpoint_start.pt"

    assert train_file.exists(), f"Train file missing: {train_file}"
    assert val_file.exists(), f"Val file missing: {val_file}"
    assert test_file.exists(), f"Test file missing: {test_file}"
    assert base_checkpoint.exists(), f"Base checkpoint missing: {base_checkpoint}"

    config = SFTConfig(
        repo_root=REPO_ROOT,
        train_file=train_file,
        val_file=val_file,
        output_dir=EXP_DIR,
        base_checkpoint=base_checkpoint,
        steps=400,
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
        tokenizer_dir=str(REPO_ROOT / "data/experiments/vocab_4096"),
        allow_exploratory_opt_in=True,
        enforce_trust_gate=False,
        strip_prompt_delimiters=True,
        save_best_checkpoint=True,
    )

    with open(EXP_DIR / "TRAINING_CONFIG.json", "w", encoding="utf-8") as f:
        json.dump({
            "experiment_id": "exp20_single_task_contextual_binding_cpu_400",
            "task_family": "context_extraction",
            "steps": config.steps,
            "batch_size": config.batch_size,
            "max_seq_len": config.max_seq_len,
            "learning_rate": config.learning_rate,
            "min_learning_rate": config.min_learning_rate,
            "weight_decay": config.weight_decay,
            "grad_clip": config.grad_clip,
            "val_interval": config.val_interval,
            "save_interval": config.save_interval,
            "seed": config.seed,
            "device": config.device,
            "base_checkpoint": str(base_checkpoint),
            "train_file": str(train_file),
            "val_file": str(val_file),
            "test_file": str(test_file)
        }, f, indent=2)

    # Execute training
    print("Initializing SFTExperimentEngine...")
    engine = SFTExperimentEngine(config)
    print("Starting optimizer steps (400 steps)...")
    train_start_t = time.perf_counter()
    train_res = engine.train()
    train_wall_sec = time.perf_counter() - train_start_t

    print("Training finished successfully in {:.2f} seconds ({:.2f} min).".format(
        train_wall_sec, train_wall_sec / 60.0
    ))
    print("Initial Val Loss: {:.4f} -> Final Val Loss: {:.4f}".format(
        train_res.initial_val_loss, train_res.final_val_loss
    ))
    print("Best Val Loss: {:.4f} achieved at Step {}".format(
        train_res.best_val_loss, train_res.best_step
    ))

    # Move checkpoints to checkpoints/ subdir if engine saved in root output_dir
    best_ckpt = CKPT_DIR / "sft_best_checkpoint.pt"
    if (EXP_DIR / "sft_best_checkpoint.pt").exists() and not best_ckpt.exists():
        (EXP_DIR / "sft_best_checkpoint.pt").rename(best_ckpt)
    elif not best_ckpt.exists() and (EXP_DIR / "checkpoints" / "sft_best_checkpoint.pt").exists():
        best_ckpt = EXP_DIR / "checkpoints" / "sft_best_checkpoint.pt"

    # Also move step checkpoints to checkpoints/ if in EXP_DIR root
    for p in EXP_DIR.glob("sft_checkpoint_step_*.pt"):
        dest = CKPT_DIR / p.name
        if not dest.exists():
            p.rename(dest)

    # Save TRAINING_LOG.jsonl
    with open(EXP_DIR / "TRAINING_LOG.jsonl", "w", encoding="utf-8") as f:
        for entry in engine.history:
            f.write(json.dumps(entry) + "\n")

    # Evaluate Best Checkpoint on Frozen Test Set (20 records)
    print("Evaluating Best Checkpoint ({}) on 20 frozen test records...".format(best_ckpt))
    with open(test_file, "r", encoding="utf-8") as f:
        test_records = [json.loads(line) for line in f if line.strip()]

    tokenizer, _ = load_tokenizer_artifacts(config.tokenizer_dir)
    contract = NeuralInferenceContract.from_checkpoint(
        checkpoint_path=best_ckpt,
        tokenizer=tokenizer,
        config=ModelConfig(),
        device=torch.device("cpu"),
    )
    sampler = Sampler()
    evaluator = TaskEvaluator()
    s_cfg = SamplingConfig(temperature=0.0, seed=42)

    eval_records = []
    correct_count = 0
    eos_count = 0
    total_latency_ms = 0.0
    total_tokens_gen = 0

    for rec in test_records:
        p_mode = rec.get("prompt_mode", "context_qa")
        prompt = rec["prompt"]
        context = rec.get("context", "")

        t_start = time.perf_counter()
        f_prompt = format_conditioned_prompt(prompt, prompt_mode=p_mode, context=context)
        p_tokens = contract.encode(f_prompt, add_bos=True, add_eos=False)

        gen_tokens = []
        curr = list(p_tokens)
        stop_reason = "MAX_TOKENS"

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
        if stop_reason == "STOP_TOKEN":
            eos_count += 1

        gen_str = contract.decode(gen_tokens)
        eval_res = evaluator.evaluate_record(rec, gen_str, stop_reason=stop_reason)
        if eval_res.outcome == EvaluationOutcome.CORRECT:
            correct_count += 1

        eval_records.append({
            "record_id": rec["id"],
            "category": rec["category"],
            "prompt": prompt,
            "context": context,
            "expected": rec["response"],
            "generated": gen_str,
            "stop_reason": stop_reason,
            "tokens_generated": len(gen_tokens),
            "latency_ms": lat_ms,
            "outcome": eval_res.outcome.value,
            "scoring_status": eval_res.scoring_status.value,
            "reasons": eval_res.reason_codes,
            "format_valid": eval_res.format_valid
        })

    # Save per record evaluation
    with open(EXP_DIR / "EVAL_EXP20_PER_RECORD.jsonl", "w", encoding="utf-8") as f:
        for r in eval_records:
            f.write(json.dumps(r) + "\n")

    acc_pct = round(correct_count / len(test_records) * 100, 2)
    eos_pct = round(eos_count / len(test_records) * 100, 2)

    eval_summary = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": str(best_ckpt),
        "best_step": train_res.best_step,
        "best_val_loss": train_res.best_val_loss,
        "total_test_records": len(test_records),
        "total_scorable_records": len(test_records),
        "total_correct": correct_count,
        "scorable_task_accuracy_pct": acc_pct,
        "total_eos_completions": eos_count,
        "eos_completion_rate_pct": eos_pct,
        "avg_tokens_generated": round(total_tokens_gen / len(test_records), 2),
        "avg_latency_ms": round(total_latency_ms / len(test_records), 2),
    }

    with open(EXP_DIR / "EVAL_EXP20_RESULTS.json", "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    print("Post-training evaluation complete:")
    print("Exact Task Accuracy: {}/{} ({:.2f}%)".format(correct_count, len(test_records), acc_pct))
    print("EOS Completion Rate: {}/{} ({:.2f}%)".format(eos_count, len(test_records), eos_pct))

if __name__ == "__main__":
    main()
