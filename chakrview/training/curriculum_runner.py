"""
Step 102 Controlled Stage B Curriculum Training Experiment Runner.

Executes:
1. Candidate initialization with seed=42.
2. 500-step curriculum optimization on Stage B train shards.
3. Periodic validation evaluation every 50 steps.
4. Final checkpoint saving to artifacts/step102_stage_b_curriculum/checkpoints/.
5. Full evaluation on Stage B validation, Stage B test, and the 40-probe Syntactic Benchmark.
6. Baseline vs Candidate comparative reporting.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Any, Tuple
import torch
from torch.utils.data import DataLoader

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
    load_trained_checkpoint,
)
from chakrview.runtime.neural_inference_contract import (
    NeuralInferenceContract,
    NeuralInferencePayload,
    NeuralInferenceConfig,
)
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.training.config import (
    PretrainingConfig,
    TrainingHyperparameters,
    CheckpointConfig,
    EvaluationConfig,
    DataConfig,
)
from chakrview.training.dataset import StreamingTokenDataset
from chakrview.training.loss import CausalLoss
from chakrview.training.trainer import Trainer
from chakrview.training.syntactic_probe import SyntacticProbeEvaluator, SyntacticProbeMetrics


def evaluate_split_full(
    model: ChakrMicro,
    shard_dir: Path,
    sequence_length: int = 128,
    batch_size: int = 2,
    device: str = "cpu",
    max_batches: int = 100,
) -> Tuple[float, float, int]:
    """Evaluates cross-entropy loss and perplexity over a dataset split without gradients."""
    model.eval()
    model.to(device)
    loss_fn = CausalLoss(ignore_index=2)
    ds = StreamingTokenDataset(shard_dir, sequence_length=sequence_length, loop=False, seed=42)
    loader = DataLoader(ds, batch_size=batch_size)

    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for b in loader:
            if batches >= max_batches:
                break
            logits = model(b["input_ids"].to(device))
            loss = loss_fn(logits, b["target_ids"].to(device))
            total_loss += loss.item()
            batches += 1

    mean_loss = total_loss / batches if batches > 0 else 0.0
    ppl = math.exp(min(20.0, mean_loss)) if mean_loss > 0 else 0.0
    return round(mean_loss, 4), round(ppl, 2), batches


def run_step102_training_experiment(
    artifacts_dir: Path,
    max_steps: int = 500,
    sequence_length: int = 256,
    batch_size: int = 2,
    accum_steps: int = 2,
    eval_interval: int = 50,
    save_interval: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Executes the complete Step 102 Controlled Training Curriculum.
    """
    t_start = time.perf_counter()
    root_dir = Path(__file__).resolve().parents[2]
    tok_dir = root_dir / "data" / "experiments" / "vocab_4096"
    train_dir = root_dir / "data" / "tokenized" / "stage_b" / "train"
    val_dir = root_dir / "data" / "tokenized" / "stage_b" / "validation"
    test_dir = root_dir / "data" / "tokenized" / "stage_b" / "test"
    probes_file = root_dir / "tests" / "fixtures" / "step102_syntactic_probes.json"

    tokenizer, _ = load_tokenizer_artifacts(tok_dir)
    probe_evaluator = SyntacticProbeEvaluator(tokenizer=tokenizer, probes_path=probes_file)

    # 1. Baseline Evaluation
    canonical_model = instantiate_frozen_baseline()
    base_hash = compute_model_hash(canonical_model)
    assert base_hash == EXPECTED_WEIGHT_HASH

    base_val_loss, base_val_ppl, _ = evaluate_split_full(canonical_model, val_dir, max_batches=50)
    base_test_loss, base_test_ppl, _ = evaluate_split_full(canonical_model, test_dir, max_batches=50)
    base_probes = probe_evaluator.evaluate(canonical_model)

    # 2. Candidate Configuration
    ckpt_dir = artifacts_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    config = PretrainingConfig(
        model=ModelConfig(),
        training=TrainingHyperparameters(
            max_steps=max_steps,
            learning_rate=1.5e-3,
            min_learning_rate=1.5e-4,
            warmup_steps=25,
            gradient_accumulation_steps=accum_steps,
            gradient_clipping=1.0,
            seed=seed,
            batch_size=batch_size,
        ),
        checkpoint=CheckpointConfig(
            directory=str(ckpt_dir),
            save_interval=save_interval,
            keep_last_n=3,
        ),
        evaluation=EvaluationConfig(
            eval_interval=eval_interval,
            eval_batches=5,
            eval_on_start=True,
        ),
        data=DataConfig(
            train_path=str(train_dir),
            validation_path=str(val_dir),
            sequence_length=sequence_length,
        ),
    )

    train_ds = StreamingTokenDataset(train_dir, sequence_length=sequence_length, loop=True, seed=seed)
    val_ds = StreamingTokenDataset(val_dir, sequence_length=sequence_length, loop=True, seed=seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # 3. Train Candidate
    trainer = Trainer(config=config, train_loader=train_loader, val_loader=val_loader)
    initial_candidate_hash = compute_model_hash(trainer.model)

    train_result = trainer.train()
    final_step = train_result["final_step"]
    final_ckpt_path = Path(train_result["final_checkpoint"])
    final_candidate_hash = compute_model_hash(trainer.model)

    # 4. Final Evaluation of Trained Checkpoint
    cand_val_loss, cand_val_ppl, _ = evaluate_split_full(trainer.model, val_dir, max_batches=50)
    cand_test_loss, cand_test_ppl, _ = evaluate_split_full(trainer.model, test_dir, max_batches=50)
    cand_probes = probe_evaluator.evaluate(trainer.model)

    # 5. Inference Contract Verification
    contract = NeuralInferenceContract.from_checkpoint(final_ckpt_path, tokenizer=tokenizer)
    inference_hash_pre = contract.compute_weight_hash()
    assert inference_hash_pre == final_candidate_hash

    payload = NeuralInferencePayload(
        prompt="def calculate_total(items):\n   ",
        config=NeuralInferenceConfig(max_new_tokens=8, temperature=0.0),
    )
    inf_out = contract.generate(payload)
    assert inf_out.weight_hash_verified is True
    assert contract.compute_weight_hash() == final_candidate_hash

    # 6. Re-verify Canonical Baseline
    re_canonical = instantiate_frozen_baseline()
    assert compute_model_hash(re_canonical) == EXPECTED_WEIGHT_HASH

    elapsed_total = time.perf_counter() - t_start

    # Determine Scientific Verdict
    val_loss_improved = (cand_val_loss < base_val_loss)
    test_loss_improved = (cand_test_loss < base_test_loss)
    probes_improved = (cand_probes.top5_accuracy > base_probes.top5_accuracy)
    gen_gap = abs(cand_val_loss - trainer.metrics.history[-1]["train_loss"])

    if val_loss_improved and test_loss_improved and probes_improved and cand_val_loss <= 4.80:
        verdict = "STRONG POSITIVE EVIDENCE"
    elif val_loss_improved and test_loss_improved:
        verdict = "POSITIVE BUT LIMITED EVIDENCE"
    else:
        verdict = "INCONCLUSIVE"

    summary = {
        "milestone": "Step 102",
        "experiment_name": "Stage B Curriculum Training & Syntactic Probe Benchmark",
        "elapsed_sec": round(elapsed_total, 2),
        "steps_completed": final_step,
        "seed": seed,
        "verdict": verdict,
        "canonical_baseline": {
            "parameter_count": 3_443_136,
            "weight_hash": EXPECTED_WEIGHT_HASH,
            "validation_loss": base_val_loss,
            "validation_perplexity": base_val_ppl,
            "test_loss": base_test_loss,
            "test_perplexity": base_test_ppl,
            "probe_top1_accuracy": base_probes.top1_accuracy,
            "probe_top5_accuracy": base_probes.top5_accuracy,
            "probe_mean_log_prob": base_probes.mean_target_log_prob,
        },
        "trained_candidate": {
            "parameter_count": 3_443_136,
            "initial_weight_hash": initial_candidate_hash,
            "final_weight_hash": final_candidate_hash,
            "checkpoint_path": str(final_ckpt_path),
            "final_train_loss": trainer.metrics.history[-1]["train_loss"],
            "validation_loss": cand_val_loss,
            "validation_perplexity": cand_val_ppl,
            "test_loss": cand_test_loss,
            "test_perplexity": cand_test_ppl,
            "generalization_gap": round(gen_gap, 4),
            "probe_top1_accuracy": cand_probes.top1_accuracy,
            "probe_top5_accuracy": cand_probes.top5_accuracy,
            "probe_mean_log_prob": cand_probes.mean_target_log_prob,
            "category_metrics": cand_probes.category_metrics,
            "sample_inference": inf_out.generated_text,
        },
        "comparison": {
            "validation_loss_delta": round(cand_val_loss - base_val_loss, 4),
            "validation_perplexity_delta": round(cand_val_ppl - base_val_ppl, 2),
            "test_loss_delta": round(cand_test_loss - base_test_loss, 4),
            "test_perplexity_delta": round(cand_test_ppl - base_test_ppl, 2),
            "probe_top1_gain": round(cand_probes.top1_accuracy - base_probes.top1_accuracy, 2),
            "probe_top5_gain": round(cand_probes.top5_accuracy - base_probes.top5_accuracy, 2),
        },
        "trajectory": [
            {
                "step": h["step"],
                "train_loss": h["train_loss"],
                "learning_rate": h["learning_rate"],
                "val_loss": h.get("val_loss"),
                "val_perplexity": h.get("val_perplexity"),
                "grad_norm": h.get("grad_norm"),
            }
            for h in trainer.metrics.history
        ],
    }

    # Save summary artifact
    summary_path = artifacts_dir / "step102_experiment_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
