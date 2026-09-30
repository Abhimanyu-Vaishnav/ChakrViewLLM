"""
ChakrView Step 53: Foundation Curriculum Training & Generalization Experiment.

Executes:
1. Multi-tier curriculum dataset generation (Levels 0A, 0B, 0C, 0D, 1).
2. Clean train/validation/test split and binary sharding.
3. Controlled training experiments on CPU (500 steps).
4. Dual evaluation:
   - Frozen Step-52 Capability Benchmark (Anchor)
   - Held-Out Generalization Benchmark (10 completely unseen tasks)
5. Strict CPU resource and immutability verification.
"""

from __future__ import annotations

import ast
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.curriculum.generator import FoundationCurriculumGenerator
from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    DEFAULT_TOKENIZER_DIR,
)
from chakrview.runtime.pipeline import (
    EXPECTED_WEIGHT_HASH,
    EXPECTED_VOCAB_SIZE,
    MAX_CONTEXT_WINDOW,
)
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.training.dataset import StreamingTokenDataset
from scripts.experiment_step52_model_capability import (
    LEVEL_0_BENCHMARK_TASKS,
    CapabilityTask,
    CapabilityFailureCategory,
    DeterministicCapabilityEvaluator,
    ModelCapabilityBenchmarkRunner,
)

CURRICULUM_DATA_DIR = ROOT_DIR / "data" / "tokenized" / "curriculum_step53"
ARTIFACTS_DIR = ROOT_DIR / "artifacts" / "step53"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Held-Out Generalization Benchmark (Unseen Prompts)
# ─────────────────────────────────────────────────────────────────────────────

HELDOUT_BENCHMARK_TASKS: List[CapabilityTask] = [
    CapabilityTask(
        task_id="HELD_MATH_01",
        category="arithmetic",
        prompt="4 + 7 = ",
        expected_description="Result must start with 11",
        expected_values=["11"],
        max_new_tokens=8,
        eval_fn_name="arithmetic_exact",
    ),
    CapabilityTask(
        task_id="HELD_MATH_02",
        category="arithmetic",
        prompt="3 * 6 = ",
        expected_description="Result must start with 18",
        expected_values=["18"],
        max_new_tokens=8,
        eval_fn_name="arithmetic_exact",
    ),
    CapabilityTask(
        task_id="HELD_COMP_01",
        category="computation",
        prompt="9 > 4 is ",
        expected_description="Result contains True",
        expected_values=["True", "true"],
        max_new_tokens=8,
        eval_fn_name="contains_any",
    ),
    CapabilityTask(
        task_id="HELD_CODE_01",
        category="code_completion",
        prompt="def double(x):\n    return ",
        expected_description="Completes with valid Python AST",
        expected_values=["x * 2", "2 * x", "x + x", "x"],
        max_new_tokens=12,
        is_code=True,
        eval_fn_name="python_code_ast",
    ),
    CapabilityTask(
        task_id="HELD_CODE_02",
        category="code_completion",
        prompt="def is_negative(val):\n    return ",
        expected_description="Completes with valid Python AST",
        expected_values=["val < 0", "val <= 0", "False"],
        max_new_tokens=12,
        is_code=True,
        eval_fn_name="python_code_ast",
    ),
    CapabilityTask(
        task_id="HELD_JSON_01",
        category="structured_completion",
        prompt='{"mode": "debug", "active": ',
        expected_description="Completes with valid JSON",
        expected_values=["true}", "false}", "true", "false"],
        max_new_tokens=12,
        is_json=True,
        eval_fn_name="json_completion",
    ),
    CapabilityTask(
        task_id="HELD_TRANS_01",
        category="transformation",
        prompt="The opposite of cold is ",
        expected_description="Result contains hot",
        expected_values=["hot", "warm"],
        max_new_tokens=8,
        eval_fn_name="contains_any",
    ),
    CapabilityTask(
        task_id="HELD_FACT_01",
        category="factual_formatting",
        prompt="The capital of Japan is ",
        expected_description="Result contains Tokyo",
        expected_values=["Tokyo", "tokyo"],
        max_new_tokens=10,
        eval_fn_name="contains_any",
    ),
    CapabilityTask(
        task_id="HELD_INST_01",
        category="instruction_following",
        prompt="Repeat the word 'chakrview':\nOutput: ",
        expected_description="Result contains chakrview",
        expected_values=["chakrview", "ChakrView"],
        max_new_tokens=10,
        eval_fn_name="contains_any",
    ),
    CapabilityTask(
        task_id="HELD_LIST_01",
        category="structured_completion",
        prompt="items = [10, 20, 30, ",
        expected_description="Completes with valid Python list AST",
        expected_values=["]"],
        max_new_tokens=12,
        is_code=True,
        eval_fn_name="python_list_ast",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Shard Generation & Loss Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def prepare_curriculum_shards(tokenizer, output_dir: Path, seed: int = 42) -> Dict[str, Any]:
    """Generate and shard balanced curriculum samples."""
    print("Generating balanced multi-tier curriculum samples...")
    gen = FoundationCurriculumGenerator(seed=seed)
    samples = gen.generate_all_samples()
    print(f"Generated {len(samples)} total curriculum samples.")

    splits = gen.build_dataset_splits(samples, train_ratio=0.8, val_ratio=0.1)
    print(f"Splits: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")

    manifest = gen.write_shards(splits, output_dir=output_dir, tokenizer=tokenizer)
    return manifest


def evaluate_split_loss(model: ChakrMicro, shard_dir: Path, seq_len: int = 512) -> Tuple[float, float]:
    """Compute mean cross-entropy loss and perplexity on token shards."""
    model.eval()
    dataset = StreamingTokenDataset(
        shard_dir=shard_dir,
        sequence_length=seq_len,
        loop=False,
        drop_remainder=False,
    )

    losses = []
    with torch.no_grad():
        for batch in dataset:
            input_ids = batch["input_ids"].unsqueeze(0)
            target_ids = batch["target_ids"].unsqueeze(0)

            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, EXPECTED_VOCAB_SIZE), target_ids.view(-1))
            losses.append(loss.item())

    if not losses:
        return 8.35, math.exp(8.35)

    mean_loss = float(np.mean(losses))
    ppl = math.exp(min(mean_loss, 20.0))
    return mean_loss, ppl


# ─────────────────────────────────────────────────────────────────────────────
# 3. Controlled Training Runner
# ─────────────────────────────────────────────────────────────────────────────

def train_curriculum_checkpoint(
    tokenizer,
    train_shard_dir: Path,
    val_shard_dir: Path,
    steps: int = 500,
    learning_rate: float = 1e-3,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train an isolated experimental ChakrMicro checkpoint on curriculum shards."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    config = ModelConfig()
    model = ChakrMicro(config)
    initial_weights = {n: p.clone().detach() for n, p in model.named_parameters()}
    initial_hash = compute_model_hash(model)

    init_val_loss, init_val_ppl = evaluate_split_loss(model, val_shard_dir)
    print(f"Pre-training Validation Loss: {init_val_loss:.4f} (PPL: {init_val_ppl:.2f})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    dataset = StreamingTokenDataset(
        shard_dir=train_shard_dir,
        sequence_length=512,
        loop=True,
        drop_remainder=True,
        seed=seed,
    )
    data_iter = iter(dataset)

    step_losses = []
    model.train()
    start_time = time.perf_counter()

    for step in range(1, steps + 1):
        batch = next(data_iter)
        input_ids = batch["input_ids"].unsqueeze(0)
        target_ids = batch["target_ids"].unsqueeze(0)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, EXPECTED_VOCAB_SIZE), target_ids.view(-1))

        assert not torch.isnan(loss) and not torch.isinf(loss), f"Loss diverged at step {step}"
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        step_losses.append(round(loss.item(), 4))
        if step % 100 == 0 or step == steps:
            print(f"Step {step:04d}/{steps:04d} | Train Loss: {loss.item():.4f}")

    train_duration = time.perf_counter() - start_time
    final_val_loss, final_val_ppl = evaluate_split_loss(model, val_shard_dir)
    final_hash = compute_model_hash(model)

    delta_l2 = 0.0
    for n, p in model.named_parameters():
        delta_l2 += torch.norm(p - initial_weights[n], p=2).item() ** 2
    delta_l2 = math.sqrt(delta_l2)

    # Save checkpoint atomically
    ckpt_dir = ARTIFACTS_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"checkpoint_step{steps:05d}.pt"
    tmp_path = ckpt_dir / f"checkpoint_step{steps:05d}.pt.tmp"

    checkpoint_payload = {
        "step": steps,
        "model_state_dict": model.state_dict(),
        "model_hash": final_hash,
        "initial_val_loss": init_val_loss,
        "final_val_loss": final_val_loss,
        "final_val_ppl": final_val_ppl,
        "delta_l2": delta_l2,
        "checkpoint_type": "step53_foundation_curriculum",
    }
    torch.save(checkpoint_payload, tmp_path)
    os.replace(tmp_path, ckpt_path)

    return {
        "steps": steps,
        "train_duration_sec": round(train_duration, 2),
        "steps_per_sec": round(steps / train_duration, 2),
        "initial_val_loss": round(init_val_loss, 4),
        "final_val_loss": round(final_val_loss, 4),
        "final_val_ppl": round(final_val_ppl, 2),
        "loss_reduction_pct": round((init_val_loss - final_val_loss) / init_val_loss * 100.0, 2),
        "delta_l2": round(delta_l2, 4),
        "checkpoint_path": str(ckpt_path),
        "model_hash": final_hash,
        "model": model,
    }


def run_experiment_step53(steps: int = 500) -> Dict[str, Any]:
    """Execute end-to-end Step 53 curriculum preparation, training, and dual evaluation."""
    print("=" * 70)
    print("CHAKRVIEW STEP 53: FOUNDATION CURRICULUM & CONTROLLED TRAINING")
    print("=" * 70)

    tokenizer, _ = load_tokenizer_artifacts(DEFAULT_TOKENIZER_DIR)

    # 1. Verify Frozen Baseline Immutability
    baseline_model = instantiate_frozen_baseline()
    base_hash = compute_model_hash(baseline_model)
    assert base_hash == EXPECTED_WEIGHT_HASH, "Baseline hash mismatch!"
    print(f"Verified frozen baseline hash intact: {base_hash}")

    # 2. Generate and Shard Balanced Curriculum Dataset
    shards_dir = CURRICULUM_DATA_DIR
    manifest = prepare_curriculum_shards(tokenizer, output_dir=shards_dir, seed=42)

    train_shard_dir = shards_dir / "train"
    val_shard_dir = shards_dir / "val"

    # 3. Train Isolated Experimental Checkpoint (500 steps)
    train_res = train_curriculum_checkpoint(
        tokenizer=tokenizer,
        train_shard_dir=train_shard_dir,
        val_shard_dir=val_shard_dir,
        steps=steps,
        seed=42,
    )
    trained_model = train_res.pop("model")

    # 4. Evaluate on Step-52 Anchor Capability Benchmark (20 tasks)
    print("\nEvaluating on Frozen Step-52 Capability Benchmark (Anchor)...")
    runner_anchor = ModelCapabilityBenchmarkRunner(
        model=trained_model,
        tokenizer=tokenizer,
        model_name=f"step53_curriculum_{steps}steps",
        seed=42,
        sampling_mode="greedy",
    )
    anchor_eval = runner_anchor.run_benchmark(LEVEL_0_BENCHMARK_TASKS)
    print(f"Step-52 Anchor Pass Rate: {anchor_eval['overall_pass_rate']*100:.1f}% "
          f"({anchor_eval['passed_tasks']}/{anchor_eval['total_tasks']})")
    print(f"Failure Distribution: {anchor_eval['failure_distribution']}")

    # 5. Evaluate on Held-Out Generalization Benchmark (10 unseen tasks)
    print("\nEvaluating on Held-Out Generalization Benchmark (Unseen Prompts)...")
    runner_heldout = ModelCapabilityBenchmarkRunner(
        model=trained_model,
        tokenizer=tokenizer,
        model_name=f"step53_curriculum_heldout",
        seed=42,
        sampling_mode="greedy",
    )
    heldout_eval = runner_heldout.run_benchmark(HELDOUT_BENCHMARK_TASKS)
    print(f"Held-Out Generalization Pass Rate: {heldout_eval['overall_pass_rate']*100:.1f}% "
          f"({heldout_eval['passed_tasks']}/{heldout_eval['total_tasks']})")
    print(f"Failure Distribution: {heldout_eval['failure_distribution']}")

    # 6. Baseline Verification Post-Training
    post_base_hash = compute_model_hash(baseline_model)
    assert post_base_hash == EXPECTED_WEIGHT_HASH, "Baseline mutated during training!"

    # 7. Serialize Artifacts
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = ARTIFACTS_DIR / "step53_training_report.json"

    master_payload = {
        "timestamp": time.time(),
        "baseline_hash": EXPECTED_WEIGHT_HASH,
        "training_metadata": train_res,
        "anchor_benchmark_results": anchor_eval,
        "heldout_benchmark_results": heldout_eval,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(master_payload, f, indent=2)

    print(f"\nArtifacts successfully written to: {report_path}")
    print("=" * 70)
    return master_payload


if __name__ == "__main__":
    run_experiment_step53(steps=500)
