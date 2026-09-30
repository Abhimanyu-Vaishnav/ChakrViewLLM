"""
ChakrView Step 55: Multi-Task Generalization & Anti-Forgetting Experiment.

Executes:
1. Multi-Task Interleaved Dataset Generation (7 domains: Foundation, Computation, Programming,
   Reasoning, Diagnosis, Trajectory, Instruction).
2. Binary sharding into data/tokenized/multitask_step55/.
3. Controlled training of isolated experimental candidate checkpoint on CPU (500 steps).
4. Quadruple capability & anti-forgetting benchmark matrix:
   - Benchmark A: Step-52 Anchor Capability Benchmark (20 tasks)
   - Benchmark B: Step-53 Held-Out Generalization Benchmark (10 tasks)
   - Benchmark C: Step-54 Structured Reasoning Benchmark (10 tasks)
   - Benchmark D: Step-55 Combinatorial Generalization Benchmark (10 unseen combinations)
5. Metric computation:
   - Pass rates across all four benchmarks.
   - Catastrophic forgetting measurement relative to best historical checkpoints.
   - Repetition collapse, latency, parameter efficiency.
6. Baseline immutability verification.
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
from chakrview.curriculum.multitask import MultiTaskCurriculumGenerator, DomainWeights
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
from scripts.experiment_step53_curriculum import HELDOUT_BENCHMARK_TASKS
from scripts.experiment_step54_reasoning import REASONING_BENCHMARK_TASKS

MULTITASK_DATA_DIR = ROOT_DIR / "data" / "tokenized" / "multitask_step55"
ARTIFACTS_DIR = ROOT_DIR / "artifacts" / "step55"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Step-55 Combinatorial Generalization Benchmark Tasks (10 Tasks)
# ─────────────────────────────────────────────────────────────────────────────
# Tests combinations of primitives: novel operands, novel identifiers, combined transforms.
# NONE of these exact strings are present in the training set.

STEP55_COMBINATORIAL_TASKS: List[CapabilityTask] = [
    CapabilityTask(
        task_id="COMB_ARITH_01",
        category="novel_arithmetic_operands",
        prompt="17 + 25 = ",
        expected_description="Novel addition operands: must start with 42",
        expected_values=["42"],
        max_new_tokens=8,
        eval_fn_name="arithmetic_exact",
    ),
    CapabilityTask(
        task_id="COMB_ARITH_02",
        category="novel_subtraction_operands",
        prompt="80 - 35 = ",
        expected_description="Novel subtraction operands: must start with 45",
        expected_values=["45"],
        max_new_tokens=8,
        eval_fn_name="arithmetic_exact",
    ),
    CapabilityTask(
        task_id="COMB_COMP_01",
        category="novel_comparison",
        prompt="Comparison: 25 > 10 is ",
        expected_description="Novel comparison boundary: contains True",
        expected_values=["True", "true"],
        max_new_tokens=8,
        eval_fn_name="contains_any",
    ),
    CapabilityTask(
        task_id="COMB_PARITY_01",
        category="novel_parity",
        prompt="Parity: 34 is ",
        expected_description="Novel parity target: contains even",
        expected_values=["even"],
        max_new_tokens=8,
        eval_fn_name="contains_any",
    ),
    CapabilityTask(
        task_id="COMB_CODE_01",
        category="novel_function_name",
        prompt="def triple(x):\n    return ",
        expected_description="Novel function triple: valid AST completing with multiplication by 3",
        expected_values=["x * 3", "3 * x", "x + x + x"],
        max_new_tokens=12,
        is_code=True,
        eval_fn_name="code_ast",
    ),
    CapabilityTask(
        task_id="COMB_CODE_02",
        category="novel_boolean_check",
        prompt="def is_negative(val):\n    return ",
        expected_description="Novel function is_negative: valid AST checking val < 0",
        expected_values=["val < 0", "0 > val"],
        max_new_tokens=12,
        is_code=True,
        eval_fn_name="code_ast",
    ),
    CapabilityTask(
        task_id="COMB_JSON_01",
        category="novel_json_structure",
        prompt='{"endpoint": "auth", "valid": ',
        expected_description="Novel JSON key-value: completes with boolean true or false and bracket",
        expected_values=["true}", "false}", "true", "false"],
        max_new_tokens=8,
        eval_fn_name="contains_any",
    ),
    CapabilityTask(
        task_id="COMB_PLAN_01",
        category="novel_state_machine",
        prompt="System State: ENGINE is STOPPED.\nPlan:\n1. Action: START -> State becomes ",
        expected_description="Novel state transition: contains RUNNING or STARTED",
        expected_values=["RUNNING", "STARTED", "running", "started"],
        max_new_tokens=8,
        eval_fn_name="contains_any",
    ),
    CapabilityTask(
        task_id="COMB_DIAG_01",
        category="novel_patch_selection",
        prompt="Diagnosis: Used addition '+' instead of multiplication '*'.\nFix:\ndef multiply(x, y):\n    return ",
        expected_description="Novel patch selection: emits x * y",
        expected_values=["x * y", "y * x"],
        max_new_tokens=8,
        is_code=True,
        eval_fn_name="code_ast",
    ),
    CapabilityTask(
        task_id="COMB_INS_01",
        category="novel_instruction",
        prompt="Instruction: Return JSON with key 'result' set to true.\nOutput: ",
        expected_description="Novel instruction following: emits JSON object",
        expected_values=['{"result": true}', '{"result":true}'],
        max_new_tokens=12,
        eval_fn_name="contains_any",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Dataset Sharding
# ─────────────────────────────────────────────────────────────────────────────

def prepare_multitask_shards(tokenizer, output_dir: Path, seed: int = 42) -> Dict[str, Any]:
    """Generate multi-task samples and serialize into shards."""
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = MultiTaskCurriculumGenerator(seed=seed)
    samples = generator.generate_all_samples(target_sample_count=800)
    splits = generator.build_dataset_splits(samples, train_ratio=0.8, val_ratio=0.1)

    print(f"Generated {len(samples)} total multi-task curriculum samples.")
    for s_name, s_list in splits.items():
        print(f"  - Split '{s_name}': {len(s_list)} samples")

    manifest = generator.write_shards(splits, output_dir, tokenizer, vocab_size=EXPECTED_VOCAB_SIZE)
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

def train_multitask_checkpoint(
    tokenizer,
    train_shard_dir: Path,
    val_shard_dir: Path,
    steps: int = 500,
    learning_rate: float = 1e-3,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train an isolated experimental ChakrMicro checkpoint on multi-task shards."""
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
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        step_losses.append(loss.item())

        if step % 100 == 0 or step == steps:
            elapsed = time.perf_counter() - start_time
            rate = step / elapsed
            print(f"Step {step:03d}/{steps:03d} | Loss: {loss.item():.4f} | Rate: {rate:.1f} steps/s")

    train_time = time.perf_counter() - start_time
    final_val_loss, final_val_ppl = evaluate_split_loss(model, val_shard_dir)
    print(f"Post-training Validation Loss: {final_val_loss:.4f} (PPL: {final_val_ppl:.2f})")

    delta_l2 = 0.0
    for n, p in model.named_parameters():
        delta_l2 += torch.norm(p.detach() - initial_weights[n]).item() ** 2
    delta_l2 = math.sqrt(delta_l2)
    print(f"Weight Divergence from Init: L2 norm = {delta_l2:.4f}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = ARTIFACTS_DIR / f"chakrmicro_step55_multitask_{steps}steps.pt"
    final_hash = compute_model_hash(model)

    torch.save({
        "step": steps,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": step_losses[-1],
        "val_loss": final_val_loss,
        "model_hash": final_hash,
        "seed": seed,
    }, ckpt_path)

    print(f"Experimental checkpoint saved to: {ckpt_path}")
    print(f"Checkpoint SHA-256: {final_hash}")

    return {
        "steps": steps,
        "learning_rate": learning_rate,
        "train_time_seconds": round(train_time, 2),
        "steps_per_second": round(steps / train_time, 2),
        "initial_val_loss": round(init_val_loss, 4),
        "initial_val_ppl": round(init_val_ppl, 2),
        "final_train_loss": round(step_losses[-1], 4),
        "final_val_loss": round(final_val_loss, 4),
        "final_val_ppl": round(final_val_ppl, 2),
        "loss_reduction_pct": round((init_val_loss - final_val_loss) / init_val_loss * 100.0, 2),
        "delta_l2": round(delta_l2, 4),
        "checkpoint_path": str(ckpt_path),
        "model_hash": final_hash,
        "model": model,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Master Multi-Task Capability & Forgetting Benchmark Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment_step55(steps: int = 500) -> Dict[str, Any]:
    """Execute end-to-end Step 55 multi-task experiment and evaluate 4-benchmark matrix."""
    print("=" * 70)
    print("CHAKRVIEW STEP 55: MULTI-TASK GENERALIZATION & ANTI-FORGETTING EXPERIMENT")
    print("=" * 70)

    tokenizer, _ = load_tokenizer_artifacts(DEFAULT_TOKENIZER_DIR)

    # 1. Baseline Immutability Check
    baseline_model = instantiate_frozen_baseline()
    base_hash = compute_model_hash(baseline_model)
    assert base_hash == EXPECTED_WEIGHT_HASH, "Baseline hash mismatch!"
    print(f"Verified frozen baseline hash intact: {base_hash}")

    # 2. Sharding
    shards_dir = MULTITASK_DATA_DIR
    manifest = prepare_multitask_shards(tokenizer, output_dir=shards_dir, seed=42)

    train_shard_dir = shards_dir / "train"
    val_shard_dir = shards_dir / "val"

    # 3. Training
    train_res = train_multitask_checkpoint(
        tokenizer=tokenizer,
        train_shard_dir=train_shard_dir,
        val_shard_dir=val_shard_dir,
        steps=steps,
        seed=42,
    )
    trained_model = train_res.pop("model")

    # 4. Benchmark A: Step-52 Anchor Capability Benchmark (20 tasks)
    print("\n--- Benchmark A: Step-52 Anchor Capability (20 tasks) ---")
    runner_anchor = ModelCapabilityBenchmarkRunner(
        model=trained_model,
        tokenizer=tokenizer,
        model_name="step55_anchor",
        seed=42,
        sampling_mode="greedy",
    )
    anchor_eval = runner_anchor.run_benchmark(LEVEL_0_BENCHMARK_TASKS)
    print(f"Anchor Pass Rate: {anchor_eval['overall_pass_rate']*100:.1f}% "
          f"({anchor_eval['passed_tasks']}/{anchor_eval['total_tasks']})")

    # 5. Benchmark B: Step-53 Held-Out Generalization Benchmark (10 tasks)
    print("\n--- Benchmark B: Step-53 Held-Out Generalization (10 tasks) ---")
    runner_heldout = ModelCapabilityBenchmarkRunner(
        model=trained_model,
        tokenizer=tokenizer,
        model_name="step55_heldout",
        seed=42,
        sampling_mode="greedy",
    )
    heldout_eval = runner_heldout.run_benchmark(HELDOUT_BENCHMARK_TASKS)
    print(f"Held-Out Pass Rate: {heldout_eval['overall_pass_rate']*100:.1f}% "
          f"({heldout_eval['passed_tasks']}/{heldout_eval['total_tasks']})")

    # 6. Benchmark C: Step-54 Structured Reasoning Benchmark (10 tasks)
    print("\n--- Benchmark C: Step-54 Structured Reasoning (10 tasks) ---")
    runner_reasoning = ModelCapabilityBenchmarkRunner(
        model=trained_model,
        tokenizer=tokenizer,
        model_name="step55_reasoning",
        seed=42,
        sampling_mode="greedy",
    )
    reasoning_eval = runner_reasoning.run_benchmark(REASONING_BENCHMARK_TASKS)
    print(f"Reasoning Pass Rate: {reasoning_eval['overall_pass_rate']*100:.1f}% "
          f"({reasoning_eval['passed_tasks']}/{reasoning_eval['total_tasks']})")

    # 7. Benchmark D: Step-55 Combinatorial Generalization Benchmark (10 tasks)
    print("\n--- Benchmark D: Step-55 Combinatorial Generalization (10 tasks) ---")
    runner_comb = ModelCapabilityBenchmarkRunner(
        model=trained_model,
        tokenizer=tokenizer,
        model_name="step55_combinatorial",
        seed=42,
        sampling_mode="greedy",
    )
    comb_eval = runner_comb.run_benchmark(STEP55_COMBINATORIAL_TASKS)
    print(f"Combinatorial Generalization Pass Rate: {comb_eval['overall_pass_rate']*100:.1f}% "
          f"({comb_eval['passed_tasks']}/{comb_eval['total_tasks']})")

    # 8. Post-Training Baseline Integrity Verification
    post_base_hash = compute_model_hash(baseline_model)
    assert post_base_hash == EXPECTED_WEIGHT_HASH, "Baseline mutated during training!"
    print(f"\nVerified frozen baseline immutable: {post_base_hash}")

    # 9. Catastrophic Forgetting & Generalization Accounting
    # Historical bests:
    # Step 53: Anchor = 40.0%, Heldout = 10.0%, Reasoning = 20.0%
    # Step 54: Anchor = 10.0%, Heldout = 10.0%, Reasoning = 80.0%
    forgetting_delta = {
        "anchor_vs_step53_best": round(anchor_eval['overall_pass_rate'] - 0.40, 4),
        "reasoning_vs_step54_best": round(reasoning_eval['overall_pass_rate'] - 0.80, 4),
        "heldout_vs_step53_best": round(heldout_eval['overall_pass_rate'] - 0.10, 4),
    }
    print(f"Catastrophic Forgetting Accounting: {forgetting_delta}")

    # 10. Gate Decision
    anchor_p = anchor_eval["overall_pass_rate"]
    heldout_p = heldout_eval["overall_pass_rate"]
    reasoning_p = reasoning_eval["overall_pass_rate"]
    comb_p = comb_eval["overall_pass_rate"]

    gate_approved = (anchor_p >= 0.80 and heldout_p >= 0.70 and reasoning_p >= 0.70 and comb_p >= 0.70)
    decision = "APPROVED" if gate_approved else "NOT_APPROVED_INSUFFICIENT_GENERALIZATION"

    # 11. Artifact Serialization
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = ARTIFACTS_DIR / "step55_multitask_report.json"

    master_payload = {
        "timestamp": time.time(),
        "baseline_hash": EXPECTED_WEIGHT_HASH,
        "checkpoint_type": "EXPERIMENTAL",
        "gate_status": decision,
        "release_0_1_approved": gate_approved,
        "training_metadata": train_res,
        "benchmark_a_anchor": anchor_eval,
        "benchmark_b_heldout": heldout_eval,
        "benchmark_c_reasoning": reasoning_eval,
        "benchmark_d_combinatorial": comb_eval,
        "catastrophic_forgetting_accounting": forgetting_delta,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(master_payload, f, indent=2)

    print(f"\nReport written to: {report_path}")
    print(f"Gate Decision: {decision}")
    print("=" * 70)
    return master_payload


if __name__ == "__main__":
    run_experiment_step55(steps=500)
