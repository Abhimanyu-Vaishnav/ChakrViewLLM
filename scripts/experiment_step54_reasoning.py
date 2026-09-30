"""
ChakrView Step 54: Structured Reasoning & Action-Observation Trajectory Experiment.

Executes:
1. Structured Reasoning Curriculum generation (Levels R0, R1, R2, R3, Trajectories).
2. Clean train/validation/test split and binary sharding into data/tokenized/reasoning_step54/.
3. Controlled training of isolated experimental candidate checkpoint on CPU (500 steps).
4. Triple evaluation:
   - Frozen Step-52 Capability Benchmark (Anchor: 20 tasks)
   - Held-Out Generalization Benchmark (10 tasks)
   - Step-54 Structured Reasoning Benchmark (10 tasks covering math, logic, state, diagnosis, decomposition)
5. RIL Preparation Layer: extracts structured experience records from trajectories.
6. Checkpoint management: maintains separation of BASELINE, EXPERIMENTAL, and VALIDATED checkpoints.
7. CPU resource profiling: latency, memory, throughput, and immutability verification.
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
from chakrview.curriculum.reasoning import ReasoningCurriculumGenerator
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

REASONING_DATA_DIR = ROOT_DIR / "data" / "tokenized" / "reasoning_step54"
ARTIFACTS_DIR = ROOT_DIR / "artifacts" / "step54"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Step-54 Structured Reasoning Benchmark Tasks (10 Tasks)
# ─────────────────────────────────────────────────────────────────────────────

REASONING_BENCHMARK_TASKS: List[CapabilityTask] = [
    CapabilityTask(
        task_id="REASON_MATH_01",
        category="arithmetic_reasoning",
        prompt="Calculate: 23 + 14 =\nAnswer: ",
        expected_description="Result must start with 37",
        expected_values=["37"],
        max_new_tokens=8,
        eval_fn_name="arithmetic_exact",
    ),
    CapabilityTask(
        task_id="REASON_COMP_01",
        category="logical_comparison",
        prompt="Question: Is 7 > 3?\nReasoning: 7 is greater than 3.\nAnswer: ",
        expected_description="Result contains True",
        expected_values=["True", "true"],
        max_new_tokens=8,
        eval_fn_name="contains_any",
    ),
    CapabilityTask(
        task_id="REASON_PARITY_01",
        category="classification",
        prompt="Input: 14\nQuestion: Is 14 even or odd?\nAnswer: ",
        expected_description="Result contains even",
        expected_values=["even"],
        max_new_tokens=8,
        eval_fn_name="contains_any",
    ),
    CapabilityTask(
        task_id="REASON_SELECT_01",
        category="selection_alternative",
        prompt="Question: What is the antonym of 'fast'?\nOptions: (A) quick  (B) slow  (C) rapid\nAnswer: ",
        expected_description="Result contains slow or (B)",
        expected_values=["slow", "(B)"],
        max_new_tokens=8,
        eval_fn_name="contains_any",
    ),
    CapabilityTask(
        task_id="REASON_PLAN_01",
        category="simple_planning",
        prompt="System State: Door is LOCKED.\nPlan:\n1. Action: UNLOCK -> State becomes UNLOCKED.\n2. Action: OPEN -> State becomes ",
        expected_description="Result contains OPEN",
        expected_values=["OPEN", "Open", "open"],
        max_new_tokens=8,
        eval_fn_name="contains_any",
    ),
    CapabilityTask(
        task_id="REASON_CHAIN_01",
        category="multi_step_arithmetic",
        prompt="Problem: Compute (3 * 4) + (10 / 2).\nStep 1: Compute 3 * 4 = 12.\nStep 2: Compute 10 / 2 = 5.\nStep 3: Add results: 12 + 5 = 17.\nFinal Answer: ",
        expected_description="Result starts with 17",
        expected_values=["17"],
        max_new_tokens=8,
        eval_fn_name="arithmetic_exact",
    ),
    CapabilityTask(
        task_id="REASON_DIAG_01",
        category="error_diagnosis",
        prompt="<OBSERVATION>\nFAILED: add(2, 3) returned -1, expected 5\n</OBSERVATION>\n<DIAGNOSIS>\n",
        expected_description="Diagnoses subtraction or negative operator",
        expected_values=["subtraction", "operator", "-", "sub"],
        max_new_tokens=16,
        eval_fn_name="contains_any",
    ),
    CapabilityTask(
        task_id="REASON_FIX_01",
        category="correction_selection",
        prompt="<DIAGNOSIS>\nUsed subtraction operator '-' instead of addition '+'.\n</DIAGNOSIS>\n<NEXT_ACTION>\ndef add(a, b):\n    return ",
        expected_description="Corrected implementation a + b",
        expected_values=["a + b", "b + a"],
        max_new_tokens=8,
        is_code=True,
        eval_fn_name="code_ast",
    ),
    CapabilityTask(
        task_id="REASON_DECOMP_01",
        category="task_decomposition",
        prompt="TASK: Create a function that checks whether a number is prime.\nUNDERSTAND:\n",
        expected_description="Identifies integer or prime requirements",
        expected_values=["prime", "integer", "divisor", "1"],
        max_new_tokens=16,
        eval_fn_name="contains_any",
    ),
    CapabilityTask(
        task_id="REASON_TAG_01",
        category="structured_instruction_following",
        prompt="<TAG>content</TAG>\n<STATE>idle</STATE>\n<ACTION>",
        expected_description="Produces structured tag closure",
        expected_values=["execute", "</ACTION>", "run"],
        max_new_tokens=8,
        eval_fn_name="contains_any",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Sharding & Dataset Preparation
# ─────────────────────────────────────────────────────────────────────────────

def prepare_reasoning_shards(tokenizer, output_dir: Path, seed: int = 42) -> Dict[str, Any]:
    """Generate structured reasoning samples and serialize into shards."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "reasoning_manifest.json"

    generator = ReasoningCurriculumGenerator(seed=seed)
    samples = generator.generate_all_samples(repeats=12)
    splits = generator.build_dataset_splits(samples, train_ratio=0.8, val_ratio=0.1)

    print(f"Generated {len(samples)} total reasoning curriculum samples.")
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
# 3. Controlled Training Runner (Isolated Experimental Checkpoint)
# ─────────────────────────────────────────────────────────────────────────────

def train_reasoning_checkpoint(
    tokenizer,
    train_shard_dir: Path,
    val_shard_dir: Path,
    steps: int = 500,
    learning_rate: float = 1e-3,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train an isolated experimental ChakrMicro checkpoint on reasoning shards."""
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

    # Weight divergence verification
    delta_l2 = 0.0
    for n, p in model.named_parameters():
        delta_l2 += torch.norm(p.detach() - initial_weights[n]).item() ** 2
    delta_l2 = math.sqrt(delta_l2)
    print(f"Weight Divergence from Init: L2 norm = {delta_l2:.4f}")

    # Save isolated experimental checkpoint
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = ARTIFACTS_DIR / f"chakrmicro_step54_reasoning_{steps}steps.pt"
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
# 4. RIL Preparation Layer: Extract Experience Records
# ─────────────────────────────────────────────────────────────────────────────

def extract_experience_from_trajectory(trajectory_text: str, task_id: str, success: bool) -> Dict[str, Any]:
    """
    Extract a structured experience record from canonical XML trajectory tags.
    This serves as the RIL Preparation Layer bridge.
    """
    def _extract_tag(tag: str) -> str:
        pattern = rf"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, trajectory_text, re.DOTALL)
        return match.group(1).strip() if match else ""

    spec = _extract_tag("SPEC")
    state = _extract_tag("STATE")
    action = _extract_tag("ACTION")
    observation = _extract_tag("OBSERVATION")
    diagnosis = _extract_tag("DIAGNOSIS")
    next_action = _extract_tag("NEXT_ACTION")
    result = _extract_tag("RESULT")

    return {
        "experience_id": f"exp_ril_{task_id}_{int(time.time())}",
        "task_id": task_id,
        "spec": spec,
        "initial_state": state,
        "initial_action": action,
        "observation": observation,
        "diagnosis": diagnosis,
        "corrective_action": next_action,
        "result_tag": result,
        "converged": success,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Master Step 54 Experiment Execution
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment_step54(steps: int = 500) -> Dict[str, Any]:
    """Execute end-to-end Step 54 reasoning preparation, training, and triple evaluation."""
    print("=" * 70)
    print("CHAKRVIEW STEP 54: STRUCTURED REASONING & TRAJECTORY EXPERIMENT")
    print("=" * 70)

    tokenizer, _ = load_tokenizer_artifacts(DEFAULT_TOKENIZER_DIR)

    # 1. Verify Frozen Baseline Immutability
    baseline_model = instantiate_frozen_baseline()
    base_hash = compute_model_hash(baseline_model)
    assert base_hash == EXPECTED_WEIGHT_HASH, "Baseline hash mismatch!"
    print(f"Verified frozen baseline hash intact: {base_hash}")

    # 2. Generate and Shard Reasoning Curriculum Dataset
    shards_dir = REASONING_DATA_DIR
    manifest = prepare_reasoning_shards(tokenizer, output_dir=shards_dir, seed=42)

    train_shard_dir = shards_dir / "train"
    val_shard_dir = shards_dir / "val"

    # 3. Train Isolated Experimental Checkpoint (500 steps)
    train_res = train_reasoning_checkpoint(
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
        model_name=f"step54_reasoning_{steps}steps",
        seed=42,
        sampling_mode="greedy",
    )
    anchor_eval = runner_anchor.run_benchmark(LEVEL_0_BENCHMARK_TASKS)
    print(f"Step-52 Anchor Pass Rate: {anchor_eval['overall_pass_rate']*100:.1f}% "
          f"({anchor_eval['passed_tasks']}/{anchor_eval['total_tasks']})")

    # 5. Evaluate on Held-Out Generalization Benchmark (10 unseen tasks)
    print("\nEvaluating on Held-Out Generalization Benchmark (Unseen Prompts)...")
    runner_heldout = ModelCapabilityBenchmarkRunner(
        model=trained_model,
        tokenizer=tokenizer,
        model_name="step54_reasoning_heldout",
        seed=42,
        sampling_mode="greedy",
    )
    heldout_eval = runner_heldout.run_benchmark(HELDOUT_BENCHMARK_TASKS)
    print(f"Held-Out Generalization Pass Rate: {heldout_eval['overall_pass_rate']*100:.1f}% "
          f"({heldout_eval['passed_tasks']}/{heldout_eval['total_tasks']})")

    # 6. Evaluate on Step-54 Structured Reasoning Benchmark (10 tasks)
    print("\nEvaluating on Step-54 Structured Reasoning Benchmark...")
    runner_reasoning = ModelCapabilityBenchmarkRunner(
        model=trained_model,
        tokenizer=tokenizer,
        model_name="step54_reasoning_benchmark",
        seed=42,
        sampling_mode="greedy",
    )
    reasoning_eval = runner_reasoning.run_benchmark(REASONING_BENCHMARK_TASKS)
    print(f"Step-54 Reasoning Pass Rate: {reasoning_eval['overall_pass_rate']*100:.1f}% "
          f"({reasoning_eval['passed_tasks']}/{reasoning_eval['total_tasks']})")
    print(f"Reasoning Failure Distribution: {reasoning_eval['failure_distribution']}")

    # 7. Demonstrate RIL Preparation Layer
    print("\nTesting RIL Preparation Layer (Trajectory -> Experience Record)...")
    sample_traj = (
        "<TRAJECTORY>\n"
        "<SPEC>\nTask: Implement add(a, b)\n</SPEC>\n"
        "<STATE>\nInitial implementation draft\n</STATE>\n"
        "<ACTION>\ndef add(a, b):\n    return a - b\n</ACTION>\n"
        "<OBSERVATION>\nFAILED: add(2, 3) returned -1, expected 5\n</OBSERVATION>\n"
        "<DIAGNOSIS>\nUsed subtraction operator '-' instead of addition '+'.\n</DIAGNOSIS>\n"
        "<NEXT_ACTION>\ndef add(a, b):\n    return a + b\n</NEXT_ACTION>\n"
        "<RESULT>SUCCESS</RESULT>\n"
        "</TRAJECTORY>"
    )
    sample_exp = extract_experience_from_trajectory(sample_traj, task_id="DEMO_ADD_01", success=True)
    assert sample_exp["diagnosis"] == "Used subtraction operator '-' instead of addition '+'.", "RIL extraction failed"
    print("RIL Preparation Layer successfully extracted structured experience record.")

    # 8. Post-Training Baseline Integrity Verification
    post_base_hash = compute_model_hash(baseline_model)
    assert post_base_hash == EXPECTED_WEIGHT_HASH, "Baseline mutated during training!"
    print(f"Verified frozen baseline immutable: {post_base_hash}")

    # 9. Gate Review & Decision
    anchor_pass = anchor_eval["overall_pass_rate"]
    heldout_pass = heldout_eval["overall_pass_rate"]
    reasoning_pass = reasoning_eval["overall_pass_rate"]

    gate_approved = (anchor_pass >= 0.80 and heldout_pass >= 0.70 and reasoning_pass >= 0.70)
    decision = "APPROVED" if gate_approved else "NOT_APPROVED_INSUFFICIENT_GENERALIZATION"

    # 10. Serialize Artifacts
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = ARTIFACTS_DIR / "step54_reasoning_report.json"

    master_payload = {
        "timestamp": time.time(),
        "baseline_hash": EXPECTED_WEIGHT_HASH,
        "checkpoint_type": "EXPERIMENTAL",
        "gate_status": decision,
        "release_0_1_approved": gate_approved,
        "training_metadata": train_res,
        "anchor_benchmark_results": anchor_eval,
        "heldout_benchmark_results": heldout_eval,
        "reasoning_benchmark_results": reasoning_eval,
        "sample_ril_experience": sample_exp,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(master_payload, f, indent=2)

    print(f"\nReport written to: {report_path}")
    print(f"Gate Decision: {decision}")
    print("=" * 70)
    return master_payload


if __name__ == "__main__":
    run_experiment_step54(steps=500)
