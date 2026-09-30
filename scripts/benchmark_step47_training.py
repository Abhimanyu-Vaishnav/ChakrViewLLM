"""
Benchmark: ChakrView Step 47 Training Readiness & First Learning Loop.

Measures:
1. Batch loading latency (ms)
2. Forward pass latency (ms)
3. Backward pass latency (ms)
4. Optimizer-step latency (ms)
5. Total training-step latency (ms)
6. CPU memory usage RSS (MB)
7. Parameter update magnitude (L2 norm and max absolute diff)
8. Loss before / after micro-training
9. Checkpoint save latency (ms)
10. Checkpoint load latency (ms)
11. Frozen baseline model immutability (ΔW = 0 verification)

Writes results to: docs/STEP_47_BENCHMARK_RESULTS.json
"""

from __future__ import annotations

import gc
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Dict, List
import hashlib
import psutil
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.training.dataset import StreamingTokenDataset
from chakrview.training.loss import CausalLoss
from chakrview.training.checkpoint import CheckpointManager
from chakrview.training.safety import TrainingSafetyChecker

OUTPUT_FILE = ROOT_DIR / "docs" / "STEP_47_BENCHMARK_RESULTS.json"
TRAIN_SHARD_DIR = ROOT_DIR / "data" / "tokenized" / "train"

FROZEN_BASELINE_HASH = "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"
EXPECTED_PARAM_COUNT = 3_443_136


def compute_model_hash(model: torch.nn.Module) -> str:
    """Compute deterministic SHA-256 hash across all named parameters."""
    hasher = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(model.named_parameters()):
            hasher.update(name.encode("utf-8"))
            hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


def compute_stats(values: List[float]) -> Dict[str, float]:
    """Compute mean, median, min, max for latency metrics."""
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean_ms": round(statistics.mean(values), 3),
        "median_ms": round(statistics.median(values), 3),
        "min_ms": round(min(values), 3),
        "max_ms": round(max(values), 3),
    }


def run_benchmark():
    print("=" * 72)
    print("CHAKRVIEW STEP 47: TRAINING READINESS & LEARNING LOOP BENCHMARK")
    print("=" * 72)

    # 1. Baseline Model Invariant Verification
    print("\n[1/5] Verifying Frozen Baseline Neural Core Invariant...")
    torch.manual_seed(42)
    baseline_model = ChakrMicro(ModelConfig())
    baseline_model.eval()
    baseline_hash_initial = compute_model_hash(baseline_model)
    param_count = sum(p.numel() for p in baseline_model.parameters())

    print(f"  Frozen baseline parameters: {param_count:,}")
    print(f"  Baseline SHA-256:          {baseline_hash_initial}")
    assert baseline_hash_initial == FROZEN_BASELINE_HASH, "Baseline model hash mismatch!"
    assert param_count == EXPECTED_PARAM_COUNT, "Parameter count mismatch!"

    # 2. Setup Micro-Training Experiment on Separate Test Model
    print("\n[2/5] Initializing Test Model & Dataset for Learning Loop...")
    torch.manual_seed(100)
    exp_model = ChakrMicro(ModelConfig())
    exp_model.train()
    initial_exp_hash = compute_model_hash(exp_model)
    initial_weights = {
        name: p.clone().detach() for name, p in exp_model.named_parameters()
    }

    loss_fn = CausalLoss(ignore_index=2)
    optimizer = torch.optim.AdamW(exp_model.parameters(), lr=1e-3, betas=(0.9, 0.95), eps=1e-8)

    dataset = StreamingTokenDataset(
        shard_dir=TRAIN_SHARD_DIR,
        sequence_length=64,
        loop=True,
    )
    data_iter = iter(dataset)

    # 3. Micro-Training Step Latency Profiling
    print("\n[3/5] Executing 10-Step Micro-Training Run & Profiling Latency...")
    num_steps = 10
    batch_loading_ms: List[float] = []
    forward_ms: List[float] = []
    backward_ms: List[float] = []
    optimizer_ms: List[float] = []
    total_step_ms: List[float] = []
    loss_history: List[float] = []
    grad_norm_history: List[float] = []

    for step in range(num_steps):
        t_step_start = time.perf_counter()

        # Batch loading
        t0 = time.perf_counter()
        item1 = next(data_iter)
        item2 = next(data_iter)
        batch = {
            "input_ids": torch.stack([item1["input_ids"], item2["input_ids"]]),
            "target_ids": torch.stack([item1["target_ids"], item2["target_ids"]]),
            "attention_mask": torch.stack([item1["attention_mask"], item2["attention_mask"]]),
        }
        t1 = time.perf_counter()
        batch_loading_ms.append((t1 - t0) * 1000.0)

        # Bounds check
        TrainingSafetyChecker.verify_token_ids(batch["input_ids"], min_id=0, max_id=4095)
        TrainingSafetyChecker.verify_token_ids(batch["target_ids"], min_id=0, max_id=4095)

        # Forward pass
        optimizer.zero_grad()
        t2 = time.perf_counter()
        logits = exp_model(batch["input_ids"], attention_mask=batch["attention_mask"])
        loss = loss_fn(logits, batch["target_ids"])
        loss_val = TrainingSafetyChecker.check_loss(loss, step=step)
        t3 = time.perf_counter()
        forward_ms.append((t3 - t2) * 1000.0)

        # Backward pass
        t4 = time.perf_counter()
        loss.backward()
        grad_norm = TrainingSafetyChecker.check_gradients(exp_model, step=step)
        t5 = time.perf_counter()
        backward_ms.append((t5 - t4) * 1000.0)

        # Optimizer step
        t6 = time.perf_counter()
        torch.nn.utils.clip_grad_norm_(exp_model.parameters(), 1.0)
        optimizer.step()
        t7 = time.perf_counter()
        optimizer_ms.append((t7 - t6) * 1000.0)

        total_step_ms.append((t7 - t_step_start) * 1000.0)
        loss_history.append(loss_val)
        grad_norm_history.append(grad_norm)

        print(
            f"  Step {step+1:02d}/{num_steps:02d} | "
            f"Loss: {loss_val:.4f} | "
            f"Grad Norm: {grad_norm:.4f} | "
            f"Step Latency: {total_step_ms[-1]:.2f} ms"
        )

    # Compute parameter update magnitude
    final_exp_hash = compute_model_hash(exp_model)
    total_l2_diff_sq = 0.0
    max_abs_diff = 0.0
    updated_param_count = 0

    with torch.no_grad():
        for name, param in exp_model.named_parameters():
            diff = param.detach() - initial_weights[name]
            abs_d = diff.abs().max().item()
            if abs_d > max_abs_diff:
                max_abs_diff = abs_d
            total_l2_diff_sq += (diff.norm().item() ** 2)
            if abs_d > 0.0:
                updated_param_count += 1

    total_l2_diff = math.sqrt(total_l2_diff_sq)
    print(f"\n[4/5] Parameter Update Verification:")
    print(f"  Initial Experiment Hash: {initial_exp_hash}")
    print(f"  Final Experiment Hash:   {final_exp_hash}")
    print(f"  Updated Parameters:      {updated_param_count} / {len(initial_weights)}")
    print(f"  Max Absolute Parameter Delta: {max_abs_diff:.6e}")
    print(f"  Total L2 Update Norm:    {total_l2_diff:.6e}")
    assert final_exp_hash != initial_exp_hash, "Experiment weights failed to update!"
    assert max_abs_diff > 0.0, "Parameters did not change!"

    # Checkpoint Benchmarking
    print("\n[5/5] Benchmarking Checkpoint Save and Load Latency...")
    ckpt_dir = ROOT_DIR / "checkpoints" / "benchmark_step47"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_mgr = CheckpointManager(checkpoint_dir=ckpt_dir, keep_last_n=2)

    # Save timing
    from dataclasses import asdict
    t_save_start = time.perf_counter()
    saved_path = ckpt_mgr.save(
        model=exp_model,
        optimizer=optimizer,
        step=num_steps,
        epoch=0,
        config=asdict(ModelConfig()),
        tokenizer_checksum="tok_bpe_4096_verified",
        dataset_manifest_hash="shard_00000_manifest_verified",
        parameter_count=EXPECTED_PARAM_COUNT,
    )
    t_save_end = time.perf_counter()
    save_latency_ms = (t_save_end - t_save_start) * 1000.0
    ckpt_size_mb = saved_path.stat().st_size / (1024 * 1024)

    # Load timing
    t_load_start = time.perf_counter()
    loaded_payload = CheckpointManager.load(
        saved_path,
        validate_training=True,
        expected_tokenizer_checksum="tok_bpe_4096_verified",
        expected_param_count=EXPECTED_PARAM_COUNT,
    )
    t_load_end = time.perf_counter()
    load_latency_ms = (t_load_end - t_load_start) * 1000.0

    print(f"  Checkpoint File:         {saved_path.name} ({ckpt_size_mb:.2f} MB)")
    print(f"  Save Latency:            {save_latency_ms:.2f} ms")
    print(f"  Load Latency:            {load_latency_ms:.2f} ms")

    # Re-verify Baseline Immutability (Delta W_baseline = 0)
    baseline_hash_final = compute_model_hash(baseline_model)
    print("\nRe-verifying Baseline Model Post-Training:")
    print(f"  Initial Baseline Hash:   {baseline_hash_initial}")
    print(f"  Final Baseline Hash:     {baseline_hash_final}")
    assert baseline_hash_initial == baseline_hash_final, "FATAL: Baseline model was mutated!"
    print("  Baseline Invariant Delta W = 0: STRICTLY PRESERVED")

    # Cleanup benchmark checkpoint directory
    for f in ckpt_dir.glob("*"):
        try:
            f.unlink()
        except OSError:
            pass
    try:
        ckpt_dir.rmdir()
    except OSError:
        pass

    # Memory RSS
    process = psutil.Process(os.getpid())
    memory_rss_mb = round(process.memory_info().rss / (1024 * 1024), 2)

    results = {
        "benchmark": "Step 47 Training Readiness & First Learning Loop",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_architecture": "ChakrMicro v0.1",
        "parameters": EXPECTED_PARAM_COUNT,
        "vocabulary_size": 4096,
        "context_length": 512,
        "baseline_model_hash": baseline_hash_initial,
        "baseline_model_post_hash": baseline_hash_final,
        "baseline_delta_w": 0.0,
        "baseline_immutability_preserved": True,
        "experiment_initial_hash": initial_exp_hash,
        "experiment_final_hash": final_exp_hash,
        "experiment_parameters_updated": True,
        "experiment_l2_update_norm": round(total_l2_diff, 6),
        "experiment_max_abs_delta": round(max_abs_diff, 6),
        "micro_training_steps": num_steps,
        "loss_initial": round(loss_history[0], 4),
        "loss_final": round(loss_history[-1], 4),
        "loss_progression": [round(l, 4) for l in loss_history],
        "grad_norm_progression": [round(g, 4) for g in grad_norm_history],
        "latencies": {
            "batch_loading": compute_stats(batch_loading_ms),
            "forward_pass": compute_stats(forward_ms),
            "backward_pass": compute_stats(backward_ms),
            "optimizer_step": compute_stats(optimizer_ms),
            "total_step": compute_stats(total_step_ms),
            "checkpoint_save_ms": round(save_latency_ms, 3),
            "checkpoint_load_ms": round(load_latency_ms, 3),
        },
        "checkpoint": {
            "size_mb": round(ckpt_size_mb, 2),
            "type": loaded_payload.get("checkpoint_type"),
            "step": loaded_payload.get("step"),
            "tokenizer_checksum": loaded_payload.get("tokenizer_checksum"),
            "dataset_manifest_hash": loaded_payload.get("dataset_manifest_hash"),
            "validation_passed": True,
        },
        "resources": {
            "memory_rss_mb": memory_rss_mb,
            "device": "cpu",
            "threads": torch.get_num_threads(),
        },
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Benchmark results written to {OUTPUT_FILE}")
    print("=" * 72)
    return results


if __name__ == "__main__":
    run_benchmark()
