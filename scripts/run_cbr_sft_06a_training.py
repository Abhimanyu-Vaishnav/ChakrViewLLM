"""
scripts/run_cbr_sft_06a_training.py

Autonomous Controlled Training Runner for CBR-SFT-06A:
- Model Architecture: ChakrMicro (4 layers, 8 heads, d_model=512, context 512)
- Parent Checkpoint: CBR-SFT-04 Step 200 (327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153)
- Dataset: data/sft/cbr_sft_06a/splits/cbr_sft_06a_train_153.jsonl (153 items)
- Validation: data/sft/cbr_sft_06a/splits/cbr_sft_06a_val_17.jsonl (17 items)
- Steps: 200
- Micro-batch: 2, Grad Accum: 2 (Effective Batch: 4)
- Learning Rate: 1.0e-4 (cosine decay to 1.0e-5)
- Weight Decay: 0.01, Grad Clip: 1.0
- Checkpoints saved at steps: 25, 50, 75, 100, 150, 200.
"""

import sys
import os
import json
import time
import math
import hashlib
from pathlib import Path
from typing import Dict, Any, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
from chakrview.training.sft_engine import SFTConfig, SFTExperimentEngine
from chakrview.training.safety import assert_safe_checkpoint_target

def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def main():
    print("=" * 70)
    print("CHAKRVIEW — CBR-SFT-06A CONTROLLED TRAINING RUNNER")
    print("=" * 70)

    # 1. Non-negotiable Preflight Verification
    parent_ckpt_path = REPO_ROOT / "experiments" / "cbr01_sft" / "runs" / "cbr_sft_04" / "checkpoints" / "sft_checkpoint_step_200.pt"
    expected_parent_sha256 = "327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153"

    if not parent_ckpt_path.exists():
        raise FileNotFoundError(f"Parent checkpoint missing: {parent_ckpt_path}")

    actual_parent_sha256 = file_sha256(parent_ckpt_path)
    print(f"Parent checkpoint: {parent_ckpt_path}")
    print(f"Parent SHA-256:    {actual_parent_sha256}")
    assert actual_parent_sha256 == expected_parent_sha256, (
        f"Parent hash mismatch! Expected {expected_parent_sha256}, got {actual_parent_sha256}"
    )
    print("Parent checkpoint lineage verified: PASS")

    # 2. Setup Isolated Output Directory Structure
    run_dir = REPO_ROOT / "experiments" / "cbr01_sft" / "runs" / "cbr_sft_06a"
    ckpt_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    metrics_dir = run_dir / "metrics"
    config_dir = run_dir / "config"
    reports_dir = run_dir / "reports"

    assert_safe_checkpoint_target(run_dir)
    for d in [ckpt_dir, logs_dir, metrics_dir, config_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)
        assert_safe_checkpoint_target(d)

    # 3. Configure SFT Training
    train_file = REPO_ROOT / "data" / "sft" / "cbr_sft_06a" / "splits" / "cbr_sft_06a_train_153.jsonl"
    val_file = REPO_ROOT / "data" / "sft" / "cbr_sft_06a" / "splits" / "cbr_sft_06a_val_17.jsonl"

    assert train_file.exists(), f"Train file missing: {train_file}"
    assert val_file.exists(), f"Val file missing: {val_file}"

    sft_cfg = SFTConfig(
        repo_root=REPO_ROOT,
        train_file=train_file,
        val_file=val_file,
        output_dir=ckpt_dir,
        base_checkpoint=parent_ckpt_path,
        steps=200,
        batch_size=2,
        grad_accum_steps=2,
        max_seq_len=512,
        learning_rate=1.0e-4,
        min_learning_rate=1.0e-5,
        weight_decay=0.01,
        grad_clip=1.0,
        val_interval=10,
        save_interval=25,
        seed=42,
        device="cpu",
        tokenizer_dir=str(REPO_ROOT / "data" / "experiments" / "vocab_4096"),
        mask_prompt=True,
        allow_exploratory_opt_in=False,
        enforce_trust_gate=True,
        strip_prompt_delimiters=False
    )

    # Save TRAINING_CONFIG.json
    cfg_record = {
        "experiment_id": "CBR_SFT_06A",
        "parent_checkpoint": str(parent_ckpt_path),
        "parent_checkpoint_sha256": actual_parent_sha256,
        "dataset": {
            "train_file": str(train_file),
            "train_sha256": file_sha256(train_file),
            "val_file": str(val_file),
            "val_sha256": file_sha256(val_file)
        },
        "training": {
            "steps": sft_cfg.steps,
            "micro_batch_size": sft_cfg.batch_size,
            "grad_accum_steps": sft_cfg.grad_accum_steps,
            "effective_batch_size": sft_cfg.batch_size * sft_cfg.grad_accum_steps,
            "learning_rate": sft_cfg.learning_rate,
            "min_learning_rate": sft_cfg.min_learning_rate,
            "weight_decay": sft_cfg.weight_decay,
            "grad_clip": sft_cfg.grad_clip,
            "val_interval": sft_cfg.val_interval,
            "save_interval": sft_cfg.save_interval,
            "seed": sft_cfg.seed,
            "device": sft_cfg.device,
            "mask_prompt": sft_cfg.mask_prompt,
            "strip_prompt_delimiters": sft_cfg.strip_prompt_delimiters,
            "max_seq_len": sft_cfg.max_seq_len
        },
        "output_dir": str(run_dir)
    }

    with open(config_dir / "TRAINING_CONFIG.json", "w", encoding="utf-8") as f:
        json.dump(cfg_record, f, indent=2)

    with open(run_dir / "TRAINING_CONFIG.json", "w", encoding="utf-8") as f:
        json.dump(cfg_record, f, indent=2)

    # 4. Instantiate Engine & Reset start_step for Child Experiment
    engine = SFTExperimentEngine(sft_cfg)

    # CBR-SFT-06A starts at step 0 for its own 200 steps
    engine.start_step = 0
    engine.history = []

    print("\nStarting training loop (Steps 1 to 200)...")
    summary = engine.train()
    print("Training loop completed successfully!")

    # 5. Save Metrics & Training Summary
    with open(metrics_dir / "training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(run_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # 6. Verify and Hash Saved Checkpoints (steps: 25, 50, 75, 100, 150, 200)
    saved_checkpoints = {}
    for step in [25, 50, 75, 100, 150, 200]:
        ckpt_p = ckpt_dir / f"sft_checkpoint_step_{step}.pt"
        if not ckpt_p.exists():
            raise FileNotFoundError(f"Expected checkpoint not found: {ckpt_p}")
        c_hash = file_sha256(ckpt_p)
        saved_checkpoints[f"step_{step}"] = {
            "step": step,
            "path": str(ckpt_p),
            "sha256": c_hash,
            "size_bytes": ckpt_p.stat().st_size
        }
        print(f"Checkpoint step {step:3d}: SHA256={c_hash}")

    with open(run_dir / "checkpoint_manifest.json", "w", encoding="utf-8") as f:
        json.dump(saved_checkpoints, f, indent=2)

    print("\nTraining execution and checkpoint verification complete!")

if __name__ == "__main__":
    main()

