"""Dedicated JSON SFT V2 Trainer for Batch 3111–3150.

Loads exclusively from experiments/batch3111_3150/SFT_EXPERIMENT_CONFIG.json.

CRITICAL SAFETY INVARIANTS:
1. Runs full preflight validation before touching model or optimizer.
2. If any preflight gate fails, exits immediately with non-zero exit code.
3. Refuses execution unless explicitly called with --execute-authorized-training.
4. Uses exclusively experiments/batch3111_3150/train.jsonl for training and
   experiments/batch3111_3150/val.jsonl for validation.
5. experiments/batch3111_3150/test.jsonl is NEVER touched or passed to the trainer.
6. Checkpoints write exclusively to experiments/batch3111_3150/json_sft_v2_pilot_run/.
7. Canonical checkpoints and tokenizer are strictly protected against modification.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXPERIMENT_DIR = REPO_ROOT / "experiments" / "batch3111_3150"
CONFIG_PATH = EXPERIMENT_DIR / "SFT_EXPERIMENT_CONFIG.json"

from chakrview.training.sft_engine import SFTConfig, SFTExperimentEngine
from experiments.batch3111_3150.run_preflight import run_preflight


def run_training_pipeline(execute_training: bool = False) -> int:
    # 1. Enforce fail-closed preflight gates FIRST
    print("Executing preflight integrity gates prior to trainer initialization...")
    preflight_code = run_preflight()
    if preflight_code != 0:
        print("[GATE FAILURE] Preflight verification failed. Trainer will not initialize.")
        return 1

    # 2. Load and verify configuration
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Configuration missing: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # 3. Path & split isolation verification
    train_file = REPO_ROOT / cfg["dataset_paths"]["train_file"]
    val_file = REPO_ROOT / cfg["dataset_paths"]["val_file"]
    test_file = REPO_ROOT / cfg["dataset_paths"]["test_file"]
    out_dir = REPO_ROOT / cfg["output_directory"]
    base_ckpt = REPO_ROOT / cfg["parent_checkpoint"]["path"]

    assert "batch3111_3150" in str(train_file), f"Stale train dataset path: {train_file}"
    assert "batch3111_3150" in str(val_file), f"Stale val dataset path: {val_file}"
    assert "batch3111_3150" in str(out_dir), f"Unsafe output directory: {out_dir}"
    assert "main_training_10000" not in str(out_dir), "Protected directory violation: Cannot write to main_training_10000"
    assert "batch3071_3110" not in str(out_dir), "Cross-experiment directory violation"

    print("=================================================================")
    print("CHAKRVIEW BATCH 3111-3150 TRAINER INVOCATION CHECK")
    print("=================================================================")
    print(f"Config File:       {CONFIG_PATH}")
    print(f"Train File:        {train_file}")
    print(f"Val File:          {val_file}")
    print(f"Held-Out Test:     {test_file} (EXCLUDED FROM TRAINING)")
    print(f"Output Directory:  {out_dir}")
    print(f"Steps:             {cfg['training_hyperparameters']['pilot_steps']}")
    print(f"Batch Size:        {cfg['training_hyperparameters']['batch_size']}")
    print(f"Learning Rate:     {cfg['training_hyperparameters']['learning_rate']}")
    print(f"Seed:              {cfg['training_hyperparameters']['seed']}")
    print(f"Val Interval:      {cfg['training_hyperparameters']['val_interval']}")
    print(f"Save Interval:     {cfg['training_hyperparameters']['save_interval']}")
    print(f"Best Save Enabled: {cfg['training_hyperparameters']['save_best_checkpoint']}")
    print("=================================================================")

    if not execute_training:
        print("[SAFE PREPARATION MODE - NO OPTIMIZER STEPS EXECUTED]")
        print("Training is NOT authorized in this command invocation.")
        print("To execute training once authorized, invoke with --execute-authorized-training.")
        print("=================================================================")
        return 0

    print("[CRITICAL] AUTHORIZED TRAINING REQUEST RECEIVED.")
    out_dir.mkdir(parents=True, exist_ok=True)

    engine_cfg = SFTConfig(
        repo_root=REPO_ROOT,
        train_file=train_file,
        val_file=val_file,
        output_dir=out_dir,
        base_checkpoint=base_ckpt,
        steps=cfg["training_hyperparameters"]["pilot_steps"],
        batch_size=cfg["training_hyperparameters"]["batch_size"],
        max_seq_len=cfg["training_hyperparameters"]["max_seq_len"],
        learning_rate=cfg["training_hyperparameters"]["learning_rate"],
        min_learning_rate=cfg["training_hyperparameters"]["min_learning_rate"],
        weight_decay=cfg["training_hyperparameters"]["weight_decay"],
        grad_clip=cfg["training_hyperparameters"]["grad_clip"],
        val_interval=cfg["training_hyperparameters"]["val_interval"],
        save_interval=cfg["training_hyperparameters"]["save_interval"],
        seed=cfg["training_hyperparameters"]["seed"],
        device=cfg["training_hyperparameters"]["device"],
        tokenizer_dir=str(REPO_ROOT / cfg["tokenizer"]["directory"]),
        strip_prompt_delimiters=cfg["training_hyperparameters"]["strip_prompt_delimiters"],
        allow_exploratory_opt_in=cfg["training_hyperparameters"]["allow_exploratory_opt_in"],
        enforce_trust_gate=cfg["training_hyperparameters"]["enforce_trust_gate"],
        save_best_checkpoint=cfg["training_hyperparameters"]["save_best_checkpoint"],
    )

    engine = SFTExperimentEngine(engine_cfg)
    results = engine.train()
    print("Training successfully finished.")
    print(f"Final Val Loss: {results['final_val_loss']}, Perplexity: {results['final_val_perplexity']}")

    summary_path = out_dir / "sft_training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return 0


def main():
    parser = argparse.ArgumentParser(description="ChakrView JSON SFT V2 Trainer (Batch 3111-3150)")
    parser.add_argument("--execute-authorized-training", action="store_true", help="Explicit gate required to execute optimizer steps")
    args = parser.parse_args()

    exit_code = run_training_pipeline(execute_training=args.execute_authorized_training)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
