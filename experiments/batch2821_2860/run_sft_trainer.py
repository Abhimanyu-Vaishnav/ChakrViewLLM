"""Reusable SFT Trainer for Batch 2821–2860.

Configured to load the canonical Step-10,000 checkpoint and output to
experiments/batch2821_2860/sft_pilot_run/.

SAFETY INVARIANT:
Will refuse to execute training unless explicitly called with --execute-authorized-training.
Without that flag, prints configuration and verifies setup cleanly with exit code 0.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.training.sft_engine import SFTConfig, SFTDataset, SFTExperimentEngine

logger = logging.getLogger("sft_trainer")


def inspect_or_train(config_path: Path, execute_training: bool = False):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    print("=================================================================")
    print("CHAKRVIEW SFT TRAINER: BATCH 2821-2860")
    print("=================================================================")
    print(f"Parent Checkpoint: {cfg['parent_checkpoint']['path']}")
    print(f"Parent SHA-256:    {cfg['parent_checkpoint']['expected_sha256']}")
    print(f"Output Directory:  {cfg['output_directory']}")
    print(f"Pilot Steps:       {cfg['training_hyperparameters']['pilot_steps']}")
    print(f"Batch Size:        {cfg['training_hyperparameters']['batch_size']}")
    print(f"Learning Rate:     {cfg['training_hyperparameters']['learning_rate']} (min {cfg['training_hyperparameters']['min_learning_rate']})")
    print(f"Seed:              {cfg['training_hyperparameters']['seed']}")
    print("-----------------------------------------------------------------")

    # Safety checks
    out_dir = REPO_ROOT / cfg["output_directory"]
    if "main_training_10000" in str(out_dir) or "artifacts" in str(out_dir):
        raise ValueError(f"Protected directory violation: Output directory cannot point to canonical artifacts: {out_dir}")

    if not execute_training:
        print("[SAFE PREPARATION MODE - NO OPTIMIZER STEPS EXECUTED]")
        print("Training is NOT authorized in this command invocation.")
        print("To execute training once authorized, invoke with --execute-authorized-training.")
        print("=================================================================")
        return

    print("Executing AUTHORIZED SFT training run...")
    out_dir.mkdir(parents=True, exist_ok=True)
    engine_cfg = SFTConfig(
        repo_root=REPO_ROOT,
        train_file=REPO_ROOT / cfg["dataset_paths"]["train_file"],
        val_file=REPO_ROOT / cfg["dataset_paths"]["val_file"],
        output_dir=out_dir,
        base_checkpoint=REPO_ROOT / cfg["parent_checkpoint"]["path"],
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
    )

    engine = SFTExperimentEngine(engine_cfg)
    results = engine.train()
    print("Training successfully finished.")
    print(f"Final Val Loss: {results['final_val_loss']}, Perplexity: {results['final_val_perplexity']}")
    summary_path = out_dir / "sft_training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="ChakrView SFT Trainer")
    parser.add_argument("--config", type=str, default="experiments/batch2821_2860/SFT_TRAINING_CONFIG.json")
    parser.add_argument("--execute-authorized-training", action="store_true", help="Explicit gate required to execute optimizer steps")
    args = parser.parse_args()

    config_path = REPO_ROOT / args.config
    inspect_or_train(config_path, execute_training=args.execute_authorized_training)


if __name__ == "__main__":
    main()
