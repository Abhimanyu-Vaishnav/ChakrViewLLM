"""Dedicated SFT V3 Pilot Training Script.

Executes exactly ONE authorized 100-step CPU training run using the canonical SFT engine.

HARD SAFETY INVARIANTS:
1. Verifies all preflight integrity gates prior to any optimizer step.
2. Uses strictly experiments/sft_v3_dataset_builder/train.jsonl and validation.jsonl.
3. Base checkpoint: canonical Step-10,000 checkpoint (96357705...).
4. Tokenizer: data/experiments/vocab_4096 (vocab: db4d6119..., merges: ada4b1dc...).
5. Max steps: strictly 100 steps.
6. Checkpoints save strictly to experiments/sft_v3_pilot_readiness/pilot_run_v3/.
7. Never touches test.jsonl.
8. Requires --execute-authorized-pilot flag to initiate training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.training.sft_engine import SFTConfig, SFTExperimentEngine

RUN_DIR = REPO_ROOT / "experiments" / "sft_v3_pilot_readiness" / "pilot_run_v3"
TRAIN_FILE = REPO_ROOT / "experiments" / "sft_v3_dataset_builder" / "train.jsonl"
VAL_FILE = REPO_ROOT / "experiments" / "sft_v3_dataset_builder" / "validation.jsonl"
BASE_CKPT = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"
TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
MANIFEST_PATH = REPO_ROOT / "experiments" / "sft_v3_dataset_builder" / "DATASET_MANIFEST.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_preflight_gates() -> None:
    print("Verifying preflight gates before training...")

    # 1. Canonical Checkpoint
    ck_hash = sha256_file(BASE_CKPT)
    expected_ck = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"
    assert ck_hash == expected_ck, f"Checkpoint hash mismatch: {ck_hash} != {expected_ck}"

    # 2. Canonical Tokenizer
    vb_hash = sha256_file(TOKENIZER_DIR / "vocab.json")
    mg_hash = sha256_file(TOKENIZER_DIR / "merges.json")
    expected_vb = "db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a"
    expected_mg = "ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7"
    assert vb_hash == expected_vb, f"Vocab hash mismatch: {vb_hash} != {expected_vb}"
    assert mg_hash == expected_mg, f"Merges hash mismatch: {mg_hash} != {expected_mg}"

    # 3. Datasets against Manifest
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        m = json.load(f)
    tr_hash = sha256_file(TRAIN_FILE)
    val_hash = sha256_file(VAL_FILE)
    assert tr_hash == m["splits"]["train"]["sha256"], "Train hash mismatch against manifest!"
    assert val_hash == m["splits"]["validation"]["sha256"], "Val hash mismatch against manifest!"

    # 4. Protected file session.py diff
    diff = subprocess.check_output(["git", "diff", "--name-only", "chakrview/runtime/session.py"]).decode("utf-8").strip()
    assert diff == "", f"Protected file chakrview/runtime/session.py is modified: {diff}"

    print("All preflight gates passed 100%.")


def execute_sft_v3_pilot():
    verify_preflight_gates()
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("CHAKRVIEW SFT V3 AUTHORIZED 100-STEP PILOT EXECUTION")
    print("=================================================================")
    print(f"Base Checkpoint:   {BASE_CKPT}")
    print(f"Train File:        {TRAIN_FILE}")
    print(f"Val File:          {VAL_FILE}")
    print(f"Tokenizer Dir:     {TOKENIZER_DIR}")
    print(f"Output Directory:  {RUN_DIR}")
    print(f"Steps:             100")
    print(f"Batch Size:        4")
    print(f"Learning Rate:     3e-5 (min: 1e-6)")
    print(f"Device:            cpu")
    print(f"Seed:              42")
    print("=================================================================")

    cfg = SFTConfig(
        repo_root=REPO_ROOT,
        train_file=TRAIN_FILE,
        val_file=VAL_FILE,
        output_dir=RUN_DIR,
        base_checkpoint=BASE_CKPT,
        steps=100,
        batch_size=4,
        max_seq_len=512,
        learning_rate=3e-5,
        min_learning_rate=1e-6,
        weight_decay=0.01,
        grad_clip=1.0,
        val_interval=20,
        save_interval=50,
        seed=42,
        device="cpu",
        tokenizer_dir=str(TOKENIZER_DIR),
        strip_prompt_delimiters=False,
        allow_exploratory_opt_in=False,
        enforce_trust_gate=True,
        save_best_checkpoint=True,
    )

    engine = SFTExperimentEngine(cfg)
    results = engine.train()

    print("\n=================================================================")
    print("SFT V3 PILOT EXECUTION COMPLETED")
    print("=================================================================")
    print(f"Initial Val Loss:       {results['initial_val_loss']}")
    print(f"Final Train Loss:       {results['final_train_loss']}")
    print(f"Final Val Loss:         {results['final_val_loss']} (Perplexity: {results['final_val_perplexity']})")
    print(f"Best Val Loss:          {results['best_val_loss']} at step {results['best_val_step']}")
    print(f"Elapsed Seconds:        {results['elapsed_seconds']}s")
    print("=================================================================")

    # Write metrics to isolated run dir
    metrics_path = RUN_DIR / "TRAINING_METRICS.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute SFT V3 100-step Pilot")
    parser.add_argument(
        "--execute-authorized-pilot",
        action="store_true",
        help="Explicit flag required to initiate the single authorized 100-step training run",
    )
    args = parser.parse_args()

    if not args.execute_authorized_pilot:
        print("[PRE-FLIGHT CHECK ONLY - NO TRAINING EXECUTED]")
        verify_preflight_gates()
        print("To execute the authorized 100-step pilot, invoke with --execute-authorized-pilot.")
        sys.exit(0)

    execute_sft_v3_pilot()
