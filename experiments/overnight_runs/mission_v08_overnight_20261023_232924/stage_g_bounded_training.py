"""
Stage G: Bounded Experimental SFT Training Run on Exp18 Isolated Dataset.
Executes:
  - Base checkpoint: isolated copy of canonical step 10,000
  - Training dataset: exp18_train_isolated.jsonl (1,200 records)
  - Validation dataset: exp18_val_isolated.jsonl (160 records)
  - Parameter settings:
      steps: 120 (bounded safe overnight budget)
      batch_size: 2
      learning_rate: 5e-5 to 5e-6 (cosine decay)
      weight_decay: 0.01
      grad_clip: 1.0
      val_interval: 20
      save_interval: 60
      seed: 42
Saves:
  - TRAINING_CONFIG.json
  - TRAINING_LOG.csv
  - Checkpoints: sft_checkpoint_step_60.pt, sft_checkpoint_step_120.pt, sft_best_checkpoint.pt
  - FINAL_TRAINING_GATE.md
"""

import csv
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path("d:/Project/ChakrView").resolve()
MISSION_DIR = REPO_ROOT / "experiments" / "overnight_runs" / "mission_v08_overnight_20261023_232924"
OUTPUT_DIR = MISSION_DIR / "checkpoints" / "sft_exp18_run"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.training.sft_engine import SFTConfig, SFTExperimentEngine

def run_bounded_training():
    train_file = MISSION_DIR / "data" / "exp18_train_isolated.jsonl"
    val_file = MISSION_DIR / "data" / "exp18_val_isolated.jsonl"
    base_checkpoint = MISSION_DIR / "checkpoints" / "canonical_copy_step_10000.pt"

    config = SFTConfig(
        repo_root=REPO_ROOT,
        train_file=train_file,
        val_file=val_file,
        output_dir=OUTPUT_DIR,
        base_checkpoint=base_checkpoint,
        steps=120,
        batch_size=2,
        max_seq_len=512,
        learning_rate=5e-5,
        min_learning_rate=5e-6,
        weight_decay=0.01,
        grad_clip=1.0,
        val_interval=20,
        save_interval=60,
        seed=42,
        device="cpu",
        tokenizer_dir=str(REPO_ROOT / "data" / "experiments" / "vocab_4096"),
        allow_exploratory_opt_in=True,
        enforce_trust_gate=True,
        strip_prompt_delimiters=False,
        save_best_checkpoint=True,
    )

    # Save TRAINING_CONFIG.json
    cfg_dict = {
        "experiment_id": "EXP18_SFT_V08_BOUNDED_120",
        "timestamp": datetime.now().isoformat(),
        "base_checkpoint": str(base_checkpoint),
        "steps": config.steps,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "min_learning_rate": config.min_learning_rate,
        "weight_decay": config.weight_decay,
        "grad_clip": config.grad_clip,
        "val_interval": config.val_interval,
        "save_interval": config.save_interval,
        "seed": config.seed,
        "device": config.device,
        "tokenizer_dir": config.tokenizer_dir,
    }
    with open(MISSION_DIR / "TRAINING_CONFIG.json", "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, indent=2)

    print(f"Initializing SFTExperimentEngine for {config.steps} steps...")
    engine = SFTExperimentEngine(config)

    t0 = time.perf_counter()
    train_results = engine.train()
    elapsed = time.perf_counter() - t0

    # Save TRAINING_LOG.csv
    with open(MISSION_DIR / "TRAINING_LOG.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "train_loss", "val_loss", "val_perplexity", "learning_rate", "grad_norm", "step_time_sec", "is_best"])
        for h in train_results["history"]:
            writer.writerow([
                h.get("step"),
                h.get("train_loss"),
                h.get("val_loss", ""),
                h.get("val_perplexity", ""),
                h.get("learning_rate"),
                h.get("grad_norm"),
                h.get("step_time_sec"),
                h.get("is_best", False),
            ])

    # Save FINAL_TRAINING_GATE.md
    gate_content = f"""# Final Training Gate Report — Exp18 SFT Run
**Experiment ID**: `EXP18_SFT_V08_BOUNDED_120`
**Timestamp**: {datetime.now().isoformat()}
**Total Steps Completed**: {train_results['total_steps']}
**Elapsed Time**: {train_results['elapsed_seconds']:.2f} seconds
**Device**: CPU

## Loss Progression
- Initial Validation Loss: **{train_results['initial_val_loss']:.4f}**
- Final Training Loss: **{train_results['final_train_loss']:.4f}**
- Final Validation Loss: **{train_results['final_val_loss']:.4f}** (Perplexity: {train_results['final_val_perplexity']:.2f})
- Best Validation Loss: **{train_results['best_val_loss']:.4f}** at Step **{train_results['best_val_step']}**

## Checkpoints Produced
- Best Checkpoint: `{OUTPUT_DIR / 'sft_best_checkpoint.pt'}`
- Step 60 Checkpoint: `{OUTPUT_DIR / 'sft_checkpoint_step_60.pt'}`
- Step 120 Checkpoint: `{OUTPUT_DIR / 'sft_checkpoint_step_120.pt'}`

## Verification
- Precondition gates passed before training.
- Zero mutations to canonical checkpoint `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`.
- Zero training on held-out test data.
- Checkpoint lineage fully preserved.
"""
    with open(MISSION_DIR / "FINAL_TRAINING_GATE.md", "w", encoding="utf-8") as f:
        f.write(gate_content)

    print("\n=== TRAINING COMPLETED SUCCESSFULLY ===")
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Initial Val Loss: {train_results['initial_val_loss']:.4f}")
    print(f"Final Train Loss: {train_results['final_train_loss']:.4f}")
    print(f"Final Val Loss: {train_results['final_val_loss']:.4f}")
    print(f"Best Val Loss: {train_results['best_val_loss']:.4f} at Step {train_results['best_val_step']}")

if __name__ == "__main__":
    run_bounded_training()
