# SFT V3 Pilot Run Execution Artifacts

This directory contains the artifacts, checkpoint weights, execution reports, and comparative evaluation results from the single authorized 100-step SFT V3 pilot.

## Directory Contents

- `README.md`: Overview of the run, reproduction commands, and directory inventory.
- `PILOT_EXECUTION_REPORT.md`: Comprehensive evaluation report analyzing loss dynamics, three-layer capability measurements, root causes, and final verdict.
- `RUN_MANIFEST.json`: Cryptographic hashes, hyperparameters, training step count, and safety gate confirmations.
- `TRAINING_METRICS.json`: Step-by-step training loss, validation loss, active response tokens, gradient norms, and learning rates.
- `EVALUATION_RESULTS.json`: Detailed comparative evaluation results comparing the canonical base checkpoint vs the pilot candidate across all 40 validation records.
- `evaluate_comparative.py`: Evaluation script used to generate comparative metrics.
- `sft_best_checkpoint.pt`: Best validation checkpoint saved at step 100 (Val loss: 3.3215).
- `sft_checkpoint_step_*.pt`: Periodic checkpoints at steps 20, 40, 50, 60, 80, and 100.

## Hard Safety Confirmation
- Exactly ONE training run was executed.
- Exactly 100 optimizer steps were performed.
- `chakrview/runtime/session.py` has ZERO modifications.
- Canonical Step-10,000 checkpoint and tokenizers remain 100% bit-identical.
- Held-out test split (`test.jsonl`) was NEVER accessed or inspected.
- The candidate checkpoint is marked `NO MEANINGFUL CAPABILITY IMPROVEMENT` and will NOT be promoted to release.
