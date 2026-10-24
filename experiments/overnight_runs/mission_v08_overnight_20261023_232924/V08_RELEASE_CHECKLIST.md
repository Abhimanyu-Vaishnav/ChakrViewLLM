# ChakrView v0.8 Release Readiness Checklist

## 1. Scope and Core Project Principles
- [x] **Indigenous Neural Core**: 100% trained from scratch without external weights, APIs, or hosted LLMs.
- [x] **CPU-First Architecture**: Pure CPU PyTorch implementation (`torch.device("cpu")`), no GPU dependencies.
- [x] **Canonical Architecture Fidelity**: 4,096 vocab, 192 d_model, 6 layers, 6 heads, 512 hidden_dim, 512 context, tied LM head (3,443,136 parameters).
- [x] **Protected Artifact Immutability**: All 8 protected files (canonical checkpoint, tokenizer vocab/merges, session.py, model.py, Exp17 train/val/test) verified 100% bitwise matching before and after all experiments.

## 2. Dataset & Evaluator Integrity (Exp18)
- [x] **In-Memory Quota & Schema**: 1,520 records synthesized (1,200 train, 160 val, 160 test) across 8 balanced categories (190 each).
- [x] **Split Isolation**: 0 exact prompt leaks across splits. Critical entities disjoint across train/val/test.
- [x] **Evaluator Defect Resolution**: Fixed hardcoded heuristics in `TaskEvaluator._score_factual_qa` (Sahara hot desert and subatomic particles electron/proton). Zero false negatives on expected ground truth.
- [x] **Evaluator Regression Suite**: 17 / 17 tests passed in `tests/test_task_evaluator.py`.
- [x] **Exp18 Generator Test Suite**: 15 / 15 tests passed in `test_exp18_generator.py`.
- [ ] **Semantic Domain Review**: 190 short-explanation records held out in `NEEDS_REVIEW`. Pending human subject-matter review before official dataset serialization.

## 3. Training & Experimental Lineage
- [x] **Training Pipeline Sanity**: Parameter tying, causal sequence masking (`ignore_index=-100`), gradient flow, and optimizer steps verified.
- [x] **Resource Bounded SFT Execution**: 120-step bounded SFT experiment completed on CPU in 4.72 seconds without crashes or memory leaks.
- [x] **Validation Loss Trajectory**: Val loss decreased from 4.6159 to best 3.3562 at Step 80.
- [x] **Checkpoint Lineage Preserved**: Isolated copies saved to `experiments/overnight_runs/mission_v08_overnight_20261023_232924/checkpoints/sft_exp18_run/`.
- [ ] **Held-Out Task Accuracy Improvement**: Experimental SFT checkpoint achieved 1.43% task accuracy vs Baseline 2.14% (EOS completion improved 0% -> 8.75%, but task accuracy did not improve).

## 4. Software & Packaging Readiness
- [x] **Regression Suite**: 123 / 123 tests passed in `tests/`.
- [x] **Neural Runtime Adapter**: CPU inference verified with Delta-W = 0 weight immutability.
- [x] **No Ghost Artifacts / Cleanup**: All experimental checkpoints, logs, and splits quarantined in mission directory.

## Gate Verdict
- **Gate Status**: **`CONDITIONAL_HOLD`**
- **Blockers**:
  1. 190 short-explanation items require human scientific review before Exp18 datasets can be frozen to official JSONL files.
  2. The 120-step experimental checkpoint does not demonstrate superior task accuracy over baseline; canonical step 10,000 must remain the active model weight artifact.
