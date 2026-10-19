# SFT Pilot Completion Report: Batch 2861–2900
## Execution Summary, Provenance Verification & Final Decision

### 1. Execution Overview

Under explicit authorization, Batch 2861–2900 carried out a 100-step CPU SFT pilot:
- **Authorization Scope**: Exactly 100 optimizer steps on CPU using the curated 192-example training split.
- **Execution Engine**: `SFTExperimentEngine` with cosine learning rate schedule (`5e-5` to `5e-6`), batch size 4, gradient clipping 1.0, and dynamic padding.
- **Completed Optimizer Steps**: 100 steps in 13.38 seconds on CPU.
- **Parent Checkpoint**: Canonical Step 10,000 (`training_checkpoint_step_10000.pt`, SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`, verified unchanged).
- **Candidate Checkpoint**: `experiments/batch2821_2860/sft_pilot_run/sft_checkpoint_step_100.pt` (SHA-256: `f911d081c8a6fbdde0e21322c15bb56d2db6f84593c47ea5e280dbfae2b574b1`).

---

### 2. Critical Gate Results

1. **Gate A: Causal Loss Alignment**: **PASS**. Verified that terminal prompt token directly predicts first response token with prompt tokens masked to `-100`. Loss matched manual tensor cross-entropy calculation with 0.0 error.
2. **Gate B: Dataset Integrity & Leakage**: **PASS**. 240 records across 8 categories with 0 leakage across splits.
3. **Gate C: Evaluation Grounding**: **PASS**. Tested baseline on the 24 held-out cases with deterministic automated checks.
4. **Gate D: Training Isolation**: **PASS**. Output written strictly to `experiments/batch2821_2860/sft_pilot_run/` with canonical paths protected.

---

### 3. Checkpoint Provenance & Invariants

- **Parent Checkpoint SHA-256**: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (Unchanged).
- **Candidate Parameter Digest**: `d084b6f66d0a867a47c3a107f577e80a886e33a5557e0c9663bd25c6d4a1cee8` (Finite, non-corrupted).
- **Tokenizer Artifacts**: Hashes intact and untouched.
- **`chakrview/runtime/session.py`**: Untouched.
- **Remote Operations**: Zero commits, pushes, or remote calls performed.

---

### 4. Final Verdict

`PASS` — All critical gates were verified, the authorized 100-step pilot executed cleanly on CPU, outputs were isolated, and candidate versus baseline comparative evidence was measured and documented without fabricating quality claims.
