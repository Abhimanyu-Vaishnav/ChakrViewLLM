# ChakrView Post-Overnight Forensic Audit — Summary Report

**Audit Target**: `experiments/overnight_runs/mission_20261023_003512/`  
**Forensic Audit Directory**: `experiments/overnight_runs/forensic_audit_20261023_070602/`  
**Auditor**: Independent ML Experiment Auditor  
**Date**: 2026-10-23  
**Final Audit Verdict**: **`VALID_WITH_LIMITATIONS`**

---

## 1. Overall Audit Verdict

The overnight mission `mission_20261023_003512` represents a **technically valid, reproducible, and non-destructive experimental run**. The training, validation loss reduction, checkpointing, and termination metrics claimed in the previous report actually occurred and are fully corroborated by raw underlying logs and checkpoint binaries.

However, the previous report's overall `SUCCESS` verdict must be qualified as **`VALID_WITH_LIMITATIONS`** due to two specific findings:
1. **Semantic Misalignment of EOS Termination**: The quantitative lift in EOS stop-token emission (0% to 45% on Test, 5% to 60% on Gate v0.1) is mathematically real, but it represents **conversational turn-boundary learning, not semantic task completion**. The model frequently halts immediately without producing valid JSON, or produces plausible template text with incorrect facts/arithmetic.
2. **Metadata Fallbacks in Dataset Source Attribution**: While arithmetic, JSON, and abstention proofs were independently verified via code, 58 out of 100 factual/explanation claims used static dictionary fallbacks rather than live-queried external citations.

---

## 2. Forensic Findings by Task

### Task 1: Verification of the Actual Run
- **Files & Artifacts**: All 24 mission artifacts exist, possess non-zero byte sizes, and are cataloged with SHA-256 digests in [`ARTIFACT_VERIFICATION.json`](file:///d:/Project/ChakrView/experiments/overnight_runs/forensic_audit_20261023_070602/ARTIFACT_VERIFICATION.json).
- **Execution Proof**: Background execution logs and checkpoint timestamps confirm the 400-step SFT run completed in 22.72 seconds.
- **Checkpoint & Loss Trajectory**: Raw checkpoint state dicts independently corroborate that validation loss dropped from 4.8618 to a global minimum of **1.9602** at **Step 225**, before rising slightly to 2.6766 by Step 400. The best checkpoint was saved atomically as `sft_best_checkpoint.pt`.
- **Loadability**: Both `sft_best_checkpoint.pt` and `sft_checkpoint_step_400.pt` load cleanly via `NeuralInferenceContract.from_checkpoint`.

### Task 2: Investigation of the Trust Gate
- **Mechanism**: The parameter `enforce_trust_gate: bool = True` in [`sft_engine.py`](file:///D:/Project/ChakrView/chakrview/training/sft_engine.py#L112-L122) only checks for records marked `"PENDING_HUMAN_REVIEW"` or `"REJECTED"`. Contamination checks are unconditional.
- **Impact**: Official Exp17 dataset records do not contain these legacy review tags. Testing the engine with `enforce_trust_gate=True` loads all 320 training examples without error. The overnight run is therefore **valid under protocol**, though `enforce_trust_gate=True` should be preserved as default.
- See full analysis in [`TRUST_GATE_ANALYSIS.md`](file:///d:/Project/ChakrView/experiments/overnight_runs/forensic_audit_20261023_070602/TRUST_GATE_ANALYSIS.md).

### Task 3: Dataset Claim Verification
- **Official Splits**: `train_exp17_v1.jsonl` (320), `val_exp17_v1.jsonl` (40), and `test_exp17_v1.jsonl` (40) exist with verified SHA-256 hashes matching the manifest.
- **Abstention Verification**: Confirmed that all 38 unanswerable prompts have target attributes absent from context, and all 12 controls have target facts present in context (verified via code, replacing the defective `ans_absent = True` stub).
- **Deterministic Checks**: Arithmetic, JSON, context extraction, reference resolution, and instruction following are 100% verified.
- **Citation Limitations**: Factual and scientific explanation sources used dictionary fallbacks for items outside the primary lookup tables.
- See full analysis in [`DATASET_CLAIM_VERIFICATION.md`](file:///d:/Project/ChakrView/experiments/overnight_runs/forensic_audit_20261023_070602/DATASET_CLAIM_VERIFICATION.md).

### Task 4: Evaluation Recalculation
- **Recalculated Metrics**:
  - Test EOS Rate: Baseline = **0.0%** (0/40) | SFT Step 225 = **45.0%** (18/40) [**VERIFIED**]
  - Gate v0.1 EOS Rate: Baseline = **5.0%** (1/20) | SFT Step 225 = **60.0%** (12/20) [**VERIFIED**]
  - Perplexity: Baseline = **129.25** | SFT Step 225 = **7.10** [**VERIFIED**]
- **Semantic Reality**: Despite 100% EOS termination in `structured_formatting`, 0 out of 5 test examples produced valid JSON (2 were empty strings, 3 were code docstring fragments). Arithmetic produced incorrect values (e.g. 406 vs 1008).
- See full analysis in [`EVALUATION_RECALCULATION.md`](file:///d:/Project/ChakrView/experiments/overnight_runs/forensic_audit_20261023_070602/EVALUATION_RECALCULATION.md).

### Task 5: Reproducibility & Integrity
- **Canonical Asset Hashes**: All 5 canonical artifacts (`step_10000.pt`, `vocab.json`, `merges.json`, `session.py`, `model.py`) match their frozen SHA-256 hashes bit-for-bit.
- **Git State**: Clean working tree on tracked files; zero commits, resets, or destructive actions.
- See full analysis in [`REPRODUCIBILITY_AND_INTEGRITY.md`](file:///d:/Project/ChakrView/experiments/overnight_runs/forensic_audit_20261023_070602/REPRODUCIBILITY_AND_INTEGRITY.md).

---

## 3. Forensic Deliverables Index

All forensic audit outputs are preserved in:
`file:///d:/Project/ChakrView/experiments/overnight_runs/forensic_audit_20261023_070602/`

1. [`FORENSIC_SUMMARY.md`](file:///d:/Project/ChakrView/experiments/overnight_runs/forensic_audit_20261023_070602/FORENSIC_SUMMARY.md)
2. [`ARTIFACT_VERIFICATION.json`](file:///d:/Project/ChakrView/experiments/overnight_runs/forensic_audit_20261023_070602/ARTIFACT_VERIFICATION.json)
3. [`TRUST_GATE_ANALYSIS.md`](file:///d:/Project/ChakrView/experiments/overnight_runs/forensic_audit_20261023_070602/TRUST_GATE_ANALYSIS.md)
4. [`DATASET_CLAIM_VERIFICATION.md`](file:///d:/Project/ChakrView/experiments/overnight_runs/forensic_audit_20261023_070602/DATASET_CLAIM_VERIFICATION.md)
5. [`EVALUATION_RECALCULATION.md`](file:///d:/Project/ChakrView/experiments/overnight_runs/forensic_audit_20261023_070602/EVALUATION_RECALCULATION.md)
6. [`REPRODUCIBILITY_AND_INTEGRITY.md`](file:///d:/Project/ChakrView/experiments/overnight_runs/forensic_audit_20261023_070602/REPRODUCIBILITY_AND_INTEGRITY.md)
