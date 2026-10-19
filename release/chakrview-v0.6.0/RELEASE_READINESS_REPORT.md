# ChakrView v0.6.0 Release Readiness Report

**Target Release**: `v0.6.0`  
**Classification**: **Experimental Research Preview**  
**Audit Date**: 2026-10-19  
**Target Release Date**: 2026-11-02  
**Overall Verdict**: **`RELEASE CANDIDATE READY`**

---

## 1. Compliance Audit Against Mandatory Safety Gates

| Gate # | Condition / Safety Boundary | Audit Result | Evidence / Verification |
| :---: | :--- | :---: | :--- |
| **G1** | Canonical Step-10,000 Checkpoint Hash | **PASS** | SHA-256 matches bit-identical (`96357705...e9f471ee`). |
| **G2** | Canonical Tokenizer Vocabulary Hash | **PASS** | SHA-256 matches bit-identical (`db4d6119...07e7218a`). |
| **G3** | Canonical Tokenizer Merges Hash | **PASS** | SHA-256 matches bit-identical (`ada4b1dc...0016efa7`). |
| **G4** | Zero Parameter Modification ($\Delta W = 0$) | **PASS** | No training, backward pass, optimizer step, or fine-tuning executed. |
| **G5** | Protected File `session.py` Preservation | **PASS** | `chakrview/runtime/session.py` is clean (`git diff` is empty). |
| **G6** | Held-Out Test Split Protection | **PASS** | `test.jsonl` was never accessed or opened. |
| **G7** | No Premature Promotion of SFT V3 Candidate | **PASS** | Step 10,000 remains canonical default; SFT V3 candidate is not promoted. |
| **G8** | Test Suite Execution | **PASS** | 7/7 release smoke tests pass; 9/9 diagnostic regression tests pass. |
| **G9** | Working Quickstart & Benchmark Execution | **PASS** | `quickstart_inference.py` and `benchmark_v060.py` execute reproducibly. |

---

## 2. Benchmark Summary (Local Reference Machine)

- **Platform**: Windows 11 (Intel Core i7/i9 20 logical threads, 32 GB RAM)
- **Cold-Start Model Initialization**: **444.25 ms**
- **Average Time-To-First-Token (TTFT)**: **29.11 ms**
- **Average Generation Throughput**: **136.30 tokens/second**
- **Peak Traced Python Memory**: **8.36 MB**
- **Total Process Working Set (RSS)**: **271.55 MB**
- **Greedy Output Determinism**: **100% reproducible** across 3 consecutive trials
- **Smoke-Test Success Rate**: **5/5 (100.0%)** on standard test prompts

---

## 3. Proposal for the Next Capability Experiment (Preparation Only)

### Context & Need
Recent diagnostics established that while dialog-style prompt framing (`User: ... Assistant:`) substantially suppresses the doctest code attractor (reducing it from 90.0% to 15.0%), ChakrMicro achieves **0/20 (0.0%)** exact copying because it lacks cross-attention induction heads. Attempting to fine-tune directly on multi-field JSON schemas overwhelmed the 100-step pilot with syntax rather than entity grounding.

### Experiment Design: "Single-Field Entity Copy Pilot"
1. **Smallest Viable Task**: Given a short context passage (15–30 tokens) stating a single entity (e.g. `Server val_01 has status active`), generate ONLY the copied entity string (e.g. `active`).
2. **Dataset Requirements**:
   - 200 synthetic training pairs and 40 validation pairs.
   - 100% verified ground-truth substring containment.
   - Uniform category distribution: server status, inventory item names, metric names, feature names.
3. **Prompt Formatting**:
   ```text
   User: In the following context: "{context}"
   Copy the exact value of {target_field}:
   Assistant: {target_value}
   ```
4. **Supervision & Loss Masking**:
   - Mask prompt tokens with `-100`.
   - Supervise ONLY the `{target_value}` and terminal `<eos>` token.
5. **Execution Budget & Constraints**:
   - 250 optimizer steps on CPU (estimated execution time: ~45–60 seconds).
   - Batch size 2, AdamW with lr=2e-4, linear warmup (25 steps), cosine decay.
   - Periodic validation loss evaluation every 25 steps.
6. **Success Criteria**:
   - Primary metric: Exact match on held-out validation set >= 60% (24/40).
   - Falsification/Stop: If validation loss plateaus or exact match is 0/40 by step 150, stop training.
   - Candidate checkpoint saved to isolated directory `experiments/single_field_copy_pilot/`.
