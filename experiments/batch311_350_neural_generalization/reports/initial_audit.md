# Batches 311–315 Initial Audit: Lineage, Checkpoint Integrity, and Evaluation Reproduction

**Date:** 2026-10-15  
**Milestone:** Batches 311–350  
**Branch:** `dev/v0.6-v0.7-validation`  
**Parent Commit:** `7b55a79` (`feat(batch281-310): train neural capability candidate, improve target loss masking, and verify multi-hop grounding`)  
**Previous Milestone Parent:** `43e44b1` (Batches 271–280)  
**Execution Environment:** CPU-First, Hardware-Adaptive (AMD/Intel x86_64, Windows NT, PyTorch 2.14.0+cpu, 14 execution threads, no CUDA GPU)

---

## 1. Batch 311 — Repository Integrity Audit

### Git State Verification
- **Branch:** `dev/v0.6-v0.7-validation`
- **HEAD Commit:** `7b55a794916a4c2810a0a5247c1eb7eecb8c82eb`
- **Parent Commit:** `43e44b1e32d16c4fa3158c89c565ffb48be04085`
- **Working Tree:** Protected user modifications in `chakrview/runtime/session.py` and untracked sprint artifacts from historical batches 103–122 remain intact and uncommitted.

---

## 2. Batch 312 — Canonical Checkpoint Audit

- **Artifact Path:** `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
- **SHA-256 Digest:** `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
- **Status:** **EXACT MATCH (VERIFIED INTACT)**.
- **Model Invariants:**
  - Architecture: `ChakrMicro`
  - Parameters: Exactly **3,443,136** parameters across 6 layers, $d_{model}=192$, 6 heads, vocabulary 4,096.
  - Weights status: Frozen, read-only baseline.

---

## 3. Batch 313 — Candidate Checkpoint Audit (Step 70)

- **Artifact Path:** `experiments/batch281_310_neural_capability/checkpoints/chakrmicro_capability_candidate_step70.pt`
- **SHA-256 Digest:** `3b6b98b821a0c1e73aa18cae25ffc866e027b6c40acd6fdb4673e25b02b6c2fe`
- **Status:** **EXACT MATCH (VERIFIED INTACT)**.
- **Lineage Details:**
  - Initialized directly from canonical step-10,000 checkpoint.
  - Parameter Count: Exactly 3,443,136 parameters.
  - Training Step: 70 steps of target-masked fine-tuning with linear warmup (15 steps) + cosine annealing.
  - Final Training Loss: 4.2653.

---

## 4. Batch 314 & 315 — Training Script Audit & Exact Reproduction

### Training Script Findings
- Evaluated `experiments/batch281_310_neural_capability/run_neural_experiments.py`:
  - **Loss Target Masking:** Verified that prompt tokens receive target ID `-100`. The causal language modeling loss strictly excludes prompt tokens via `CrossEntropyLoss(ignore_index=-100)`.
  - **Delimiter Alignment:** Preserves trailing `\n` in `<|im_start|>assistant\n`.
  - **Optimizer State:** The step-70 checkpoint saved `model_state_dict` and `train_meta`, but omitted the full optimizer state dictionary. Any extended training starting from step 70 must initialize fresh optimizer moments or branch from step 0 with full continuity.

### Reproduction of Evaluation Loss Metrics
Using independent evaluation scripts with the frozen tokenizer and datasets:
- **Baseline Validation Loss:** Reported: `8.4447` | Reproduced: `8.4447` (**MATCH**)
- **Baseline Held-Out Test Loss:** Reported: `8.3141` | Reproduced: `8.3141` (**MATCH**)
- **Candidate Step-70 Validation Loss:** Reported: `5.9164` | Reproduced: `5.9164` (**MATCH**)
- **Candidate Step-70 Held-Out Test Loss:** Reported: `5.8067` | Reproduced: `5.8067` (**MATCH**)

### Conclusion & Eligibility Gate
The candidate checkpoint's reported numerical performance has been independently reproduced to 4 decimal places. The candidate is eligible for controlled comparison against extended training regimes.
