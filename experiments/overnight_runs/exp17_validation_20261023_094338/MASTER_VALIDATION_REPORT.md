# ChakrView Exp17 End-to-End Validation — Master Report

**Mission Directory**: `experiments/overnight_runs/exp17_validation_20261023_094338/`  
**Target Experiment**: `experiments/research_v09_conversation_sft/exp17_instruction_sft_v1/`  
**Primary Evaluated Run**: `experiments/overnight_runs/mission_20261023_003512/`  
**Auditor**: Senior ML Systems Engineer, Model Evaluation Researcher & Dataset Auditor  
**Date**: 2026-10-23  
**Final Validation Verdict**: **`VALIDATED_WITH_LIMITATIONS`**

---

## 1. Executive Summary

This independent validation audit comprehensively investigated the Exp17 Instruction SFT v1 experiment, including its environment, dataset integrity, semantic correctness, training convergence, checkpoint serialization, and dual-track evaluation metrics.

### Key Audit Findings:
1. **Canonical & Dataset Integrity (100% Verified)**:
   - All five protected canonical files match their frozen SHA-256 digests bit-for-bit.
   - The three official dataset split files (`train_exp17_v1.jsonl`, `val_exp17_v1.jsonl`, `test_exp17_v1.jsonl`) exist, contain exactly 320, 40, and 40 records across 8 categories, and exhibit zero cross-split leakage.
2. **Training & Convergence (100% Verified)**:
   - The 400-step SFT training run on CPU was real and fully reproducible.
   - Validation loss dropped from **4.8618** (perplexity: 129.25) to a global minimum of **1.9602** (perplexity: **7.10**) at **Step 225**.
   - The model checkpoint `sft_best_checkpoint.pt` was confirmed to be the exact Step 225 model state dict with finite weights and valid architecture.
3. **Automated Test Suites (100% Pass)**:
   - Full repository test suite: **106 tests run, 0 failures, 0 errors** (120.7s).
   - Exp17 focused test suite: **9 tests run, 0 failures, 0 errors** (32.5s).
4. **Task-Level Evaluation Reality (Critical Finding)**:
   - The model acquired **conversational turn boundaries**, achieving an EOS termination rate of **45.0%** on held-out test prompts (vs. 0.0% baseline) and **60.0%** on the frozen Normal Chat Gate v0.1 suite (vs. 5.0% baseline).
   - However, **actual task accuracy remains 0.0%**. In `structured_formatting`, 100% of responses emitted EOS, but 0% produced valid JSON (emitting empty strings or code docstrings). In `basic_arithmetic`, the model stopped cleanly but calculated wrong values.

---

## 2. Quantitative Verification Summary

| Evaluation Dimension | Baseline (Step 10,000) | Exp17 SFT (Step 225 Best) | Audit Finding & Status |
| :--- | :---: | :---: | :--- |
| **Track 1 Test EOS Rate** | **0.0%** (0 / 40) | **45.0%** (18 / 40) | **VERIFIED** (+45.0% clean turn termination) |
| **Track 1 Test Task Accuracy** | **0.0%** (0 / 40) | **0.0%** (0 / 40) | **VERIFIED** (0.0% semantic task completion) |
| **Track 2 Gate v0.1 EOS Rate** | **5.0%** (1 / 20) | **60.0%** (12 / 20) | **VERIFIED** (+55.0% clean turn termination) |
| **Track 2 Gate Babble Loops** | **95.0%** (19 / 20) | **20.0%** (4 / 20) | **VERIFIED** (-75.0% stutter reduction) |
| **Validation Loss / Perplexity** | **4.8618 / 129.25** | **1.9602 / 7.10** | **VERIFIED** (Global minimum at Step 225) |
| **Unit Test Coverage** | — | **115 / 115 Passed** | **VERIFIED** (106 repo tests + 9 exp17 tests) |
| **Protected Canonical Hashes** | — | **5 / 5 Match** | **VERIFIED** (Bit-exact hash integrity) |

---

## 3. Detailed Stage-by-Stage Findings

### Stage A: Environment & Artifact Forensics
- Environment: Python 3.14.7, PyTorch 2.14.0+cpu, Windows 11 on 20 CPU threads.
- Canonical Checkpoint (`step_10000.pt`): `96357705...` (**MATCH**).
- Tokenizer Vocab & Merges: `db4d6119...` / `ada4b1dc...` (**MATCH**).
- Runtime & Brain Model: `9080ff51...` / `1343d7d1...` (**MATCH**).
- Experimental Checkpoints: `sft_best_checkpoint.pt` (SHA-256: `fa9b5e8c...`) loads cleanly via `NeuralInferenceContract.from_checkpoint` with weight digest `276ba09b...`.

### Stage B: Dataset Integrity
- Records parsed: 320 Train, 40 Val, 40 Test. Exactly 8 categories per split.
- Response tokens: Minimum 6, maximum 34, mean 17.82. 100% within the target 5–35 token bounds.
- Leakage: 0 exact prompt matches, 0 exact response matches across splits.

### Stage C: Semantic Dataset Audit
- 300 records **`VERIFIED`** through programmatic deterministic checks:
  - `basic_arithmetic` (50): 100% calculated operands match expected response.
  - `structured_formatting` (50): 100% valid RFC 8259 JSON objects.
  - `uncertainty_abstention` (50): 38 unanswerable prompts confirmed absent from context; 12 controls confirmed present and answered.
  - `context_extraction` (50) & `reference_resolution` (50): Grounding confirmed.
  - `instruction_following` (50): Deterministic string manipulations confirmed.
- 100 records **`PARTIALLY_VERIFIED`**:
  - `factual_qa` (50) & `short_explanation` (50): Factually and causally sound under general scientific consensus, but used static dictionary lookup fallbacks rather than dynamic external web queries.

### Stage D & E: Task-Level Evaluation & Gate v0.1 Re-evaluation
- Evaluated on exact stable IDs against held-out test split (40 prompts) and Gate v0.1 (20 prompts).
- The model successfully learned response brevity and turn termination, suppressing the pre-training infinite repetition babble loops (`"The total number of the city of..."`).
- However, semantic task accuracy remains 0.0%:
  - JSON formatting: outputs are empty strings or code fragments (`"""if not isinstance..."""`).
  - Arithmetic: outputs have proper structure (`The sum of...`) but wrong math (406 vs 1008).

### Stage F: Validation Trajectory
- Evaluated every 25 steps across 40 validation sequences.
- Step 0 (4.8618) $\to$ Step 100 (2.8433) $\to$ Step 200 (2.2236) $\to$ **Step 225 (1.9602)** $\to$ Step 300 (2.3050) $\to$ Step 400 (2.6766).
- Global minimum at Step 225 correctly captured in `sft_best_checkpoint.pt`.

### Stage G: Trust Gate & Reproducibility
- Parameter `enforce_trust_gate=False` in training script did not mask any defects. The trust gate only checks legacy tags (`"PENDING_HUMAN_REVIEW"`, `"REJECTED"`), which do not exist in Exp17. Initializing with `enforce_trust_gate=True` loads all 320 records without error.

---

## 4. Final Next-Experiment Recommendation

**`PRIMARY ACTION: B. IMPROVE TASK-LEVEL EVALUATION AND SCORING BEFORE ANY FURTHER TRAINING`**

### Rationale:
Exp17 demonstrated that optimizing for turn boundaries (`EOS%`) alone is insufficient. The model learned to emit `<EOS>` immediately or emit superficial template shells without fulfilling the prompt's instructions.
Before launching Exp18 or continuing SFT, an automated offline scoring module must be implemented to measure:
1. JSON syntax and key presence for `structured_formatting`.
2. Extracted numeric values for `basic_arithmetic`.
3. Refusal detection vs. entity extraction for `uncertainty_abstention`.
4. Constraint satisfaction for `instruction_following`.

Only when automated task scorers are integrated can training be tuned (via curriculum loss weighting or sequence constraints) to achieve true instruction following.
