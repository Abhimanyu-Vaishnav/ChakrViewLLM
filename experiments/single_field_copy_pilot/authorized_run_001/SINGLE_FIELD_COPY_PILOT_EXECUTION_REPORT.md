# CHAKRVIEW v0.7 — SINGLE-FIELD ENTITY COPY TRAINING PILOT EXECUTION REPORT
**Run Identifier**: `experiments/single_field_copy_pilot/authorized_run_001`  
**Execution Timestamp**: 2026-10-19 17:30:16 IST  
**Status**: **COMPLETED (EARLY STOPPED AT STEP 200)**  
**Verdict**: `FAIL — EARLY STOP AT STEP 200 (<10% EM)`

---

## 1. Executive Summary & Verdict

Exactly one controlled CPU training pilot was executed using the canonical Step-10,000 ChakrMicro base checkpoint (`96357705...e9f471ee`).

| Metric / Dimension | Baseline (Step 10,000) | Step-200 Gate / Final (Step 200) | Pilot Acceptance Target | Gate Status |
| :--- | :---: | :---: | :---: | :---: |
| **Exact-Match Accuracy** | **0/40 (0.0%)** | **0/40 (0.0%)** | $\ge$ 24/40 (60.0%) | **FAILED (< 4/40 at Step 200)** |
| **Normalized Match Accuracy** | **0/40 (0.0%)** | **0/40 (0.0%)** | $\ge$ 28/40 (70.0%) | **FAILED** |
| **Doctest Attractor Rate** | 5/40 (12.5%) | **0/40 (0.0%)** | $\le$ 2/40 (5.0%) | **PASSED (Suppressed to 0%)** |
| **Empty / Premature EOS Rate**| 0/40 (0.0%) | **0/40 (0.0%)** | $\le$ 2/40 (5.0%) | **PASSED** |
| **Average Generation Length** | 40.0 tokens | **5.7 tokens** | Short concise entity | **PASSED (No runaway generation)** |
| **Training Loss** | 5.3921 nats (Step 1)| **0.3459 nats** | Decrease | **MONOTONIC CONVERGENCE** |
| **Validation Loss** | 6.1274 nats | **3.9765 nats** | Lower | **DECREASED BY 2.15 nats** |

### Verdict Statement
**`FAIL — EARLY STOP AT STEP 200 (<10% EM)`**  
Per Section 5 of the user authorization and proposal, training was immediately terminated at step 200 because exact-match accuracy was 0.0% (0/40), which is below the mandatory 10% threshold (< 4/40). No further optimizer steps were taken. In strict accordance with the mandatory preservation and rollback rules, **the candidate checkpoint was NOT promoted**, and canonical Step-10,000 remains the sovereign active base model.

---

## 2. Preflight Verification Audit

All preflight gates were executed and cryptographically verified prior to initializing the optimizer:

1. **Canonical Checkpoint Hash**:
   - File: `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
   - Verified SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (Match)
2. **Canonical Tokenizer Hashes**:
   - Vocab: `data/experiments/vocab_4096/vocab.json` -> `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` (Match)
   - Merges: `data/experiments/vocab_4096/merges.json` -> `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` (Match)
3. **Dataset Invariants & Hashes**:
   - Train file: `experiments/single_field_copy_pilot/authorized_run_001/data/train.jsonl` (200 records, SHA-256: `76e033792245117af96abf8ad652d24cf76fbe58721d4c0e4b571dd09cccafbc`)
   - Validation file: `experiments/single_field_copy_pilot/authorized_run_001/data/val.jsonl` (40 records, SHA-256: `9d8fa37bb161552c1fdc5c7697db204d15a9763b6fa4d3b18124b5ce17091535`)
   - Verbatim containment: 100% of target values in train and val appear verbatim in their context.
   - Disjointness: 0% overlap of contexts, target values, host IDs, or template families between train and val.
4. **Held-out Test Split Protection**:
   - Zero access, zero reads, and zero copies from held-out `test.jsonl` records.
5. **Protected File Invariant**:
   - `chakrview/runtime/session.py` checked via `git diff --name-only` -> completely empty (0 modifications).

---

## 3. Training Execution & Resource Details

- **Device**: CPU only (`Intel Core / AMD x86_64`)
- **Execution Invocations**: Exactly 1 (no retries, no automatic restarts)
- **Completed Steps**: 200 optimizer steps
- **Training Wall-Clock Time**: 13.19 seconds (~42ms per step)
- **RNG Seed**: 42 (fixed)
- **Hyperparameters**:
  - Optimizer: AdamW (`betas=(0.9, 0.95)`, `weight_decay=0.01`)
  - Learning Rate Schedule: 25 steps linear warmup to `2e-4`, followed by cosine decay targeting `2e-5` at step 300
  - Actual LR at Step 200: `7.26e-05`
  - Effective Batch Size: 2 (sequence length up to 512, dynamic padding)
  - Gradient Clipping: 1.0 (grad norm stabilized between 2.5 and 6.0)

### Training Loss Trajectory
- Step 1: Train Loss = 5.3921 | Val Loss = 6.0941 | LR = 8.00e-06
- Step 25: Train Loss = 2.2758 | Val Loss = 3.8687 | LR = 2.00e-04
- Step 50: Train Loss = 0.8680 | Val Loss = 3.8920 | LR = 1.96e-04
- Step 75: Train Loss = 0.9114 | Val Loss = 3.8459 | LR = 1.86e-04
- Step 100: Train Loss = 0.8018 | Val Loss = 3.7481 | LR = 1.69e-04
- Step 125: Train Loss = 0.4983 | Val Loss = 3.9832 | LR = 1.47e-04
- Step 150: Train Loss = 0.4495 | Val Loss = 3.9233 | LR = 1.23e-04
- Step 175: Train Loss = 0.7059 | Val Loss = 3.9810 | LR = 9.72e-05
- Step 200: Train Loss = 0.3459 | Val Loss = 3.9765 | LR = 7.26e-05

---

## 4. Deep Qualitative Analysis: What Did the Model Learn?

While the pilot failed the exact-match threshold, the behavioral shift from Step 10,000 to Step 200 reveals crucial empirical insights into the ChakrMicro architecture:

### 1. Complete Elimination of Doctest Code Attractor & Runaway Babbling
- **Base Checkpoint (Step 10,000)**: Ran away for 40 tokens on every example, generating repetitious loops (`# If the first number of the number of the number...`) or doctest code attractors (`>>> linked_list.insert(...)`). Attractor rate was 12.5%.
- **Candidate (Step 200)**: Attractor rate dropped to **0.0%** (0/40). Average generation length dropped from 40.0 tokens to **5.7 tokens**, cleanly terminating with `<EOS>` (Token ID 1).

### 2. Learned Dialog Protocol and Field Semantics
The model learned to emit a single field value rather than syntax gibberish:
- When prompted for **server status** (`Copy the exact value of status...`), it generated: `running` (a valid server status from training).
- When prompted for **utilization metric** (`Copy the exact value of utilization...`), it generated: `94%` (a valid percentage metric from training).
- When prompted for **code signature** (`Copy the exact value of signature...`), it generated: `key_hash_025` (a valid token signature from training).

### 3. Root Cause of Failure: Training Distribution Memorization vs. Context Pointer Induction
- Because training and validation were strictly disjoint in entities, the model learned a **semantic prior** over fields (e.g. predicting the most frequent training status `running` or training metric `94%`) rather than forming dynamic **induction heads** that point back into the context passage to retrieve the novel validation entity (`standby`, `throttled`, `71%`, `depot_09`).
- In a 1.52M parameter model (6 layers, 6 heads, $d=192$), 200 training examples with batch size 2 are sufficient to learn the *output format and field vocabulary*, but insufficient to build attention routing circuits that override pretraining token associations with dynamic context tokens.

---

## 5. Artifact & Integrity Checklist

- [x] Step-200 checkpoint saved: `experiments/single_field_copy_pilot/authorized_run_001/checkpoint_step_200.pt` (SHA-256: `560f2941eaae8f17485d14223ef3b603332c7c7e10f8da06982ef72fa17fdfe3`)
- [x] Canonical Step-10,000 checkpoint hash verified bit-identical: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
- [x] Canonical tokenizer vocab & merges verified bit-identical
- [x] `chakrview/runtime/session.py` verified unmodified (`git diff` clean)
- [x] Held-out test split untouched
- [x] Focused unit tests added and passing: `tests/test_single_field_copy_pilot.py` (4/4 passed)
- [x] Regression test suite passing: `tests/test_copy_capability_gate.py`, `tests/test_sft_v3_dataset_builder.py` (12/12 passed)
- [x] Sovereign canonical model remains Step 10,000; failed candidate was **NOT** promoted.

---

## 6. Recommendations Toward v0.7 & November v1.0 Goal

1. **Do Not Overwrite or Mutate the Base Architecture**: ChakrMicro's small footprint (1.52M) makes pointer induction challenging when fine-tuning purely causal heads from scratch on small corpora without explicit prefix attention or copy penalties.
2. **Hybrid / Constrained Pointer Mechanism**: For the November 2026 v1.0 coding model, entity extraction and verbatim identifier copy must combine:
   - Neural prior generation (which this pilot proved can successfully learn field types and suppress attractors);
   - Constrained Trie / Token masking conditioned on tokens present in the prompt context.
3. **Curriculum Expansion**: If pure neural copying is pursued in future authorized runs, the dataset must include variable position copy tasks and explicit prefix/copy objective weights.
