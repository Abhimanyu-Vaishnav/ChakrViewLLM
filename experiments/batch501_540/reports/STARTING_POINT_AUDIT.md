# Starting Point & Root Cause Audit: Batches 501–505

**Execution Timestamp:** 2026-10-15T12:25:30  
**Current Branch:** `dev/v0.6-v0.7-validation`  
**Starting Commit Verified:** `35e0b06` — `feat(batch471-500): stress-test scratchpad generalization, integrate evidence gate into runtime, and audit attribution`  
**Working Tree Status:** Clean except for user modifications in `chakrview/runtime/session.py` (strictly preserved).

---

### 1. Checkpoint Integrity & Hash Verification

All active and historical checkpoints were verified by computing SHA-256 digests directly from disk:

| Checkpoint Identifier | File Path | Actual SHA-256 Digest | Status |
| :--- | :--- | :--- | :--- |
| **Canonical Step 10,000** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **VERIFIED IMMUTABLE** |
| **Prior Candidate Step 150 (EOS)** | `experiments/batch381_410_integrity/checkpoints/candidate_step150_corrected_eos.pt` | `27812873dde718485e70825d7be101ede2d1f54d8dd60e3f6a169c51e91a8f9e` | **VERIFIED** |
| **Paraphrase Candidate** | `experiments/batch411_440_paraphrase/checkpoints/candidate_step150_paraphrase_augmented.pt` | `b8b82aaf2b713ea73ff4337a1db7a5b2cce7efd24ae875c70934b2555e0c48b0` | **VERIFIED** |
| **Scratchpad Candidate** | `experiments/batch441_470_reasoning_gate/checkpoints/candidate_step150_scratchpad_reasoning.pt` | `72784a614b2e669242aae1d67adf4e474f186178c18d4ec25070e78378b19759` | **VERIFIED** |

---

### 2. Root Cause Audit: Why Previous Scratchpad Training Failed on Novel Compositions

An in-depth code inspection of `experiments/batch441_470_reasoning_gate/data/train_scratchpad.jsonl` and `train_scratchpad_candidate.py` revealed four primary causes for the 0% generalization on novel multi-step expressions:

1. **Severe Dataset Underspecification (N = 9 examples):**
   The training set contained only 9 scratchpad examples across 4 patterns:
   - 3 items for `(A + B) - C`
   - 2 items for `A * B - C`
   - 2 items for `(A + B)^2`
   - 2 items for triangle angle sum
   With only 2 or 3 examples per pattern, a neural network with 3.4M parameters cannot disentangle the invariant mathematical grammar (e.g. `Step 1: {A} + {B} = {SUM}. Step 2: {SUM} - {C} = {DIFF}`) from the specific tokens of the training instances. The weights simply memorized the exact literal sequence `14 + 28 = 42`.
2. **Evaluation Contamination in the 6-item "Held-Out" Benchmark:**
   Inspection of `test_heldout_scratchpad.jsonl` revealed:
   - `sp_test_001` (`Question: Calculate (14 plus 28) minus 10.\nReasoning:\n`) was an **exact duplicate** of training item `sp_train_001`!
   - `sp_test_002` (`Question: What is 7 multiplied by 6 minus 10?\nReasoning:\n`) was an **exact duplicate** of training item `sp_train_004`!
   - `sp_test_003` (`Question: Calculate the square of (3 plus 5).\nReasoning:\n`) was an **exact duplicate** of training item `sp_train_006`!
   Therefore, 3 out of the 6 "held-out" items were identical memorization tests, creating a false impression of 50% generalization. On the remaining 3 items with novel numbers, performance was **0%**.
3. **Absence of Operator Composition Curriculum:**
   The training set exposed the model directly to complex multi-step scratchpads without intermediate single-step building blocks or shared variable bindings. The model had no bridge connecting its learned single-step arithmetic to two-step chaining.
4. **Tokenization & Shift Alignment:**
   The training pipeline in `train_scratchpad_candidate.py` used:
   ```python
   shift_logits = logits[..., :-1, :].contiguous()
   shift_labels = y[..., 1:].contiguous()
   ```
   This causal shift alignment and `labels = [-100]*len(prompt)` were mathematically correct. The failure was not a loss-masking bug; it was **dataset sparsity and benchmark leakage**.

---

### 3. Strategy for Batches 506–540

To achieve genuine neural compositional generalization:
1. **Build a Substantial, Leakage-Free Dataset (N = 120+ items):**
   Generate diverse, mathematically verified examples covering varied digits, operator permutations, word problems, and symbolic rule chains.
2. **Strict Disjoint Partitioning:**
   Ensure zero operand or prompt overlap between training, validation, and held-out test splits.
3. **Curriculum & Rehearsal Balancing:**
   Combine elementary single-step building blocks (40%), intermediate two-step scratchpads (40%), paraphrase variants (15%), and Stage-C pretraining rehearsal (5%).
4. **Controlled Evaluation:**
   Evaluate on 60+ held-out test items categorized across 6 capability levels.
