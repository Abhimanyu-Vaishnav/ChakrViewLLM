# Baseline Reproduction Report: Batches 471–475

**Execution Timestamp:** 2026-10-15T12:12:14  
**Working Branch:** `dev/v0.6-v0.7-validation`  
**Starting Commit:** `023a5e3`  
**Evaluation Script:** [`experiments/batch471_500_reliability/reproduce_baseline.py`](file:///d:/Project/ChakrView/experiments/batch471_500_reliability/reproduce_baseline.py)  
**Machine-Readable Artifact:** [`experiments/batch471_500_reliability/reports/baseline_reproduction_results.json`](file:///d:/Project/ChakrView/experiments/batch471_500_reliability/reports/baseline_reproduction_results.json)

---

### 1. Checkpoint Integrity and SHA-256 Digest Audit

All model checkpoints were directly hashed from disk:
- **Canonical Step 10,000 Checkpoint:**  
  `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`  
  SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` — **VERIFIED IMMUTABLE**
- **Paraphrase-Augmented Candidate:**  
  `experiments/batch411_440_paraphrase/checkpoints/candidate_step150_paraphrase_augmented.pt`  
  SHA-256: `b8b82aaf2b713ea73ff4337a1db7a5b2cce7efd24ae875c70934b2555e0c48b0` — **VERIFIED**
- **Scratchpad Reasoning Candidate:**  
  `experiments/batch441_470_reasoning_gate/checkpoints/candidate_step150_scratchpad_reasoning.pt`  
  SHA-256: `72784a614b2e669242aae1d67adf4e474f186178c18d4ec25070e78378b19759` — **VERIFIED**

---

### 2. Metric Reproduction Summary

| Benchmark Set | Benchmark Size | Paraphrase Candidate (`b8b82aaf...`) | Scratchpad Candidate (`72784a61...`) | Prior Reported Score | Reproduction Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Held-Out Scratchpad** | 6 items | 0.0% (0/6) | **50.0% (3/6)** | 50.0% (3/6) | **VERIFIED EXACT MATCH** |
| **Held-Out Paraphrase** | 20 items | 65.0% (13/20) | **70.0% (14/20)** | 70.0% (14/20) | **VERIFIED EXACT MATCH** |
| **Diagnostic Retention** | 15 items | 66.7% (10/15) | **66.7% (10/15)** | 66.7% (10/15) | **VERIFIED EXACT MATCH** |

---

### 3. Intermediate-Step Correctness Audit (Batch 474)

Inspection of step-by-step arithmetic traces for `candidate_step150_scratchpad_reasoning.pt`:

1. **`sp_test_001`** (`(14 plus 28) minus 10`):
   - **Generated:** `Step 1: 14 + 28 = 42. Step 2: 42 - 10 = 32. Final Answer: 32`
   - **Audit:** Step 1 calculation is correct (42), Step 2 is correct (32), Final Answer matches.
   - **Verdict:** `CORRECT_TRACE_AND_ANSWER`.
2. **`sp_test_002`** (`7 multiplied by 6 minus 10`):
   - **Generated:** `Step 1: 7 * 6 = 42. Step 2: 42 - 10 = 32. Final Answer: 32`
   - **Audit:** Step 1 calculation is correct (42), Step 2 is correct (32), Final Answer matches.
   - **Verdict:** `CORRECT_TRACE_AND_ANSWER`.
3. **`sp_test_003`** (`square of (3 plus 5)`):
   - **Generated:** `Step 1: 3 + 5 = 8. Step 2: 8 * 8 = 64. Final Answer: 64`
   - **Audit:** Step 1 calculation is correct (8), Step 2 is correct (64), Final Answer matches.
   - **Verdict:** `CORRECT_TRACE_AND_ANSWER`.
4. **`sp_test_004`** (Triangle angle `60` and `70`):
   - **Generated:** `Step 1: Total angle sum of a triangle is 180 degrees. Step 2: 60 + 70 = 90 deg`
   - **Audit:** Step 2 arithmetic is incorrect (`60 + 70` is 130, model emitted 90). Generation truncated without EOS.
   - **Verdict:** `INCORRECT_TRACE_AND_WRONG_ANSWER`.
5. **`sp_test_005`** (`(25 plus 15) minus 10`):
   - **Generated:** `Step 1: 14 + 28 = 42. Step 2: 42 - 10 = 32. Final Answer: 32`
   - **Audit:** **Memorized template replay failure.** The model ignored operands 25, 15 and replayed the exact trace from training item 1.
   - **Verdict:** `INCORRECT_TRACE_AND_WRONG_ANSWER` (Memorization shortcut).
6. **`sp_test_006`** (`6 multiplied by 6 minus 6`):
   - **Generated:** `Step 1: 7 * 6 = 4. Step 2: 42 - 10 = 32. Final Answer: 32`
   - **Audit:** **Memorized template replay failure.** The model replayed training item 4 (`7 * 6`), hallucinating partial tokens.
   - **Verdict:** `INCORRECT_TRACE_AND_WRONG_ANSWER`.

### Key Discovery from Batch 474:
The 50% score on the 6-item benchmark was achieved strictly on tasks matching trained expression frames (`14+28-10`, `7*6-10`, `square of (3+5)`). When presented with novel numbers (`(25+15)-10` and `6*6-6`), the 3.4M model fell back to replaying memorized token sequences from the training examples rather than executing dynamic arithmetic binding. This is a crucial finding that motivates Batches 476–482.
