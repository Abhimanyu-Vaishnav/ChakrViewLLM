# Evaluator Validation Report

## Batches 1461–1500: Adversarial Testing and Integrity Verification of Benchmark Evaluators

### 1. Purpose of the Evaluator Audit

To ensure benchmark scores are trustworthy and free from evaluator artifacts, all evaluation harnesses and scoring functions used in Batches 1461–1500 were tested against deliberate adversarial edge cases before accepting final results.

---

### 2. Evaluator Edge Cases Tested

| Test Suite / Function | Adversarial Test Input | Expected Evaluator Behavior | Observed Evaluator Behavior | Status |
|---|---|---|---|---|
| **Numeric Evaluator (Track C)** | Expected: `"40"`, Output: `"Evaluate 100 - 20 * 3 = 2000"` | Reject as Incorrect | Rejected (`is_correct: false`, extracted `2000`) | **PASS** |
| **Numeric Evaluator (Track C)** | Substring collision: `"4"` in `"The 40 workers"` | Reject (No substring match) | Rejected (Extracts `40`, not `4`) | **PASS** |
| **Numeric Evaluator (Track C)** | Empty or malformed string: `""` or `None` | Return `None`, mark Incorrect | Handled safely, marked Incorrect | **PASS** |
| **Code Sandbox (Track D)** | Unterminated docstring `""" \n if not ...` | Catch `SyntaxError`, mark test failed | Captured `SyntaxError`, marked failed | **PASS** |
| **Code Sandbox (Track D)** | Infinite loop `while True: pass` | Enforce 2.0s subprocess timeout | Timed out, returned timeout error | **PASS** |
| **Code Sandbox (Track D)** | Unrestricted file/network access | Block access in restricted environment | Safe execution confirmed | **PASS** |
| **Code Sandbox (Track D)** | Pass stub `def foo(): pass` | Syntax valid, test assertions fail | Valid syntax (1), test failed (0) | **PASS** |
| **Grounding Evaluator (Track B)** | Substring hallucination without context | Require exact entity presence | Verified against expected string | **PASS** |
| **Control Evaluator (Controls)** | Empty answer or non-abstention on false premise | Catch failure to refute/abstain | Scored strictly against decision policy | **PASS** |

---

### 3. Checkpoint & Dataset Cryptographic Integrity

All checkpoints and benchmark datasets were verified cryptographically before and after evaluation:

1. **Canonical Baseline Checkpoint:**
   - Path: `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
   - SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
   - Status: **VERIFIED & UNMODIFIED**

2. **Domain-Adapted Candidate Checkpoint:**
   - Path: `experiments/batch1381_1420/checkpoints/chakrmicro_domain_adapted_experimental.pt`
   - SHA-256: `15d48176efd6941415cc446b0f0602573a266a49a17da56d72d365d445091139`
   - Status: **VERIFIED & UNMODIFIED**

3. **Frozen 200-Task Capability Dataset:**
   - Path: `experiments/batch1461_1500/data/capability_audit_200.json`
   - SHA-256: `931c877e766d427b1c6619f43b0dda1b3fa507d8f23ca54f4ac1da1e81d4c07a`
   - Status: **VERIFIED & FROZEN**

4. **Raw Benchmark Results:**
   - Path: `experiments/batch1461_1500/results/CAPABILITY_AUDIT_RESULTS.json`
   - Machine-readable traces recorded for all 200 tasks.

---

### 4. Evaluator Audit Acceptance

The evaluation harnesses are verified to be strictly objective, reproducible, and immune to superficial substring hits or silent runtime exceptions.
