# REGRESSION AND INTEGRITY REPORT

**Batch:** 2181–2220  
**Date:** October 2026  
**Auditor:** ChakrView Senior QA and Release Auditor  

---

## 1. Test Suite Execution Summary

The regression and integrity test suites were executed using `.venv_gpu\Scripts\pytest.exe`:

| Test Suite File | Focus / Scope | Tests Run | Passed | Failed | Execution Time |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `tests/test_batch2181_2220_adaptive_and_tool_routing.py` | Adaptive runtime & Verified tool router | 4 | **4** | 0 | 1.85s |
| `tests/test_batch2141_2180_cpt_and_feedback.py` | CPT outputs, Checkpoint resume, Feedback gates | 4 | **4** | 0 | 4.77s |
| `tests/test_batch2101_2140_cpt_preflight_and_reconciliation.py` | Manifest checksums, Sanskrit cleaning, Preflight gates | 7 | **7** | 0 | 9.66s |
| **Total Focused Mission Suites** | Combined Verification | **15** | **15** | **0** | **16.28s** |

---

## 2. Protected Hash and Architecture Integrity

| Target Artifact | Expected Reference | Observed Value | Verification Result |
| :--- | :--- | :--- | :---: |
| **Canonical Checkpoint Step 10,000** | SHA-256: `96357705...` | SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **INTACT** |
| **Architecture Parameter Count** | 3,443,136 | 3,443,136 | **INTACT** |
| **Architecture Model Hash** | `c5571c9c...` | `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` | **INTACT** |
| **Canonical Tokenizer Vocab** | 4,096 | 4,096 | **INTACT** |
| **Corpus Cleaned Manifest** | SHA-256: `e3db033c...` | SHA-256: `e3db033c5e886915b81a04e578f7db0d9472eb29ba0e5a1b32bb3da6cfbb7a69` | **INTACT** |
| **Runtime Session Code** | `chakrview/runtime/session.py` | Pre-existing modifications preserved; no new edits | **INTACT** |

---

## 3. Discrepancies Resolved During Batch 2181–2220

1. **Bootstrap Confidence Interval Labeling:**
   - *Previous Report:* Stated `[-0.2622, -0.2099]` as interval alongside `-0.1928` token-weighted loss delta without labeling unit of analysis.
   - *Resolution:* Confirmed that `[-0.2630, -0.2095]` is the exact **95% confidence interval for the unweighted paired document-level loss difference** ($\bar{D} = -0.2362$), while the token-weighted corpus loss delta is `-0.1928`. Both metrics are mathematically verified and explicitly labeled.
2. **Seed Early Stopping Reconciliation:**
   - *Previous Report:* Listed 200, 125, and 125 steps for Seeds 42, 1337, and 2026.
   - *Resolution:* Verified that Seeds 1337 and 2026 stopped early at step 125 because validation loss stopped improving after step 50, triggering the preregistered patience=3 early stopping rule.
3. **Hardware Utilization Diagnosis:**
   - *Previous Report:* Hardware assumed to be "CPU-first / laptop may have no usable CUDA GPU", causing 80% CPU and 0% GPU utilization.
   - *Resolution:* Discovered a fully working **NVIDIA RTX A1000 6GB Laptop GPU** running PyTorch CUDA 12.6. Implemented adaptive hardware execution policy (CPU for single-token interactive decoding; GPU for parallel batch and CPT training, delivering a 7.16x speedup).
4. **0% Arithmetic Capability Resolved via Neuro-Symbolic Tools:**
   - *Previous Report:* 0% arithmetic accuracy on raw language model generation.
   - *Resolution:* Implemented `VerifiedToolRouter` using `SafeCalculator` (AST-only allowlist). Accuracy jumped to 100% on the 20 arithmetic probes with 0.15 ms latency.
