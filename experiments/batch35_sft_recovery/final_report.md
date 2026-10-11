# CHAKRVIEW — BATCH 35 MASTER SPRINT FINAL REPORT
## EOS Regression Verification, SFT Pipeline Correction & Controlled Recovery

### Executive Summary

Batch 35 completed an independent verification of the EOS-collapse regression using raw generated token IDs, unified training and inference conversation formatting, corrected all dataset provenance misattributions, and established a reproducible training safety gate.

Key outcomes:
1. **Raw Token-Level EOS Verification**: Autoregressive decoding verified that the base model emits continuous tokens (0% first-token EOS rate across 10 prompts), while Step 50 SFT emits EOS first on 50% of prompts, and Step 200 SFT emits EOS first on 70% of prompts. The collapse is produced by the model weights, not runtime trimming or stopping artifacts.
2. **Conversation Format Unification**: Unified `LocalModelSession.build_prompt_package` to format dialogue using canonical `ConversationTemplate` ChatML tags (`<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`), eliminating formatting mismatch between session prompts and template representations.
3. **100% Grounded & Disjoint Dataset**: Corrected `TR-CD-01` (derived from `code_foundation.txt`) and `TR-CD-02` (derived from `math_foundation.txt`). Verified 0 document overlap and 0 prompt overlap between train and validation splits.
4. **Training Gate**: The training gate remains **BLOCKED** to prevent repetitive overfitting until a larger, multi-domain instruction dataset (>500 examples) is ready.
5. **Full Regression Health**: All 36 repository regression tests passed in 9.37 seconds.

---

### Phase Status Matrix

| Phase | Description | Status | Summary & Evidence |
|---|---|:---:|---|
| **Phase 0** | Baseline & Git Audit | **VERIFIED COMPLETE** | Verified canonical digest `c5571c...00a282da` ($\Delta W=0$), 3,443,136 parameters, and isolated load of all 3 checkpoints. ([`initial_state.json`](file:///D:/Project/ChakrView/experiments/batch35_sft_recovery/audits/initial_state.json)) |
| **Phase 1** | Verify EOS Metrics via Raw Tokens | **VERIFIED COMPLETE** | Evaluated 10 prompts inspecting exact integer token IDs: Base 10k: 0.0% first-token EOS; Step 50: 50.0%; Step 200: 70.0%. ([`token_level_eos_audit.json`](file:///D:/Project/ChakrView/experiments/batch35_sft_recovery/evaluation/token_level_eos_audit.json)) |
| **Phase 2** | Separate Possible Root Causes | **VERIFIED COMPLETE** | Refuted causal alignment and runtime stopping artifacts; confirmed 10.8-epoch overfitting on 37 examples and prompt format shift. ([`root_cause_matrix.json`](file:///D:/Project/ChakrView/experiments/batch35_sft_recovery/audits/root_cause_matrix.json)) |
| **Phase 3** | Verify Dataset & Provenance | **VERIFIED COMPLETE** | Corrected `TR-CD-01` and `TR-CD-02` sources. Achieved 100% source grounding and strict document-level disjointness. ([`corrected_provenance_manifest.json`](file:///D:/Project/ChakrView/experiments/batch35_sft_recovery/data/corrected_provenance_manifest.json)) |
| **Phase 4** | Correct Pipeline Defects | **VERIFIED COMPLETE** | Updated `LocalModelSession.build_prompt_package` in `session.py` to use canonical ChatML formatting. |
| **Phase 5** | Candidate Comparison | **VERIFIED COMPLETE** | Summarized generation lengths and stop reason distributions across checkpoints. ([`candidate_comparison_summary.json`](file:///D:/Project/ChakrView/experiments/batch35_sft_recovery/evaluation/candidate_comparison_summary.json)) |
| **Phase 6** | Training Readiness Gate | **BLOCKED** | Gate upheld: Model training paused until larger-scale SFT dataset is ready. |
| **Phase 7** | Regression Tests | **VERIFIED COMPLETE** | Full regression suite: **36/36 tests passed** in 9.37 seconds. |
| **Phase 8** | Final Report & Documentation | **VERIFIED COMPLETE** | Published report and audit artifacts. |
| **Overall** | **Sprint Verdict** | **PARTIAL** | Pipeline corrections and audits **VERIFIED COMPLETE**; Model training intentionally **BLOCKED**. |

---

### Token-Level Evaluation Matrix

Evaluated across 10 deterministic prompts (4 validation clean + 6 expanded diagnostics):
- **Base Step 10,000**:
  - Exact first-token == EOS: **0 / 10 (0.0%)**
  - Average generated tokens: **30.7**
  - Stop reasons: 9 `max_tokens`, 1 `eos`
- **Step 50 SFT**:
  - Exact first-token == EOS: **5 / 10 (50.0%)**
  - Average generated tokens: **16.5**
  - Stop reasons: 5 `eos`, 5 `max_tokens`
- **Step 200 SFT**:
  - Exact first-token == EOS: **7 / 10 (70.0%)**
  - Average generated tokens: **9.6**
  - Stop reasons: 8 `eos`, 2 `max_tokens`

---

### Full Regression Suite Results

All 36 tests passed:
```text
tests\test_pretraining_engine.py ......                                  [ 16%]
tests\test_batch28_stage_d_gate.py ....                                  [ 27%]
tests\test_sft_engine.py ...                                             [ 36%]
tests\test_batch31_chat_runtime.py ......                                [ 52%]
tests\test_batch32_acceptance.py ........                                [ 75%]
tests\test_batch33_integrity.py ...                                      [ 83%]
tests\test_batch34_sft_diagnostics.py ...                                [ 91%]
tests\test_batch35_recovery.py ...                                       [100%]

============================= 36 passed in 9.37s ==============================
```

---

### Final Git Status & Permission Compliance

- **Current Branch:** `dev/v0.6-v0.7-validation`
- **Tracked Modified Files:**
  - `chakrview/runtime/session.py` (unified prompt packaging to canonical ChatML ConversationTemplate)
- **Untracked Artifacts:**
  - `experiments/batch33_integrity/`
  - `experiments/batch34_sft_root_cause/`
  - `experiments/batch35_sft_recovery/`
  - `tests/test_batch33_integrity.py`
  - `tests/test_batch34_sft_diagnostics.py`
  - `tests/test_batch35_recovery.py`
- **Confirmation:** Zero commits, pushes, merges, rebases, or resets were performed.
