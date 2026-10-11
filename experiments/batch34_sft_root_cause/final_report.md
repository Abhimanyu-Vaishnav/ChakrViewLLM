# CHAKRVIEW — BATCH 34 MASTER SPRINT FINAL REPORT
## SFT EOS-Collapse Root-Cause Investigation, Dataset Provenance Audit & Training Safety

### Executive Summary

Batch 34 completed an in-depth forensic investigation into the premature EOS termination observed in Stage D SFT checkpoints, audited the factual provenance of existing instruction records, and evaluated repository training safety.

Key findings:
1. **Root Cause Confirmed**: The premature EOS collapse (Step 50: 50%, Step 200: 80% on 10 evaluation prompts) is **not** caused by tokenization bugs, tokenizer special-token insertion, or causal loss shifting defects. Instead, it is caused by **severe epoch-level overfitting on a tiny 37-example dataset** sampled over ~10.8 effective epochs, coupled with a format mismatch between unformatted training prompts and formatted inference prompts.
2. **Provenance Audit**: 11 of 13 records in the clean SFT dataset were verified as directly grounded in their cited source texts; 2 records (`TR-CD-01` and `TR-CD-02`) were flagged as `unsupported` due to file misattribution.
3. **Training Gate**: The training gate remains **BLOCKED** to prevent repeating the overfitting failure mode until a larger, verified instruction corpus with length regularization is available.
4. **Regression Integrity**: All 33 repository regression tests passed cleanly.

---

### Phase Status Matrix

| Phase | Description | Status | Summary & Evidence |
|---|---|:---:|---|
| **Phase 0** | Baseline & Git Audit | **VERIFIED COMPLETE** | Verified canonical digest `c5571c...00a282da` ($\Delta W=0$), 3,443,136 params, and isolated load of all 3 checkpoints. ([`initial_state.json`](file:///D:/Project/ChakrView/experiments/batch34_sft_root_cause/audits/initial_state.json)) |
| **Phase 1** | Reproduce EOS Collapse | **VERIFIED COMPLETE** | Measured EOS-first rate across 10 prompts: Base 10k: 10% (1/10), Step 50 SFT: 50% (5/10), Step 200 SFT: 80% (8/10). ([`reproduced_eos_diagnostics.json`](file:///D:/Project/ChakrView/experiments/batch34_sft_root_cause/evaluation/reproduced_eos_diagnostics.json)) |
| **Phase 2** | Investigate EOS Collapse Systematically | **VERIFIED COMPLETE** | Refuted tokenizer and causal alignment defects; confirmed overfitting on 37 records over 10.8 epochs and prompt-format distribution shift. ([`eos_root_cause.json`](file:///D:/Project/ChakrView/experiments/batch34_sft_root_cause/audits/eos_root_cause.json)) |
| **Phase 3** | Verify Data Provenance Honestly | **VERIFIED COMPLETE** | Audited all 13 records: 11 verified source-grounded, 2 flagged unsupported (`TR-CD-01`, `TR-CD-02`). ([`provenance_record_review.json`](file:///D:/Project/ChakrView/experiments/batch34_sft_root_cause/audits/provenance_record_review.json)) |
| **Phase 4** | Minimal Reproducible Diagnostic | **VERIFIED COMPLETE** | Created synthetic diagnostic verifying causal alignment and target sensitivity. ([`test_batch34_sft_diagnostics.py`](file:///D:/Project/ChakrView/tests/test_batch34_sft_diagnostics.py)) |
| **Phase 5** | Fair Candidate Evaluation | **VERIFIED COMPLETE** | Base Step 10k retains continuous text generation (90% non-EOS); SFT checkpoints deteriorate progressively. |
| **Phase 6** | Training Readiness Gate | **BLOCKED** | Gate upheld: Further training blocked until dataset scaling and length regularization are introduced. |
| **Phase 7** | Regression & Acceptance Testing | **VERIFIED COMPLETE** | Full regression suite: **33/33 tests passed** in 10.14 seconds. |
| **Phase 8** | Final Report & Handoff | **VERIFIED COMPLETE** | Documented evidence and root causes. |
| **Overall** | **Sprint Verdict** | **PARTIAL** | Root cause and audit **VERIFIED COMPLETE**; Model training intentionally **BLOCKED**. |

---

### Root Cause Analysis Details

#### Hypotheses Investigated
1. **Tokenizer & Special Token Insertion (Refuted)**: Tokenizer IDs are BOS=0, EOS=1, PAD=2. Delimiters `<|im_start|>` and `<|im_end|>` decompose cleanly into byte-pair tokens without out-of-bounds indexing or phantom EOS insertion.
2. **Causal Target Shifting Alignment (Refuted)**: In `sft_engine.py`, `target_seq` masks prompt positions with `-100` and aligns logits at prompt token $N-1$ with response token $0$. The loss computation is mathematically correct.
3. **Overfitting & Dataset Scale (Confirmed)**: Batch 30 trained 200 steps on only 37 examples (batch size 2 = 400 total training samples, averaging ~10.8 full epochs). In a 3.4M parameter model, this concentrated exposure rapidly over-penalized non-EOS continuation tokens, boosting token 1 (EOS) to rank 1 in output probabilities.
4. **Prompt Format Mismatch (Confirmed)**: Batch 30 trained on raw string prompts (`What characterizes...`) without `User:` or `<|im_start|>` delimiters. When presented with formatted chat dialogue or unseen prompt structures at inference, the model fails to find pretraining continuation paths and triggers EOS immediately.

---

### Provenance Audit Results

- **Total Records Audited:** 13
- **Verified Source-Grounded (11):** Content verified in source files (e.g., `english_foundation.txt`, `stage_b_systems.txt`, `hindi_computing.txt`, `math_formulas.txt`).
- **Unsupported / Misattributed (2):**
  - `TR-CD-01`: Python attention assertion was attributed to `code_foundation.txt` (actual passage is located in `code_python.txt`).
  - `TR-CD-02`: Matrix attention formula was attributed to `stage_b_algorithms.txt` (actual passage is located in `math_formulas.txt`).

---

### Full Regression Suite Results

All 33 tests passed:
```text
tests\test_pretraining_engine.py ......                                  [ 18%]
tests\test_batch28_stage_d_gate.py ....                                  [ 30%]
tests\test_sft_engine.py ...                                             [ 39%]
tests\test_batch31_chat_runtime.py ......                                [ 57%]
tests\test_batch32_acceptance.py ........                                [ 81%]
tests\test_batch33_integrity.py ...                                      [ 90%]
tests\test_batch34_sft_diagnostics.py ...                                [100%]

============================= 33 passed in 10.14s =============================
```

---

### Final Git Status & Permission Boundary Compliance

- **Current Branch:** `dev/v0.6-v0.7-validation`
- **Tracked Changes:** None (`git diff` is clean).
- **Untracked Artifacts:**
  - `experiments/batch33_integrity/`
  - `tests/test_batch33_integrity.py`
  - `experiments/batch34_sft_root_cause/`
  - `tests/test_batch34_sft_diagnostics.py`
- **Git Actions Taken:** **Zero commits, pushes, merges, or resets executed.** All actions strictly adhered to the Absolute Permission Rule.
