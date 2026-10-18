# FINAL EVIDENCE REPORT — BATCH 2501–2540

**Batch:** 2501–2540  
**Phase:** Phase 6 — Final Evidence Gate  
**Validator Tool:** [`experiments/batch2501_2540/HARDENED_EVIDENCE_VALIDATOR.py`](file:///d:/Project/ChakrView/experiments/batch2501_2540/HARDENED_EVIDENCE_VALIDATOR.py)  
**Machine-Readable Record:** [`experiments/batch2501_2540/FINAL_EVIDENCE.json`](file:///d:/Project/ChakrView/experiments/batch2501_2540/FINAL_EVIDENCE.json)  
**Overall Mission Status:** **`PARTIALLY_VERIFIED`**  
**Final Project Decision:** **`REMAIN_IN_AUDIT_MODE`**  

---

## 1. Executive Summary

This report establishes the final evidence gate for Batch 2501–2540. Every gate status was dynamically derived by the hardened validator from primary ground-truth assets (checkpoints, manifests, raw JSONL records, and test runner outputs).

The overall project decision is strictly designated as **`REMAIN_IN_AUDIT_MODE`** because:
1. The uncommitted working copy diff in `chakrview/runtime/session.py` remains uncommitted in the local tree (classified honestly as `PARTIALLY_VERIFIED`).
2. Our rigorous methodological audit of the Sanskrit curriculum experiment protocol determined that it requires concrete scientific revisions (detailed in [`PREREGISTRATION_REQUIRED_CHANGES.md`](file:///d:/Project/ChakrView/experiments/batch2501_2540/PREREGISTRATION_REQUIRED_CHANGES.md)) prior to human review or experiment authorization.

---

## 2. Gate Verification Matrix

| Gate Identifier | Status | Primary Evidence Path | Validation Check Performed | Observed Result | Limitations & Discrepancies |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **`artifact_integrity`** | **VERIFIED** | Checkpoints, manifests, vocab, merges | SHA-256 byte recomputation & size sanity check | All 6 canonical assets matched expected digests bit-exact. | None. |
| **`test_split_linkage`** | **VERIFIED** | `SPLIT_V2_MANIFEST.json`, `CORPUS_V2_CLEANED_MANIFEST.json`, `PER_DOCUMENT_LOSSES.jsonl` | Bidirectional set equality & corpus source fidelity | 290 test document IDs verified unique, resolved, and source-faithful. | None. |
| **`raw_loss_and_regression`** | **VERIFIED** | `PER_DOCUMENT_LOSSES.jsonl` | Per-document float finite check, NLL consistency, source summation | Global Delta CE=-0.1928, Win Rate=76.55%. Classical Sanskrit Delta=+0.0776 CE, Hindi Delta=-0.0147 CE. | None. |
| **`training_coverage`** | **VERIFIED** | `TRAINING_COVERAGE_PER_DOCUMENT.jsonl`, `SPLIT_V2_MANIFEST.json` | Resolution of 782 train documents and canonical tokenization | 782 train docs resolved across 6 sources. Classical Sanskrit in train: 0 docs, 0 target tokens. | None. |
| **`regression_tests`** | **VERIFIED** | `REGRESSION_TEST_RESULTS.json`, `REGRESSION_TEST_RESULTS.txt` | Subprocess exit code and pass count cross-verification | 25/25 regression tests passed cleanly with exit code 0. | None. |
| **`preregistration_completeness`** | **VERIFIED** | `SANSKRIT_CURRICULUM_PREREGISTRATION.json` | Verification of structural keys and 4 experimental arms | Preregistration contains all 4 experimental arms and decision rules. | Protocol requires scientific revisions specified in Phase 5. |
| **`protected_file_provenance`** | **PARTIALLY_VERIFIED** | `chakrview/runtime/session.py` (mtime, blob SHA-1) | Forensic filesystem timestamp and git blob inspection | Filesystem mtime (`2026-10-11 16:04:15`) substantiates pre-existence, but diff is uncommitted in working tree. | Unstaged working tree diff lacks immutable commit object on HEAD. |

---

## 3. Decision Rationale
- **Model Training:** Exactly 0 steps executed. Zero model training occurred in this batch.
- **Empirical Grounding:** All metrics, loss sums, artifact digests, and zero Sanskrit training coverage are 100% verified from primary records.
- **Actionable Path Forward:** Maintain audit mode until the experiment preregistration protocol is updated to address the five required scientific amendments.
