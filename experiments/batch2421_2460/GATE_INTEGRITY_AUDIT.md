# GATE INTEGRITY AUDIT REPORT

**Batch:** 2421–2460  
**Audit Phase:** Phase 1 — Systematic Analysis & Repair of Evidence Gate Logic  
**Audited File:** [`experiments/batch2381_2420/FINAL_EVIDENCE.json`](file:///d:/Project/ChakrView/experiments/batch2381_2420/FINAL_EVIDENCE.json)  
**Deliverable Tool:** [`experiments/batch2421_2460/evaluate_evidence_gates.py`](file:///d:/Project/ChakrView/experiments/batch2421_2460/evaluate_evidence_gates.py)  
**Machine-Readable Audit:** [`experiments/batch2421_2460/GATE_INTEGRITY_AUDIT.json`](file:///d:/Project/ChakrView/experiments/batch2421_2460/GATE_INTEGRITY_AUDIT.json)  

---

## 1. Executive Summary & Audit Mandate

In previous batches (e.g. Batch 2341–2380 and Batch 2381–2420), `FINAL_EVIDENCE.json` relied on high-level summaries or static assertions. Several gates:
1. Marked status as `VERIFIED` unconditionally or based on manually populated JSON files without recomputing underlying values.
2. Treated a report's own assertion of success as evidence of its correctness (circular reasoning).
3. Hardcoded expected values rather than recalculating them dynamically from raw evidence.
4. Conflated filesystem `st_mtime` with proof of diff provenance.
5. Hardcoded test pass counts (e.g. "25/25 passed") without parsing actual test runner output files.

This audit details the specific flaws in each prior gate and defines the rigorous replacement logic implemented in [`evaluate_evidence_gates.py`](file:///d:/Project/ChakrView/experiments/batch2421_2460/evaluate_evidence_gates.py).

---

## 2. Gate-by-Gate Vulnerability Analysis of Batch 2381–2420

### Gate 1: `source_identity`
- **Prior Flaw:** Trusted `SOURCE_IDENTITY_AUDIT.json` summary numbers directly without re-verifying that the document IDs in `SPLIT_V2_MANIFEST.json` and `CORPUS_V2_CLEANED_MANIFEST.json` matched 1-to-1 without missing entries or duplicates.
- **Repair in Batch 2421–2460:** The new evaluator reads `SPLIT_V2_MANIFEST.json` and `CORPUS_V2_CLEANED_MANIFEST.json` directly, checks ID completeness, verifies zero missing IDs, zero duplicate IDs, and computes source counts directly from raw documents.

### Gate 2: `raw_loss_aggregation`
- **Prior Flaw:** Relied on `LOSS_AGGREGATION_AUDIT.json` and accepted a discrepancy range without dynamically re-summing all 290 lines in `PER_DOCUMENT_LOSSES.jsonl`.
- **Repair in Batch 2421–2460:** The new evaluator parses `PER_DOCUMENT_LOSSES.jsonl`, accumulates $NLL$ across all lines, compares exact sums against the reported values, and verifies precision margins mathematically.

### Gate 3: `baseline_vs_candidate_reproducibility`
- **Prior Flaw:** Trusted `FRESH_EVALUATION.json` summary metrics without independently validating that candidate win counts and delta cross-entropies corresponded to the document records in `PER_DOCUMENT_LOSSES.jsonl`.
- **Repair in Batch 2421–2460:** The new evaluator computes document win rate directly by comparing `candidate_exact_loss < baseline_exact_loss` for all 290 document records.

### Gate 4: `sanskrit_coverage_and_regression`
- **Prior Flaw:** Stated "training split confirmed to have 0 Classical Sanskrit tokens" merely by checking if `sa` was listed in the summary `train_languages`. It did not resolve actual document IDs to ensure that no unlabeled Sanskrit texts existed in the training split.
- **Repair in Batch 2421–2460:** Consumes `CPT_TRAINING_COVERAGE_RECHECK.json` which resolved all 782 train documents, tokenized texts with the canonical BPE tokenizer, and verified zero Sanskrit documents and zero Sanskrit target tokens. Consumes raw per-document records to confirm the +0.0776 CE regression.

### Gate 5 & 6: `checkpoint_identity` and `canonical_artifact_integrity`
- **Prior Flaw:** Verified hashes against expected values in a single static JSON file without dynamically re-hashing large binary checkpoint files on disk during the gate check.
- **Repair in Batch 2421–2460:** Evaluator computes SHA-256 digests dynamically for all artifacts and cross-references them against immutable expected constants.

### Gate 7: `protected_file_provenance`
- **Prior Flaw:** Unconditionally marked `VERIFIED` on the basis of filesystem `st_mtime` alone, violating the principle that `mtime` does not prove uncommitted working tree diff content.
- **Repair in Batch 2421–2460:** Marked `PARTIALLY_VERIFIED`. The evaluator distinguishes historical documentation from committed tree hashes, recording the honest state of the working tree.

### Gate 8: `regression_tests`
- **Prior Flaw:** Referenced a directory (`tests/`) and asserted "25/25 tests passed cleanly" without reading an actual test execution log or exit code.
- **Repair in Batch 2421–2460:** Evaluator parses `REGRESSION_TEST_RESULTS.json` and `REGRESSION_TEST_RESULTS.txt`, verifies exit code == 0, passed == 25, failed == 0, and collected == 25.

---

## 3. Evaluator Architecture

The new script [`experiments/batch2421_2460/evaluate_evidence_gates.py`](file:///d:/Project/ChakrView/experiments/batch2421_2460/evaluate_evidence_gates.py) enforces:
- Strictly no hardcoded status or verdict.
- Missing evidence raises `UNVERIFIED`.
- Contradictory evidence raises `FAILED`.
- Any critical non-pass degrades overall verdict from `VERIFIED` to `PARTIALLY_VERIFIED` or `FAILED`.
- The final recommendation is deterministically derived from gate statuses.
