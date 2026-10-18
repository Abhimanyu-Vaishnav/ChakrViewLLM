# INDEPENDENT EVIDENCE AUDIT REPORT

**Batch:** 2461–2500  
**Phase:** Phase 1 — Comprehensive Independent Audit of Gate Evaluator & Evidence Pipeline  
**Audited Target:** [`experiments/batch2421_2460/evaluate_evidence_gates.py`](file:///d:/Project/ChakrView/experiments/batch2421_2460/evaluate_evidence_gates.py)  
**Independent Validator Script:** [`experiments/batch2461_2500/INDEPENDENT_EVIDENCE_VALIDATOR.py`](file:///d:/Project/ChakrView/experiments/batch2461_2500/INDEPENDENT_EVIDENCE_VALIDATOR.py)  
**Machine-Readable Audit Record:** [`experiments/batch2461_2500/INDEPENDENT_EVIDENCE_AUDIT.json`](file:///d:/Project/ChakrView/experiments/batch2461_2500/INDEPENDENT_EVIDENCE_AUDIT.json)  

---

## 1. Executive Summary & Audit Mandate

In Phase 1, we audited the previous batch evaluator ([`evaluate_evidence_gates.py`](file:///d:/Project/ChakrView/experiments/batch2421_2460/evaluate_evidence_gates.py)) to identify any residual assumptions, intermediate JSON dependencies, or gaps in validation logic.

We then engineered a standalone, independent validator ([`INDEPENDENT_EVIDENCE_VALIDATOR.py`](file:///d:/Project/ChakrView/experiments/batch2421_2460/INDEPENDENT_EVIDENCE_VALIDATOR.py)) that eliminates intermediate reporting layers and verifies primary ground-truth artifacts directly.

---

## 2. Analysis of the Previous Gate Evaluator (`batch2421_2460/evaluate_evidence_gates.py`)

### Strengths in Batch 2421–2460:
1. Dynamically computed SHA-256 hashes of canonical files rather than reading static reports.
2. Summed lines from `PER_DOCUMENT_LOSSES.jsonl` rather than copying pre-summarized CE values.
3. Separated working-tree provenance of `session.py` as `PARTIALLY_VERIFIED`, avoiding false claims of full commit-level verification.

### Identified Limitations & Vulnerabilities:
1. **Intermediate JSON Dependency in Sanskrit Check:**
   - In `evaluate_gate_sanskrit_regression()`, the evaluator read `experiments/batch2421_2460/SANSKRIT_REGRESSION_RAW_RECHECK.json` instead of directly computing the Sanskrit and Hindi deltas from `PER_DOCUMENT_LOSSES.jsonl`. If someone manipulated `SANSKRIT_REGRESSION_RAW_RECHECK.json`, the gate would pass without checking the raw lines.
2. **Intermediate JSON Dependency in Training Coverage:**
   - In `evaluate_gate_training_coverage()`, the evaluator read `experiments/batch2421_2460/CPT_TRAINING_COVERAGE_RECHECK.json` and checked `sa_docs == 0` from that JSON. It did not dynamically re-resolve the 782 train document IDs from `SPLIT_V2_MANIFEST.json` against `CORPUS_V2_CLEANED_MANIFEST.json`.
3. **Missing Cross-Consistency Checks in Test Split:**
   - The evaluator verified that `SPLIT_V2_MANIFEST.json` test IDs resolved in `CORPUS_V2_CLEANED_MANIFEST.json`, but did not check that the document IDs in `PER_DOCUMENT_LOSSES.jsonl` exactly matched the document IDs in `SPLIT_V2_MANIFEST.json`.
4. **Hardcoded Tolerance Boundaries:**
   - Tolerance checking `abs(base_ce - 4.478909) > 1e-4` acted as an assertion of an expected target rather than deriving the loss directly from ground-truth document weights.

---

## 3. Independent Validator Architecture (`INDEPENDENT_EVIDENCE_VALIDATOR.py`)

The new validator introduces robust, direct verification principles:
1. **Direct Raw Parsing:** Re-sums all 290 documents from `PER_DOCUMENT_LOSSES.jsonl` directly. Computes exact source NLLs, source CEs, global NLL, global CE, document win counts, and Sanskrit/Hindi deltas without consulting any intermediate report.
2. **Strict Manifest-to-Record Linkage:** Enforces bidirectional ID set equivalence: $\text{Set}(\text{Test IDs in Split Manifest}) \equiv \text{Set}(\text{Document IDs in Raw Loss File})$.
3. **Dynamic Training Set Resolution:** Directly parses `SPLIT_V2_MANIFEST.json` and maps all 782 training documents to `CORPUS_V2_CLEANED_MANIFEST.json`, independently counting sources, languages, and Sanskrit presence.
4. **Adversarial Resilience:** Any mismatch, missing ID, duplicate ID, altered target-token count, or modified loss value causes the validator to immediately report `FAILED` with an explicit reason.

---

## 4. Independent Audit Results

| Validated Dimension | Status | Primary Evidence Used | Observed Result |
| :--- | :---: | :--- | :--- |
| **Artifact Cryptographic Integrity** | **VERIFIED** | Checkpoints, manifests, vocab, merges | 6/6 canonical files dynamically hashed and matched bit-exact. |
| **Test Split 1-to-1 Linkage** | **VERIFIED** | `SPLIT_V2_MANIFEST.json`, `CORPUS_V2_CLEANED_MANIFEST.json`, `PER_DOCUMENT_LOSSES.jsonl` | 290 unique document IDs verified identical across split manifest, corpus, and loss records. |
| **Raw Loss & Regression Recomputation** | **VERIFIED** | 290 records in `PER_DOCUMENT_LOSSES.jsonl` | Global Delta CE: -0.1928, Win Rate: 76.55%. Classical Sanskrit Delta: +0.0776 CE (+8.07% PPL), Hindi Control Delta: -0.0147 CE. |
| **Training Coverage Resolution** | **VERIFIED** | 782 train documents mapped from split to corpus manifest | 782 unique train IDs resolved. Sources: 6, Languages: `en` (636), `hi` (90), `code` (56). Classical Sanskrit in train: 0 docs, 0 tokens. |
| **Regression Test Execution** | **VERIFIED** | `REGRESSION_TEST_RESULTS.json` | 25/25 tests passed cleanly with exit code 0. |
| **Protected File Provenance** | **PARTIALLY_VERIFIED** | `chakrview/runtime/session.py` (mtime, blob SHA-1) | Filesystem mtime (2026-10-11 16:04:15) and commit logs substantiate pre-existence, but diff is uncommitted in working tree. |

- **Derived Overall Verdict:** **`PARTIALLY_VERIFIED`**
- **Decision:** **`READY_FOR_EXPERIMENT_REVIEW`**
