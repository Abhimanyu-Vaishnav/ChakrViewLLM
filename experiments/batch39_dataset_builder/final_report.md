# CHAKRVIEW — BATCH 39 MASTER SPRINT FINAL REPORT
## Grounded Dataset Builder, Quality Filtering & Safe Training-Gate Integration

### Executive Summary

Batch 39 established a repeatable, deterministic source-document extraction pipeline and candidate dataset builder that extracts grounded supervised candidates from existing raw corpus documents while enforcing strict quality filters, near-duplicate detection, and document-level disjoint splits. It also constructed a fail-closed SFT batch safety guard that prevents all-masked, empty-target, out-of-bounds, or overlength batches from propagating `NaN` loss or corrupted gradients.

Zero Git mutations (`commit`, `push`, `rebase`, `reset`, `clean`) were performed. All historical checkpoints, datasets, and reports were preserved. No SFT training runs were launched.

---

### 1. Initial & Final Git State

- **Branch:** `dev/v0.6-v0.7-validation`
- **HEAD Commit:** `7c3be923acf997d98f693b3509bece86ddf97641`
- **Tracked Modifications Preserved:** [`chakrview/runtime/session.py`](file:///d:/Project/ChakrView/chakrview/runtime/session.py)
- **New Experiment Artifacts:** [`experiments/batch39_dataset_builder/`](file:///d:/Project/ChakrView/experiments/batch39_dataset_builder/)
- **New Tests:** [`tests/test_batch39_dataset_builder.py`](file:///d:/Project/ChakrView/tests/test_batch39_dataset_builder.py)
- **Git Mutation Operations:** 0 commits, 0 pushes, 0 merges, 0 rebases, 0 resets.

---

### 2. Canonical ChakrMicro Baseline Invariance

- **Parameter Count:** 3,443,136
- **Architecture:** 6 layers, d_model 192, 6 heads, hidden_dim 512, context 512, vocab 4096
- **Fresh-Initialization SHA-256:** `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`
- **Verification Status:** **CONFIRMED 100% INVARIANT**

---

### 3. Deterministic Source Document Pipeline & Passage Inventory (Stage A)

Built [`DeterministicCorpusPipeline`](file:///d:/Project/ChakrView/experiments/batch39_dataset_builder/corpus_pipeline.py):
1. **Document Verification:**
   - 27 of 27 documents inspected across 9 categories (`code`, `english`, `hindi`, `hinglish`, `mathematics`, `mixed`, `numbers`, `reasoning`, `sanskrit`).
   - 27 of 27 file bytes SHA-256 digests verified with 100% match against `data/raw/manifest.json`.
   - 0 decoding failures; all documents confirmed valid UTF-8.
   - Saved in [`source_inventory.json`](file:///d:/Project/ChakrView/experiments/batch39_dataset_builder/data/source_inventory.json).
2. **Passage Extraction:**
   - Document-specific parsing: multi-line code block extraction for code files, markdown table row parsing for tables, and paragraph/line parsing for textual documents.
   - Total extracted passages: **6,044**.
   - Total rejected passages: **131** (128 rejected for length $< 25$ characters, 3 rejected for table headers/separators).
   - Saved in [`passage_inventory.jsonl`](file:///d:/Project/ChakrView/experiments/batch39_dataset_builder/data/passage_inventory.jsonl).

---

### 4. Candidate Example Construction & Quality Filtering (Stage B)

Built [`CandidateExampleBuilder`](file:///d:/Project/ChakrView/experiments/batch39_dataset_builder/candidate_builder.py) and [`QualityAndDeduplicationFilter`](file:///d:/Project/ChakrView/experiments/batch39_dataset_builder/candidate_builder.py):
1. **Separation of Extraction from Synthetic Invention:**
   - Evaluated 6,044 passages against grounded domain templates.
   - Filtered out 379 synthetic repetitive kernel functions (`def evaluate_tensor_kernel_vXXXX`), 180 identical observation lines, and repetitive dev log Q&A pairs.
   - Raw candidate count: **353**.
2. **Deduplication & Near-Duplicate Filtering:**
   - Applied exact duplicate hash filtering across prompt, response, and passage evidence text.
   - Applied `difflib.SequenceMatcher` near-duplicate filtering with a threshold of **0.75**.
   - Candidates retained after filtering: **46 distinct candidates** (39 train, 7 val).
   - Rejected duplicates: **307**.
   - Saved in [`candidate_records.jsonl`](file:///d:/Project/ChakrView/experiments/batch39_dataset_builder/data/candidate_records.jsonl).
3. **Multi-Dimensional Validation Statuses:**
   - Schema Validity: **46 PASS**
   - Source File Validity: **46 PASS**
   - Source Hash Validity: **46 PASS**
   - Evidence Passage Matching: **46 PASS**
   - Answer-to-Evidence Support: **46 PASS**
   - Technical Correctness: **46 PASS**
   - Duplicate Risk: **46 PASS**
   - Code/Math Validation: **16 PASS**, **30 N/A** (behaviorally validated using [`SafeCodeExecutionHarness`](file:///d:/Project/ChakrView/experiments/batch38_audit_hardening/audits/validator.py))
   - Human Review Status: **46 PENDING_HUMAN_REVIEW** (automated inspection held distinct from human review)

---

### 5. Leakage-Resistant Split Planning

Built [`split_manifest.json`](file:///d:/Project/ChakrView/experiments/batch39_dataset_builder/data/split_manifest.json):
- Partitioning enforced at the **source-document level**: every document belongs exclusively to `train` or `val`.
- Document partition: **17 train documents, 10 val documents**.
- **Document Overlap:** **0 (Strictly Disjoint)**.
- Candidate inheritance: All 46 candidates inherit the partition of their parent source document (39 train, 7 val).
- Domain Sufficiency Accounting:
  - `code`, `english`, `hindi`, `hinglish`, `mathematics`, `mixed`, `numbers`, `sanskrit`: Disjoint documents present in both train and val.
  - `reasoning`: Contains only 1 document (`reasoning/stage_b_procedures.txt`), flagged as `INSUFFICIENT_DOCUMENTS_FOR_VAL_SPLIT` rather than silently leaking across splits.

---

### 6. Dataset Quality Dashboard

Full metrics recorded in [`quality_summary.json`](file:///d:/Project/ChakrView/experiments/batch39_dataset_builder/data/quality_summary.json):
- **Source Documents:** 27 inspected, 27 SHA matches, 0 failures.
- **Passages:** 6,044 extracted, 131 rejected.
- **Candidates:** 353 raw, 46 distinct retained, 307 duplicates rejected.
- **Domain Breakdown:** Code: 16, English: 12, Mathematics: 9, Hindi: 6, Hinglish: 2, Sanskrit: 1.
- **Unresolved Quality Issues:**
  1. Dataset has 46 distinct candidates, well below the $\ge 500$ threshold required to prevent severe SFT overfitting.
  2. All 46 candidates remain `PENDING_HUMAN_REVIEW`.
  3. Category `reasoning` has only 1 source document and cannot support a disjoint validation split.

---

### 7. SFT Batch Safety Guard

Built [`SFTBatchSafetyGuard`](file:///d:/Project/ChakrView/experiments/batch39_dataset_builder/sft_batch_safety.py):
- **All-Masked Targets:** Detected and safely rejected before forward pass. Returns `None` loss and status `ALL_TARGETS_MASKED`, preventing `NaN` loss propagation.
- **Target Vocab Out of Bounds:** Token IDs outside $[0, 4096)$ rejected before loss calculation with status `TARGET_VOCAB_OUT_OF_BOUNDS`.
- **Missing EOS Supervision:** Target tensors lacking the EOS token ID (`1`) rejected with status `MISSING_EOS_SUPERVISION`.
- **Overlength Sequences:** Sequences exceeding `max_seq_len=512` rejected with status `EXCEEDS_CONTEXT_LENGTH`.
- **Valid Batches:** Compute finite scalar loss and valid parameter gradients.

---

### 8. Enforced SFT Readiness Gate

Machine-readable gate output in [`sft_readiness.json`](file:///d:/Project/ChakrView/experiments/batch39_dataset_builder/gates/sft_readiness.json):
- **`DATASET_READY`:** **`false`**
- **`PIPELINE_READY`:** **`true`**
- **`BASELINE_READY`:** **`true`**
- **`SFT_READY`:** **`false`**
- **Overall Decision:** **`BLOCKED`**

#### Blocking Reasons:
1. **Record Count Below Overfitting Threshold:** Dataset has 46 distinct candidates (required $\ge 500$).
2. **Pending Human Review:** All 46 candidates are marked `PENDING_HUMAN_REVIEW`.
3. **Insufficient Domain Split Coverage:** Domain `reasoning` contains only 1 source document and lacks a disjoint validation split document.

---

### 9. Test Suite Execution Results

Executed regression suite:
```text
tests\test_pretraining_engine.py ......                                  [  8%]
tests\test_batch28_stage_d_gate.py ....                                  [ 14%]
tests\test_sft_engine.py ...                                             [ 18%]
tests\test_batch31_chat_runtime.py ......                                [ 27%]
tests\test_batch32_acceptance.py ........                                [ 39%]
tests\test_batch33_integrity.py ...                                      [ 43%]
tests\test_batch34_sft_diagnostics.py ...                                [ 47%]
tests\test_batch35_recovery.py ...                                       [ 52%]
tests\test_batch36_readiness.py ...                                      [ 56%]
tests\test_batch37_independent_audit.py ....                             [ 62%]
tests\test_batch38_audit_hardening.py .............                      [ 81%]
tests\test_batch39_dataset_builder.py .............                      [100%]

============================= 69 passed in 9.71s ==============================
```
- **Exit Code:** 0
- **Total Tests Passed:** 69 / 69

---

### 10. Remaining Uncertainties & Smallest Justified Next Step

#### Remaining Uncertainties:
1. The 27 existing raw corpus documents contain high repetition in Stage B files (e.g. 379 synthetic tensor kernels, 180 scenario variations), yielding only 46 distinct high-quality candidates under strict deduplication. Additional distinct corpus documents are required to reach 500 candidates.
2. Independent human review has not yet been conducted on the 46 generated candidates.

#### Smallest Justified Next Step:
Source or author additional diverse, non-repetitive raw corpus texts across core domains (reasoning, mathematics, literature, systems) to expand the raw corpus inventory beyond 27 files, allowing the deterministic pipeline to extract at least 500 distinct, human-reviewable candidates.
