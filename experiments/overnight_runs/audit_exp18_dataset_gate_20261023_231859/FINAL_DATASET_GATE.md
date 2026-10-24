# FINAL DATASET GATE AUDIT REPORT — CHAKRVIEW EXP18

**Experiment**: `exp18_scaled_instruction_format_v1`  
**Timestamp**: 2026-10-23T23:23:00+05:30  
**Evaluator**: TaskEvaluator v0.1 (`chakrview.evaluation.task_evaluator`)  
**Audit Directory**: `experiments/overnight_runs/audit_exp18_dataset_gate_20261023_231859/`  
**Final Verdict**: **`DATASET_AUDIT_PASS`**  

---

## 1. Executive Summary

An independent, rigorous, in-memory dataset quality audit of the **Exp18 Scaled Instruction SFT Generator** (`generate_exp18_datasets.py`) and its candidate banks (`candidate_banks.py`) was conducted. 

All 1,520 records were synthesized in memory without writing any official JSONL files. The audit verified schema conformity, quota allocations, prompt-conditioning contracts, token length bounds, mathematical correctness, RFC 8259 JSON validity, context grounding, abstention ratios, cross-split isolation, and TaskEvaluator scoring.

In accordance with all verified empirical evidence, the final release verdict is:
$$\mathbf{DATASET\_AUDIT\_PASS}$$

---

## 2. Evidence Verification & Subprocess Results

### 2.1 Live Test Suite Execution
- **Command**: `python -m unittest -v experiments/research_v09_conversation_sft/exp18_scaled_instruction_format_v1/tests/test_exp18_generator.py`
- **Subprocess Exit Code**: `0`
- **Execution Time**: 10.506 seconds
- **Test Summary**: **15 / 15 PASSED** (0 failures, 0 errors)
- **Raw Log Captured**: [RAW_TEST_OUTPUT.txt](file:///d:/Project/ChakrView/experiments/overnight_runs/audit_exp18_dataset_gate_20261023_231859/RAW_TEST_OUTPUT.txt)
- **JSON Manifest**: [TEST_EXECUTION.json](file:///d:/Project/ChakrView/experiments/overnight_runs/audit_exp18_dataset_gate_20261023_231859/TEST_EXECUTION.json)

### 2.2 Repository & Protected Asset Integrity
All eight immutable ChakrView assets match their frozen authoritative SHA-256 hashes bit-for-bit:
1. `training_checkpoint_step_10000.pt`: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (**MATCH**)
2. `vocab_4096/vocab.json`: `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` (**MATCH**)
3. `vocab_4096/merges.json`: `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` (**MATCH**)
4. `chakrview/runtime/session.py`: `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` (**MATCH**)
5. `chakrview/brain/model.py`: `1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361` (**MATCH**)
6. `train_exp17_v1.jsonl`: `074fafe5c1dbcbd7467430cb6f5eccd2c11f2f8679ff27e13c0d38cc90bac181` (**MATCH**)
7. `val_exp17_v1.jsonl`: `15d9c942c01fb6afaf6280b7515d9e621ff8df7740be8010daa2be98841f4564` (**MATCH**)
8. `test_exp17_v1.jsonl`: `be10afaa3957713cead5399c2b01e5ad2f4fb61d6355835432c7d6e0fdcc6430` (**MATCH**)

Git status recorded in [INTEGRITY_MANIFEST.json](file:///d:/Project/ChakrView/experiments/overnight_runs/audit_exp18_dataset_gate_20261023_231859/INTEGRITY_MANIFEST.json): branch `dev/v0.6-v0.7-validation`, HEAD commit `f30136a5dc02934180a503fdf0a4c287469bf106`.

---

## 3. Quota & Distribution Verification

Total in-memory records synthesized: **1,520 records** across 8 orthogonal categories:

| Category | Train Count | Val Count | Test Count | Total Count | Quota Verified | Scorable by TaskEvaluator |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`basic_arithmetic`** | 150 | 20 | 20 | 190 | **YES** | **100% SCORED** |
| **`structured_formatting`** | 150 | 20 | 20 | 190 | **YES** | **100% SCORED** |
| **`context_extraction`** | 150 | 20 | 20 | 190 | **YES** | **100% SCORED** |
| **`reference_resolution`** | 150 | 20 | 20 | 190 | **YES** | **100% SCORED** |
| **`instruction_following`** | 150 | 20 | 20 | 190 | **YES** | **100% SCORED** |
| **`factual_qa`** | 150 | 20 | 20 | 190 | **YES** | **100% SCORED** |
| **`uncertainty_abstention`** | 150 | 20 | 20 | 190 | **YES** | **100% SCORED** |
| **`short_explanation`** | 150 | 20 | 20 | 190 | **YES** | **HELD OUT (`NEEDS_REVIEW`)** |
| **TOTAL** | **1,200** | **160** | **160** | **1,520** | **YES** | **87.5% Deterministically Scored** |

Full CSV manifest: [CATEGORY_SPLIT_COUNTS.csv](file:///d:/Project/ChakrView/experiments/overnight_runs/audit_exp18_dataset_gate_20261023_231859/CATEGORY_SPLIT_COUNTS.csv).

---

## 4. In-Memory Data Quality Audit Findings

1. **Prompt Conditioning Contract**:
   - `plain_instruction`: `Instruction: {clean_p}\nResponse:`
   - `context_qa`: `Context: {clean_ctx}\nQuestion: {clean_p}\nAnswer:`
   - 100% of records match the runtime adapter `format_conditioned_prompt` with exact whitespace and token-level fidelity.
2. **Response Token Length Bounds**:
   - Hard bounds `[3, 45]`: Minimum response = **4 tokens**, Maximum response = **44 tokens**. 100% compliance.
   - Provisional target `[5, 35]`: **93.03%** of records fall within target bounds (exceeds the 90.0% requirement). Mean length = **21.87 tokens**.
3. **Arithmetic Correctness**:
   - 190 / 190 arithmetic items independently verified against mathematical calculation. 0 oracle calculation errors.
4. **Structured JSON Formatting**:
   - 190 / 190 structured formatting responses parse into valid RFC 8259 JSON objects. 0 syntax errors.
5. **Context Extraction & Reference Resolution**:
   - 190 / 190 extraction targets are explicitly supported by their context paragraphs.
   - 190 / 190 reference pronouns resolve to valid antecedent entities in context.
6. **Uncertainty & Abstention Consistency**:
   - 3:1 ratio strictly observed (150 train: 113 unanswerable / 37 controls; 20 val: 15 unanswerable / 5 controls; 20 test: 15 unanswerable / 5 controls).
   - 100% of unanswerable responses contain valid refusal phrasing (`does not state`, `does not mention`, `does not record`, `does not specify`).
   - 100% of control responses are grounded in context facts.
7. **Semantic Review Queue**:
   - 190 short explanations + 20 held-out test factual QA records queued in [SEMANTIC_REVIEW_QUEUE.csv](file:///d:/Project/ChakrView/experiments/overnight_runs/audit_exp18_dataset_gate_20261023_231859/SEMANTIC_REVIEW_QUEUE.csv).
   - Short explanations are appropriately withheld from scorable task accuracy and classified as `NEEDS_REVIEW`.

---

## 5. Cross-Split Isolation & Leakage Audit

Audit of [LEAKAGE_FINDINGS.csv](file:///d:/Project/ChakrView/experiments/overnight_runs/audit_exp18_dataset_gate_20261023_231859/LEAKAGE_FINDINGS.csv):
- **Exact Prompt Overlap**:
  - Train $\cap$ Validation: **0 items (0.0%)**
  - Train $\cap$ Test: **0 items (0.0%)**
  - Validation $\cap$ Test: **0 items (0.0%)**
- **Critical Entity / Target Leakage**: **0 occurrences (0.0%)**. Context extraction entities, reference resolution objects, arithmetic operands, JSON keys, and instruction words are completely partitioned and disjoint across splits.
- **Shared N-Grams**: 484 shared 5-grams were audited; 100% of them consist of standard grammatical task boilerplate (e.g. `what is the capital city`, `according to the record what is`, `as a valid json object`). Zero entity-bearing 5-grams leak between Train and Test.

---

## 6. TaskEvaluator v0.1 Compatibility & Known Heuristic Limitations

When evaluated through `TaskEvaluator.evaluate_record`:
- **Scorable Total Records**: 1,330 records across 7 categories.
- **Scorable Task Accuracy on Ground Truth**: **99.7%** (1,326 / 1,330 correct).
- **Identified Evaluator Limitation**: 
  - 4 Factual QA items (`EXP18_TRAIN_FACTUAL_QA_0031`, `0124`, `EXP18_VAL_FACTUAL_QA_0006`, `0007`) score `INCORRECT` because `TaskEvaluator._score_factual_qa` contains legacy hardcoded heuristics designed specifically for Exp17 prompt questions (e.g., if prompt contains `"desert"`, it expects target entity `"antarctic"`; if prompt contains `"particle"`, it expects `"neutron"`).
  - In Exp18, new questions about the Sahara desert (hot desert) or electron/proton (subatomic particles) correctly answer "Sahara Desert" or "electron", which triggers the legacy heuristic.
  - Per instruction guidelines, **generator code was not modified to game the evaluator heuristic**. This limitation in `TaskEvaluator._score_factual_qa` is documented and flagged for an evaluator update prior to checkpoint evaluation.

---

## 7. Deliverables Index

All audit artifacts reside in `experiments/overnight_runs/audit_exp18_dataset_gate_20261023_231859/`:
- [RAW_TEST_OUTPUT.txt](file:///d:/Project/ChakrView/experiments/overnight_runs/audit_exp18_dataset_gate_20261023_231859/RAW_TEST_OUTPUT.txt): Captured stdout/stderr of the test runner with exit code 0.
- [TEST_EXECUTION.json](file:///d:/Project/ChakrView/experiments/overnight_runs/audit_exp18_dataset_gate_20261023_231859/TEST_EXECUTION.json): Verified test execution manifest.
- [DATASET_AUDIT_SUMMARY.json](file:///d:/Project/ChakrView/experiments/overnight_runs/audit_exp18_dataset_gate_20261023_231859/DATASET_AUDIT_SUMMARY.json): Comprehensive numerical audit metrics and category evaluator stats.
- [CATEGORY_SPLIT_COUNTS.csv](file:///d:/Project/ChakrView/experiments/overnight_runs/audit_exp18_dataset_gate_20261023_231859/CATEGORY_SPLIT_COUNTS.csv): Category and split counts matrix.
- [LEAKAGE_FINDINGS.csv](file:///d:/Project/ChakrView/experiments/overnight_runs/audit_exp18_dataset_gate_20261023_231859/LEAKAGE_FINDINGS.csv): 484 audited overlapping 5-grams with classification.
- [SEMANTIC_REVIEW_QUEUE.csv](file:///d:/Project/ChakrView/experiments/overnight_runs/audit_exp18_dataset_gate_20261023_231859/SEMANTIC_REVIEW_QUEUE.csv): Queued records for human / semantic review.
- [INTEGRITY_MANIFEST.json](file:///d:/Project/ChakrView/experiments/overnight_runs/audit_exp18_dataset_gate_20261023_231859/INTEGRITY_MANIFEST.json): Recomputed cryptographic hashes of all protected assets and Git state.
- [FINAL_DATASET_GATE.md](file:///d:/Project/ChakrView/experiments/overnight_runs/audit_exp18_dataset_gate_20261023_231859/FINAL_DATASET_GATE.md): Official dataset gate audit report.

---

## 8. Final Gate Verdict

$$\mathbf{DATASET\_AUDIT\_PASS}$$

Official split files in `experiments/research_v09_conversation_sft/exp18_scaled_instruction_format_v1/data/` remain unwritten. Serialization can now proceed upon explicit user command.
