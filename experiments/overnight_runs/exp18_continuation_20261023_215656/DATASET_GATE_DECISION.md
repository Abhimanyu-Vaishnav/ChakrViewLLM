# DATASET GATE DECISION — CHAKRVIEW EXP18

**Experiment**: `exp18_scaled_instruction_format_v1`  
**Timestamp**: 2026-10-23T23:11:00+05:30  
**Evaluator**: TaskEvaluator v0.1 (`chakrview.evaluation.task_evaluator`)  
**Decision**: **`GENERATOR_READY_FOR_DATASET_AUDIT`**  

---

## 1. Decision Determination

The Exp18 dataset generator, deterministic candidate banks, and test suite have successfully completed Phase 1 (Safe State Inspection), Phase 2 (Resume Implementation Safely), and Phase 3 (Tests Only). 

All 15 focused generator unit tests passed with exit code 0 (15/15 passed in 16.3s). The 8 immutable repository artifacts match their cryptographic SHA-256 hashes bit-for-bit. No official dataset split files have been written to disk.

Therefore, the official gate status is:
$$\mathbf{GENERATOR\_READY\_FOR\_DATASET\_AUDIT}$$

---

## 2. Gate Verification Checklist

| Criterion | Requirement | Empirical Result | Gate Status |
| :--- | :--- | :--- | :---: |
| **Protected Artifact Hashes** | 100% bitwise parity on 8 protected assets | 8 / 8 SHA-256 hashes match authoritative values | **PASS** |
| **Tokenizer Serialization API** | Verify via official `load_tokenizer_artifacts` | Loaded successfully, roundtrip verified | **PASS** |
| **Split Quotas & Record Counts** | Exactly 1,200 Train / 160 Val / 160 Test (1,520 total) | 1,200 / 160 / 160 synthesized in memory | **PASS** |
| **Per-Category Distribution** | Exactly 150 / 20 / 20 across all 8 capabilities | 8 categories balanced at 150 / 20 / 20 | **PASS** |
| **Natural Language Entities** | Zero synthetic tokens (`Apollo_train_0`, `alpha_test_1`) | 100% natural names, entities, and JSON keys | **PASS** |
| **Split Isolation** | 0% prompt overlap; 0% entity/number leakage | 0 cross-split prompt overlaps; disjoint entities | **PASS** |
| **Gate v0.1 Disjointness** | Zero blacklisted entities; similarity < 0.70 | Zero blacklist hits; max similarity < 0.70 | **PASS** |
| **Arithmetic Oracle** | 100% mathematically verified | 190 / 190 arithmetic items evaluate CORRECT | **PASS** |
| **Structured JSON Validity** | 100% valid RFC 8259 JSON objects | 190 / 190 JSON items evaluate CORRECT | **PASS** |
| **Abstention Consistency** | 3:1 unanswerable-to-control ratio | 3:1 ratio strictly observed; controls grounded | **PASS** |
| **Short Explanation Policy** | Held out from scorable accuracy metrics | 100% classified as `NEEDS_REVIEW` | **PASS** |
| **Response Length Bounding** | Target `[5, 35]` tokens >= 90%; hard bounds `[3, 45]` | 93.0% in target range; min 4, max 44 tokens | **PASS** |
| **Focused Test Suite** | 100% pass rate on test suite | 15 / 15 tests PASSED (exit code 0) | **PASS** |
| **Write Protection Gating** | 0 official files written before user authorization | `data/` directory confirmed 100% empty | **PASS** |

---

## 3. Next Steps & Authorization Path

To proceed from `GENERATOR_READY_FOR_DATASET_AUDIT` to official dataset generation:
1. User provides explicit confirmation to serialize official Exp18 dataset splits.
2. The generator will be invoked with `--write-splits` to write `train_exp18_v1.jsonl`, `val_exp18_v1.jsonl`, and `test_exp18_v1.jsonl` to `experiments/research_v09_conversation_sft/exp18_scaled_instruction_format_v1/data/`.
3. Compute and seal final SHA-256 hashes in `ARTIFACT_MANIFEST.json` prior to any fine-tuning preflight checks.
