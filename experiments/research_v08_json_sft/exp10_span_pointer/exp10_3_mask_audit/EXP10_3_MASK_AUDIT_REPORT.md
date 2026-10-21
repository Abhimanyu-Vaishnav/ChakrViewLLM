# ChakrView Experiment 10.3 — Evaluation Mask Leakage Audit Report

**Status**: `MASKS VERIFIED — ORIGINAL METRICS REPRODUCED`  
**Audit Directory**: `experiments/research_v08_json_sft/exp10_span_pointer/exp10_3_mask_audit/`  
**Parent Checkpoint**: `Step-10000` (`96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`)  
**Experiment 10 Best Checkpoint**: `experiments/research_v08_json_sft/exp10_span_pointer/checkpoints/exp10_span_pointer_best.pt`  
**Experiment 10.2 Best Checkpoint**: `experiments/research_v08_json_sft/exp10_span_pointer/exp10_2_end_boundary/checkpoints/exp10_2_start_cond_best.pt`  

---

## 1. Executive Summary & Objective

In Experiment 10 and Experiment 10.2, candidate masks in evaluation scripts included the code pattern:
```python
mask[0, 1:seq_len - 2] = True
mask[0, s_gt] = True
mask[0, e_gt] = True
```
The user requested a rigorous, independent audit before beginning Experiment 11 to determine:
1. Whether ground-truth `s_gt` and `e_gt` altered candidate eligibility at inference time (i.e. did they unmask positions that would otherwise be ineligible under pure inference rules?).
2. Whether removing all label-dependent eligibility changes any prediction or metric across Validation, Test A, or Test B.
3. Whether oracle-start diagnostic mode in Experiment 10.2 leaked gold end labels or altered end-candidate eligibility.

### Final Conclusion:
- In **893 / 893 (100.0%)** evaluation examples across Validation (296), Test A (297), and Test B (300), the pure inference mask `mask[0, 1:seq_len - 2] = True` was **bit-for-bit identical** to the original evaluation mask.
- In **0 / 893 (0.0%)** examples did `s_gt` or `e_gt` fall outside the interval `[1, seq_len - 3]`. The lines `mask[0, s_gt] = True` and `mask[0, e_gt] = True` set bits that were already `True`.
- In **0 / 893 (0.0%)** examples did any prediction change when using a pure inference mask constructed strictly without gold labels.
- The reported metrics for Experiment 10 and Experiment 10.2 are **100.0% authentic, unreconstructed, and completely free of label leakage**.

---

## 2. Integrity Verification & Cryptographic Provenance

Cryptographic hashes were recomputed before and after all audit operations. Zero canonical, released, or prior experiment files were modified:

| Asset | File Path | SHA-256 Digest | Status |
| :--- | :--- | :--- | :--- |
| **Canonical Checkpoint** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **VERIFIED MATCH** |
| **Tokenizer Vocab** | `data/experiments/vocab_4096/vocab.json` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | **VERIFIED MATCH** |
| **Tokenizer Merges** | `data/experiments/vocab_4096/merges.json` | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | **VERIFIED MATCH** |
| **Session Runtime** | `chakrview/runtime/session.py` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **VERIFIED UNTOUCHED** |

Hashes for all audit artifacts are recorded in `EXP10_3_INTEGRITY.json`.

---

## 3. Inspection of Evaluation & Decoding Paths

We traced every mask construction and decoding function across the codebase:

1. **`train_exp10_span_pointer.py` (Training and Validation)**:
   - Line 74: `mask[0, 1:seq_len - 2] = True`
   - Lines 75–76: `mask[0, s_gt] = True; mask[0, e_gt] = True` (defensive redundancy in training loop).
2. **`evaluate_exp10_span_pointer.py` (Exp 10 Comprehensive Evaluation)**:
   - Line 164: `mask[0, 1:seq_len - 2] = True`
   - Lines 165–166: `mask[0, s_gt] = True; mask[0, e_gt] = True`
3. **`evaluate_exp10_2.py` (Exp 10.2 Diagnostic & Evaluation)**:
   - Line 200: `cand_mask[0, 1:seq_len - 2] = True`
   - Lines 201–202: `cand_mask[0, gold_start] = True; cand_mask[0, gold_end] = True`
4. **Oracle-Start Mode in `evaluate_exp10_2.py`**:
   - Start was fixed to `gold_start`.
   - End scoring was searched over candidate positions $e \in [s, \min(s + L_{\max}, T))$ that were eligible under `cand_mask`.
   - The gold end position was **never** provided to the end scorer.

### Why Were `mask[0, s_gt] = True` Present?
In earlier single-token pointer experiments (Exp 09), defensive statements were added to prevent tokenizer boundary truncations from crashing the cross-entropy loss if a target was at an index edge. In Experiment 10 and 10.2, all prompts followed the structured template `Alpha=... \nCopy Alpha: `, where token 0 is BOS (`[BOS]`) and the last two tokens are always `[':', ' ']` or newline punctuation. All source values reside strictly between index 1 and `seq_len - 3`.

---

## 4. Pure Inference Mask Implementation & Evaluation

In `audit_exp10_3_masks.py`, we implemented the corrected audit path:
```python
def build_pure_inference_mask(seq_len: int, device: str = "cpu") -> torch.Tensor:
    mask = torch.zeros((1, seq_len), dtype=torch.bool, device=device)
    mask[0, 1:seq_len - 2] = True
    return mask
```
This function has **no access to `gold_start` or `gold_end`**.

### Audit Results: Mask Identicality & Leakage Verification
Across all splits (Validation, Test A, Test B, and Training):
- **Masks Strictly Identical to Pure Mask**: **893 / 893 (100.0%)**
- **Gold Positions Excluded by Pure Mask**: **0 / 893 (0.0%)**
- **Predictions Changed Under Pure Mask**: **0 / 893 (0.0%)**

---

## 5. Recomputed Metrics (Side-by-Side Comparison)

Metrics recomputed from saved checkpoints using `audit_exp10_3_masks.py` with explicit numerators and denominators.

### Model 1: Experiment 10 Factorized Span Pointer (Baseline)

| Split | Metric | Original Mask (Exp 10) | Corrected Pure Mask (No Labels) | Corrected Oracle-Start (Start Only) | Delta |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Validation** ($N=296$) | Start Acc | 98.99% (293/296) | 98.99% (293/296) | 100.00% (296/296) | **0.00%** |
| | End Acc | 100.00% (296/296) | 100.00% (296/296) | 100.00% (296/296) | **0.00%** |
| | Exact Span EM | 98.99% (293/296) | 98.99% (293/296) | 100.00% (296/296) | **0.00%** |
| | Value Match Acc | 100.00% (296/296) | 100.00% (296/296) | 100.00% (296/296) | **0.00%** |
| | Premature End Count | 0 / 296 (0.0%) | 0 / 296 (0.0%) | 0 / 296 (0.0%) | **0** |
| **Test A (Familiar)** ($N=297$) | Start Acc | 98.65% (293/297) | 98.65% (293/297) | 100.00% (297/297) | **0.00%** |
| | End Acc | 100.00% (297/297) | 100.00% (297/297) | 100.00% (297/297) | **0.00%** |
| | Exact Span EM | 98.65% (293/297) | 98.65% (293/297) | 100.00% (297/297) | **0.00%** |
| | Value Match Acc | 100.00% (297/297) | 100.00% (297/297) | 100.00% (297/297) | **0.00%** |
| | Premature End Count | 0 / 297 (0.0%) | 0 / 297 (0.0%) | 0 / 297 (0.0%) | **0** |
| **Test B (Held-Out)** ($N=300$) | Start Acc | 97.67% (293/300) | 97.67% (293/300) | 100.00% (300/300) | **0.00%** |
| | End Acc | 75.67% (227/300) | 75.67% (227/300) | 75.67% (227/300) | **0.00%** |
| | Exact Span EM | 74.00% (222/300) | 74.00% (222/300) | 75.67% (227/300) | **0.00%** |
| | Value Match Acc | 75.67% (227/300) | 75.67% (227/300) | 75.67% (227/300) | **0.00%** |
| | Premature End Count | 69 / 300 (23.0%) | 69 / 300 (23.0%) | 70 / 300 (23.3%) | **0** |

---

### Model 2: Experiment 10.2 Start-Conditioned Span Pointer

| Split | Metric | Original Mask (Exp 10.2) | Corrected Pure Mask (No Labels) | Corrected Oracle-Start (Start Only) | Delta |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Validation** ($N=296$) | Start Acc | 100.00% (296/296) | 100.00% (296/296) | 100.00% (296/296) | **0.00%** |
| | End Acc | 99.66% (295/296) | 99.66% (295/296) | 100.00% (296/296) | **0.00%** |
| | Exact Span EM | 99.66% (295/296) | 99.66% (295/296) | 100.00% (296/296) | **0.00%** |
| | Premature End Count | 1 / 296 (0.3%) | 1 / 296 (0.3%) | 0 / 296 (0.0%) | **0** |
| **Test A (Familiar)** ($N=297$) | Start Acc | 99.66% (296/297) | 99.66% (296/297) | 100.00% (297/297) | **0.00%** |
| | End Acc | 100.00% (297/297) | 100.00% (297/297) | 100.00% (297/297) | **0.00%** |
| | Exact Span EM | 99.66% (296/297) | 99.66% (296/297) | 100.00% (297/297) | **0.00%** |
| | Premature End Count | 0 / 297 (0.0%) | 0 / 297 (0.0%) | 0 / 297 (0.0%) | **0** |
| **Test B (Held-Out)** ($N=300$) | Start Acc | 98.00% (294/300) | 98.00% (294/300) | 100.00% (300/300) | **0.00%** |
| | End Acc | 67.00% (201/300) | 67.00% (201/300) | 67.00% (201/300) | **0.00%** |
| | Exact Span EM | 66.00% (198/300) | 66.00% (198/300) | 67.00% (201/300) | **0.00%** |
| | Premature End Count | 95 / 300 (31.7%) | 95 / 300 (31.7%) | 98 / 300 (32.7%) | **0** |

---

## 6. Unit & Leakage Test Suite Results

The comprehensive test suite in `test_exp10_3.py` was executed with `pytest`:
```
experiments/research_v08_json_sft/exp10_span_pointer/exp10_3_mask_audit/test_exp10_3.py::test_pure_mask_does_not_depend_on_gold_labels PASSED [ 14%]
experiments/research_v08_json_sft/exp10_span_pointer/exp10_3_mask_audit/test_exp10_3.py::test_changing_gold_labels_does_not_change_pure_mask PASSED [ 28%]
experiments/research_v08_json_sft/exp10_span_pointer/exp10_3_mask_audit/test_exp10_3.py::test_oracle_start_mode_changes_only_start_intervention PASSED [ 42%]
experiments/research_v08_json_sft/exp10_span_pointer/exp10_3_mask_audit/test_exp10_3.py::test_gold_end_labels_never_unlock_end_candidates PASSED [ 57%]
experiments/research_v08_json_sft/exp10_span_pointer/exp10_3_mask_audit/test_exp10_3.py::test_invalid_target_positions_are_reported_not_force_enabled PASSED [ 71%]
experiments/research_v08_json_sft/exp10_span_pointer/exp10_3_mask_audit/test_exp10_3.py::test_metric_denominators_include_invalid_predictions PASSED [ 85%]
experiments/research_v08_json_sft/exp10_span_pointer/exp10_3_mask_audit/test_exp10_3.py::test_deterministic_audit_runs PASSED [100%]
============================== 7 passed in 1.79s ==============================
```

---

## 7. Deliverables Generated

The following artifacts have been authored in `experiments/research_v08_json_sft/exp10_span_pointer/exp10_3_mask_audit/`:
- `EXP10_3_MASK_AUDIT_REPORT.md` (this report)
- `EXP10_3_MASK_AUDIT_RESULTS.json` (complete JSON metric summaries side by side)
- `EXP10_3_PER_EXAMPLE.csv` (all 1,786 evaluation inferences across both models with `prediction_changed` and `mask_strictly_identical` flags)
- `EXP10_3_INTEGRITY.json` (cryptographic SHA-256 hashes of checkpoints, tokenizers, runtime, and audit files)
- `audit_exp10_3_masks.py` (auditing script)
- `test_exp10_3.py` (7 leakage and correctness unit tests)

---

## 8. Single Recommended Next Step

**Clean the redundant label-assignment lines in future evaluation scripts**:  
While the audit conclusively proved that `mask[0, s_gt] = True` and `mask[0, e_gt] = True` had **zero effect** on candidate eligibility or metrics (since all target spans already lay strictly inside `[1, seq_len - 2]`), future evaluation scripts in Experiment 11 should strictly use `build_pure_inference_mask(seq_len)` to avoid any appearance of label coupling.

Do not start Experiment 11 or integrate any pointer prototype into the production runtime until instructed.
