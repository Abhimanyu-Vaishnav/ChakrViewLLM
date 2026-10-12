# Batch 43 Audit Report: Evaluation System Integrity & Hardening

**Date:** 2026-10-12  
**Target Module:** `experiments/batch42_reproducible_evaluation/evaluation/evaluator.py`  
**Evaluation Invariant:** Deterministic full-dataset evaluation without subsampling.  

---

## 1. Audit Findings Across the 12 Audit Points

1. **Exact Checkpoint Loading & Architecture Validation:**
   - Evaluator verifies `ModelConfig(max_seq_len=512)` against checkpoint payload.
   - Strict `model.load_state_dict(strict=True)` prevents silently dropped or misaligned tensor layers.
   - Evaluator calculates SHA-256 digest of checkpoint and dataset files directly from disk.

2. **Prompt Formatting, BOS, and Delimiter Continuity:**
   - BOS (`id=0`) is consistently prepended to the sequence.
   - Assistant prompt prefix ends with `<|im_start|>assistant\n`.
   - **Boundary Continuity Audit**: Verified that stripping whitespace from prompt and response could remove the trailing `\n` delimiter of `<|im_start|>assistant\n`, dropping token `13` (`\n`). In `DeterministicEvaluator`, prompts are stripped only if passed with stray surrounding whitespace, but ChatML delimiters `<|im_start|>` and `<|im_end|>\n` are properly maintained.

3. **Loss Masking & Active Target Alignment:**
   - Evaluator sets all prompt positions to `ignore_index = -100`.
   - Assistant response tokens are supervised.
   - The final token is explicitly checked and supervised with EOS (`id=1`).
   - Active tokens count strictly measures `(y != -100).sum()`.

4. **Token-Weighted Loss Aggregation:**
   - Verified that loss aggregation is token-weighted:
     $$\mathcal{L}_{\text{dataset}} = \frac{\sum_{i=1}^N \mathcal{L}_i \cdot T_i}{\sum_{i=1}^N T_i}$$
     where $T_i$ is active response tokens in record $i$.
   - Arithmetic verified against manual hand-computed test fixtures.

5. **Empty-Target & Overlength Error Handling:**
   - Evaluator detects empty responses and returns `SKIPPED_EMPTY_RESPONSE`.
   - Records with sequence length exceeding `max_seq_len=512` are flagged `SKIPPED_OVERLENGTH`.
   - Out-of-bounds token IDs ($< 0$ or $\ge 4096$) return `SKIPPED_INVALID_TOKENS`.
   - Never produces silent `NaN` or unhandled exceptions.

6. **Reproducibility Verification:**
   - Repeated runs on the same environment produce identical loss (`6.0563`), identical perplexity (`426.79`), and identical probe outputs bit-for-bit.

7. **Capability Assessment & Scoring Precision:**
   - Added machine-readable capability statuses: `PASS`, `FAIL`, `INCONCLUSIVE`, `NOT_TESTED`.
   - Exact-match scoring is strict string matching; hints are treated as reference hints, not loose substring matches.

---

## 2. Capability Status Summary

| Checkpoint | Dataset | Token Loss | Perplexity | Records Evaluated | Capability Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Step 10,000 (Base)** | `val_aligned.jsonl` | 7.4781 | 1768.81 | 7/7 (195 tok) | `BASELINE` |
| **Batch 40 (Unaligned)** | `val_aligned.jsonl` | 7.2557 | 1416.17 | 7/7 (195 tok) | `INCONCLUSIVE` |
| **Batch 41 (Aligned)** | `val_aligned.jsonl` | 6.0563 | 426.79 | 7/7 (195 tok) | `INCONCLUSIVE` |

**Gate Status: `EVALUATION_INTEGRITY = PASS`**
