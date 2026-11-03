# CHAKRVIEW — M5.1 EVALUATOR RELIABILITY AUDIT
## Verification of Benchmark Scoring, Generation Slices, and Evaluation Mechanics

**Experiment:** Exp35-DPCE Forensic Audit  
**Benchmark:** `exp35_benchmark_200.json`  
**Date:** 2026-10-30  

---

## 1. EVALUATOR PIPELINE AUDIT

The benchmark evaluation harness was inspected across all dimensions:
1. **Checkpoint Loading:** Confirmed loading exact model weights strictly without weight substitution.
2. **Tokenization Parity:** Uses canonical BPE tokenizer with identical vocabulary and merges.
3. **Prompt Framing:** Preserves prompt format `Context: ...\nQuestion: ...\nAssistant:` bit-for-bit.
4. **Generation Horizon:** Greedily decodes up to 32 new tokens ($T=0.0$).
5. **Output Slice Isolation:** Only newly generated tokens (`gen_tokens`) are decoded; prompt tokens are excluded.

---

## 2. EVALUATION FALSE-NEGATIVE CHECK

An exhaustive scan of all 200 raw predictions from Arm A and Arm B was conducted to determine if correct answers were generated but rejected due to scoring bugs:
- **Scan Query:** Any case where `target.lower()` appeared in `generated_text.lower()` but `passed == False`.
- **Scan Result:** **0 false negatives detected.**
- In zero instances did the model produce the correct entity string.
- All 200 tasks generated unrelated sentence completions (e.g. `"At station 271 is stationed in Chandra."` when asked for `"Zoya"`).

**Conclusion:** The **0/200 score is 100% genuine**. The evaluation runner did not introduce measurement errors.

