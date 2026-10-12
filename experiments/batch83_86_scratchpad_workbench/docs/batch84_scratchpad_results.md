# Batch 84 — Scratchpad Learning vs Direct-Answer Evaluation Results

## 1. Experimental Overview

Batch 84 compared two matched conditions trained for 100 steps from the immutable canonical step-10k checkpoint on 99,417 tokens:
- **Baseline**: Step 10,000 Unmodified (`96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`).
- **Condition A (Direct-Answer Arithmetic)**: Trained on `train_direct_tokens.bin`. Checkpoint SHA-256: `cc554926216a40cd51d438874352595ae940e7f2ee105a5239ed6bea01cc2766`.
- **Condition B (Scratchpad Arithmetic)**: Trained on `train_scratchpad_tokens.bin`. Checkpoint SHA-256: `f55b87aff8b2c4d7e46f2336fc39e4929c171497bde0c6508c48a823fd8e318d`.

---

## 2. Quantitative Results Across All Evaluated Splits

| Condition | Format Evaluated | Train Exact Match | Val Exact Match | Test Exact Match | Valid Format Rate | EOS Rate | Mean Loss | Predominant Answer |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Baseline (Step 10k)** | Direct | 0 / 64 (0.0%) | 0 / 18 (0.0%) | 0 / 18 (0.0%) | 0.0% | 0.0% | 6.7330 | Wikitable loops |
| **Baseline (Step 10k)** | Scratchpad | 0 / 64 (0.0%) | 0 / 18 (0.0%) | 0 / 18 (0.0%) | 0.0% | 0.0% | 5.6089 | Wikitable loops |
| **Condition A (Direct)** | Direct | 6 / 64 (9.4%) | 0 / 18 (0.0%) | 3 / 18 (16.7%)* | 100.0% | 100.0% | 0.2567 | **`8` (100% modal)** |
| **Condition A (Direct)** | Scratchpad | 6 / 64 (9.4%) | 0 / 18 (0.0%) | 3 / 18 (16.7%)* | 100.0% | 100.0% | 5.6367 | `8` (100% modal) |
| **Condition B (Scratchpad)**| Direct | 3 / 64 (4.7%) | 1 / 18 (5.6%) | 0 / 18 (0.0%) | 100.0% | 100.0% | 1.7985 | `4` (83%), `15` (17%) |
| **Condition B (Scratchpad)**| Scratchpad | 3 / 64 (4.7%) | 2 / 18 (11.1%) | **0 / 18 (0.0%)** | **100.0%** | **100.0%** | **0.3081** | `4` (72%), `15` (28%) |

*\*Note on Condition A*: Condition A collapsed completely to the constant digit `8`. The 3/18 test match was an artifact of test pairs whose true sum was 8 (`0+8`, `8+0`, `6+2`).

---

## 3. Mechanistic Diagnosis of Condition B (Scratchpad)

1. **Flawless Template & Counting Syntax Mastery**:
   - Condition B achieved **100.0% valid format rate** and **100.0% EOS rate**.
   - Generated outputs followed the exact requested scratchpad syntax without grammatical errors:
     * `Start at 0; count 4 more: 1, 2, 3, 4; answer: 4`
     * `Start at 9; count 4 more: 10, 11, 12, 13, 14, 15; answer: 15`
2. **Scratchpad Memorization Without Input Conditioning**:
   - When presented with arbitrary prompts (e.g. `Add with scratchpad: 1 + 8 = ?`), the model generated canned scratchpad sequences from its training memory rather than dynamically starting from $a$ and counting $b$ steps.
   - For small operands, it emitted `Start at 0; count 4 more: 1, 2, 3, 4; answer: 4` (72% of cases).
   - For larger operands, it emitted `Start at 9; count 4 more: 10, 11, 12, 13, 14, 15; answer: 15` (28% of cases).
3. **Held-Out Test Generalization Verdict**:
   - Exact match on genuinely held-out test operand pairs was **0 / 18 (0.0%)**.
   - **Conclusion**: Providing intermediate autoregressive sequence positions alone is insufficient if the training objective simply allows the model to memorize canned scratchpad strings. Causal next-token prediction treats the intermediate steps as an extended template rather than a dynamic computational algorithm.
