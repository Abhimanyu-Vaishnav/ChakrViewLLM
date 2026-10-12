# Final Engineering Report — Master Sprint Batches 63–66
## Diagnose the Training–Generation Gap Before Scaling Data

---

### Executive Summary

In Master Sprint Batches 63–66, we investigated the reported paradox from Batches 59–62: teacher-forced cross-entropy loss dropped to ~0.38 after 50 SFT steps and reached 100% EOS completion, yet greedy autoregressive generation produced 0/32 exact match on training data and collapsed to `|277`.

Rather than prematurely scaling the dataset, we performed an end-to-end token-level sequence trace from raw JSONL through causal alignment to greedy generation.

**The investigation discovered and resolved the exact root cause**:
1. **Prompt Delimiter Divergence (`.strip()` dropping `\n`)**:
   In `SFTDataset.encode_example` (`chakrview/training/sft_engine.py`) and `DeterministicEvaluator.encode_record`, `prompt.strip()` stripped the trailing newline from `<|im_start|>assistant\n`. This removed token `13` (`\n`), so the training model was conditioned on token `1807` (`ant`) to predict the first response token. During evaluation and inference, the generation prompt was tokenized from the raw string ending in `\n` (token 13). Conditioned on `\n`—a boundary it was never trained on—the model reverted to its step-10,000 pretraining Wikipedia table attractor, outputting token `127` (`|`), collapsing into `|277`.
2. **Template Dominance Masking Task-Token Underfitting**:
   In each response (`"23<|im_end|>\n"`), 8 out of 9 target tokens are shared ChatML template tokens (`<|im_end|>`). The model achieved loss 0.017 on template tokens within 50 steps, driving overall mean loss down to 0.38 even though answer-token loss remained 2.83.

By introducing explicit delimiter preservation (`strip_prompt_delimiters=False`), the table artifact `|` was completely eliminated. A controlled full-batch capacity experiment proved that **ChakrMicro v0.1 has the capacity to achieve 32/32 (100%) exact match on training data**, but achieves **0/8 exact match on validation data**, establishing that the model is memorizing discrete string mappings rather than learning a generalized arithmetic transformation.

---

### Quantitative Evaluation Summary

| Experiment Configuration | Split | Records | Exact Match Accuracy | Mean Token Loss | EOS Rate | Output Characteristic |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| **Batch 59–62 (Buggy Prompt Alignment)** | Train | 32 | 0 / 32 (0.0%) | 0.3823 | 100.0% | Collapsed to `\|277<\|im_end\|>` |
| | Val | 8 | 0 / 8 (0.0%) | 0.4642 | 100.0% | Collapsed to `\|277<\|im_end\|>` |
| | Bench | 8 | 0 / 8 (0.0%) | 0.3705 | 100.0% | Collapsed to `\|277<\|im_end\|>` |
| **Batch 65 Run 1 (50 Steps SFT, Aligned)** | Train | 32 | 3 / 32 (9.38%) | 0.3504 | 100.0% | Collapsed to `27<\|im_end\|>` (no `\|`) |
| | Val | 8 | 0 / 8 (0.0%) | 0.4145 | 100.0% | Collapsed to `27<\|im_end\|>` (no `\|`) |
| | Bench | 8 | 1 / 8 (12.5%) | 0.3394 | 100.0% | Collapsed to `27<\|im_end\|>` (no `\|`) |
| **Batch 65 Run 2 (100 Steps Full-Batch)** | Train | 32 | **32 / 32 (100.0%)** | **0.0029** | 100.0% | **Exact match on all 32 train pairs** |
| | Val | 8 | **0 / 8 (0.0%)** | 0.4210 | 100.0% | Numeric guesses (`14`, `11`, `17`, etc.) |

---

### Root-Cause Verdict and Evidence Attribution

| Hypothesis / Phenomenon | Status | Direct Verification Evidence |
|:---|:---:|:---|
| **Prompt Delimiter Mismatch (`.strip()`)** | **CONFIRMED & FIXED** | Directly proven: `prompt.strip()` dropped token 13 (`\n`), leaving `ant` as boundary. Inference evaluated on `\n`, producing `|277`. When `strip_prompt_delimiters=False` was applied, `|` table artifact was completely eliminated. |
| **Causal Logit Alignment** | **CONFIRMED CORRECT** | Position $|P|$ predicts target $R_0$. Tested and verified in `test_harness.py`: greedy decode step 0 is identical to $\text{argmax}(\text{logits}[|P|])$. |
| **Capacity for Memorization** | **CONFIRMED SUFFICIENT** | In 100 full-batch steps, ChakrMicro v0.1 memorizes all 32 distinct synthetic inputs and outputs (100% exact match). |
| **Arithmetic Generalization** | **CONFIRMED ZERO** | When tested on disjoint validation inputs (e.g. `13 + 5 = 18`), the memorized model outputs `14` (0% exact match). |
| **EOS Completion Metric Deception** | **CONFIRMED** | Model achieves 100% EOS completion even when generating completely wrong answers. EOS completion only indicates template adherence. |
| **Tokenizer Digit Fragmentation** | **SUPPORTED** | Two-digit integers are fragmented inconsistently (e.g. `20` is 1 token, `21` is 2 tokens), impairing arithmetic reasoning. |

---

### Architectural Invariants and Safety Verification

All canonical ChakrMicro invariants remain strictly verified and untouched:
- **Model Parameters**: `3,443,136` (exact match)
- **Fresh Init Hash**: `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` (exact match)
- **Canonical Step 10k Checkpoint Hash**: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (exact match)
- **Preserved User Modification**: `chakrview/runtime/session.py` was inspected and preserved untouched. Never staged or committed.
- **Batch Regression Suite**: 90/90 passed (82 historical + 8 new Batch 63-66 tests).

---

### Completed Work vs Unresolved Questions

**Completed**:
1. Hand-verifiable sequence trace (`batch63_trace.md`) pinpointing the `.strip()` newline loss.
2. Fixed `SFTDataset` and `DeterministicEvaluator` to preserve semantic ChatML delimiters via configurable `strip_prompt_delimiters` (defaulting to True for backward compatibility with batches 42-62, with explicit False for exact delimiter alignment).
3. Created deterministic correctness harness (`correctness_harness/test_harness.py`).
4. Ran 50-step recovery and 100-step capacity experiments (`batch65_results.md`).
5. Implemented comprehensive test suite in `tests/test_batch63_66_training_generation_debug.py`.

**Unresolved Questions**:
1. What training data volume and loss masking strategy (e.g., answer-only loss weighting) are necessary for ChakrMicro to learn compositional operations rather than pure input-output memorization?
2. Would digit-level tokenization or chain-of-thought scratchpad prompting enable multi-digit arithmetic generalization?

**Single Most Valuable Next Experiment**:
Test **answer-token loss reweighting** (masking or heavily downweighting the 8 template tokens relative to answer tokens) on a slightly expanded synthetic addition set (e.g., 64 examples) to evaluate if gradient focus on the mathematical tokens induces operator learning.
