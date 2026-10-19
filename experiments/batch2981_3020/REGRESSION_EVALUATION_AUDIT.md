# Regression Evaluation Audit — Batch 2981–3020
## Baseline vs. SFT Step-100 Evaluation Protocol & Provenance

### 1. Provenance of `BASELINE_REGRESSION_RESULTS.json`

In Batch 2941–2980, `BASELINE_REGRESSION_RESULTS.json` was created to summarize the comparative metrics between:
1. **Canonical Step-10,000 Baseline** (`experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`)
2. **SFT Pilot Candidate at Step 100** (`experiments/batch2821_2860/sft_pilot_run/sft_checkpoint_step_100.pt`)

#### Audit Finding on Data Origin:
Comparison against `experiments/batch2901_2940/CANDIDATE_EVALUATION_RESULTS.json` establishes that `BASELINE_REGRESSION_RESULTS.json` in Batch 2941–2980 was **extracted directly from the evaluation executed during Batch 2901–2940**.
- It was **not a freshly recomputed inference run** in Batch 2941–2980.
- Re-running expensive benchmark sweeps across 24 SFT test cases and 28 generation cases when model weights and test suites have not changed would be redundant.
- However, labeling this data as a newly executed evaluation without acknowledging its historical provenance was misleading. This audit explicitly confirms that **the numbers represent the identical evaluation run recorded in Batch 2901–2940**.

---

### 2. Evaluation Protocol & Environmental Parity

We verified that the evaluation protocol used in Batch 2901–2940 satisfies strict environmental parity:

| Evaluation Dimension | Baseline (Step 10,000) | Candidate (Step 100) | Parity Status |
| :--- | :--- | :--- | :--- |
| **Tokenizer** | `vocab_4096` (`release/chakrview-v0.5.0/data/experiments/vocab_4096`) | `vocab_4096` | **IDENTICAL** |
| **Model Config** | ChakrMicro v0.1 (`vocab=4096, d_model=192, layers=6, heads=6, ffn=512`) | ChakrMicro v0.1 | **IDENTICAL** |
| **Prompt Text** | Exactly identical 24 SFT prompts and 28 generation prompts | Identical prompts | **IDENTICAL** |
| **Decoding Config** | Greedy decoding (`temperature=0.0`, `max_new_tokens=32`, `top_k=0`) | Greedy decoding | **IDENTICAL** |
| **Stop Conditions** | `<EOS>` token id 1 or `max_new_tokens=32` | `<EOS>` token id 1 or `max_new_tokens=32` | **IDENTICAL** |
| **Execution Hardware**| Local CPU, float32 precision | Local CPU, float32 precision | **IDENTICAL** |

---

### 3. Metric Reconciliation

| Metric | Baseline (Step 10,000) | SFT Candidate (Step 100) | Regression / Improvement |
| :--- | :--- | :--- | :--- |
| **SFT QA Exact Match** | 0/24 (0.0%) | 0/24 (0.0%) | Unchanged (0 exact matches) |
| **Structured JSON Validity** | 0/3 (0.0%) | 2/3 (66.7%) | **+66.7% Improvement** |
| **Abstention on Insufficient Info** | 1/3 (33.3%) | 0/3 (0.0%) | -33.3% Regression |
| **Raw-Completion Empty / Immediate EOS Rate** | 0/28 (0.0%) | 15/28 (53.6%)* | **Severe Regression** |
| **Plain-Instruction Empty Rate** | 0/28 (0.0%) | 0/28 (0.0%) | Preserved (no empty outputs) |

*\*Note: In `BASELINE_REGRESSION_RESULTS.json`, `raw_completion_empty_count` was reported as 20 (71.43%) based on a broader whitespace-stripped emptiness criterion; the detailed token-level audit in `EOS_AND_LABEL_AUDIT.md` verified that exactly 15/28 cases generated immediate `<EOS>` as the first token.*

---

### 4. Conclusion & Recommendations

1. **Benchmark Distinction:** The 24-case evaluation suite (`experiments/batch2821_2860/SFT_EVALUATION_SET.json`) must be treated as a **historical regression baseline**, not an independent held-out benchmark for the proposed balanced SFT experiment, because 21 of its examples were absorbed into the 320-example training set of Batch 2941–2980.
2. **Future Candidate Evaluation:** The newly created 40-case held-out test split (`experiments/batch2941_2980/test.jsonl`) provides a genuinely uncontaminated test suite for evaluating future training runs.
