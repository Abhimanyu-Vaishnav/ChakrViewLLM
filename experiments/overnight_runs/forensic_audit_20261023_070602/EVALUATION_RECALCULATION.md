# ChakrView Post-Overnight Forensic Audit — Evaluation Recalculation

**Auditor**: Independent ML Experiment Auditor  
**Audit Target Files**:
- `BASELINE_TEST_EVAL.json` (SHA-256: `1b523171...`)
- `SFT_TEST_EVAL.json` (SHA-256: `d6cde2e9...`)
- `GATE_V01_COMPARATIVE_EVAL.json` (SHA-256: `6a6a0956...`)
- `TRAINING_SUMMARY.json` (SHA-256: `8551f41e...`)

---

## 1. Metric Recalculation from Raw Data

We independently parsed and recalculated the evaluation metrics directly from the raw record objects stored in the JSON output files.

### 1.1 Track 1: In-Distribution Held-Out Test Split (40 Records)

| Metric | Raw Baseline Calculation | Claimed Baseline | Raw SFT Calculation | Claimed SFT | Verification Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Total Test Prompts** | 40 | 40 | 40 | 40 | **VERIFIED** |
| **Clean Turn Termination (EOS)** | 0 / 40 (0.0%) | 0.0% | 18 / 40 (45.0%) | 45.0% | **VERIFIED** |
| **Exceeded Max Tokens Ceiling** | 40 / 40 (100.0%) | 100.0% | 22 / 40 (55.0%) | 55.0% | **VERIFIED** |
| **Best Val Loss** | 4.8618 | 4.8618 | 1.9602 (Step 225) | 1.9602 | **VERIFIED** |
| **Val Perplexity** | 129.25 | 129.25 | 7.10 | 7.10 | **VERIFIED** |

#### Category Breakdown of Clean Turn Termination (`EOS%`):
- `structured_formatting`: 5 / 5 (100.0%)
- `factual_qa`: 4 / 5 (80.0%)
- `basic_arithmetic`: 3 / 5 (60.0%)
- `short_explanation`: 3 / 5 (60.0%)
- `reference_resolution`: 2 / 5 (40.0%)
- `uncertainty_abstention`: 1 / 5 (20.0%)
- `instruction_following`: 0 / 5 (0.0%)
- `context_extraction`: 0 / 5 (0.0%)

### 1.2 Track 2: Normal Chat Gate v0.1 Frozen Suite (20 Prompts)

| Metric | Raw Baseline Calculation | Claimed Baseline | Raw SFT Calculation | Claimed SFT | Verification Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Total Gate Prompts** | 20 | 20 | 20 | 20 | **VERIFIED** |
| **Clean Turn Termination (EOS)** | 1 / 20 (5.0%) | 5.0% | 12 / 20 (60.0%) | 60.0% | **VERIFIED** |
| **Exceeded Max Tokens Ceiling** | 19 / 20 (95.0%) | 95.0% | 8 / 20 (40.0%) | 40.0% | **VERIFIED** |

---

## 2. Qualitative & Semantic Failure Analysis

While the quantitative claim of **EOS improvement is 100% verified**, a strict forensic inspection of the generated text reveals that **EOS termination does NOT equate to semantic task success**:

### 2.1 Structured Formatting (Claim: 100% EOS Termination)
- In `EXP17_TEST_STRUCTURED_FORMATTING_000` and `001`, the SFT model emitted an immediate `EOS` token without generating any characters (`generated_text = ""`).
- In `EXP17_TEST_STRUCTURED_FORMATTING_002` through `004`, the model emitted code-like docstring fragments: `'\n        """\n        if not isinstance("metric_train_3")'` followed by `EOS`.
- **Verdict**: None of the generated outputs are valid JSON objects conforming to the requested schema. The model learned that the response turn should be short and terminated, but it failed to master valid JSON emission under greedy decoding.

### 2.2 Factual QA & Arithmetic (Claim: 80% & 60% EOS Termination)
- In `EXP17_TEST_FACTUAL_QA_000` (Prompt: *"Which Australian city serves as the federal capital of the country?"*), the SFT model produced: `"\nStep-by-Step Solution:\nThe total of 240."` `[STOP_TOKEN]`.
- In `EXP17_TEST_BASIC_ARITHMETIC_000` (Prompt: *"What is the sum of 503 and 505?"*), the model produced: `"\nThe sum of 2005 is 406."` `[STOP_TOKEN]`.
- **Verdict**: The model learned the template boundary and stopped cleanly, but the arithmetic value (406 vs 1008) and factual entity (240 vs Canberra) are incorrect.

### 2.3 Instruction Following & Context Extraction (0% EOS Termination)
- For transformations (`ALPHA_TEST_0`, `beta_test_1`, reverse strings) and long context passages, the model failed to emit EOS and continued until reaching `max_new_tokens=48`.
- The outputs degenerated into Python assignment loops (`self._2 = self._3`).

---

## 3. Methodological Rigor & Split Separation

- **Prompt Alignment**: All 40 test prompts matched the inference format exactly (0 prompt mismatches).
- **Decoding Configuration**: Both baseline and SFT evaluations used identical parameters (`greedy`, `temperature=0.0`, `max_new_tokens=48`, `device="cpu"`).
- **Leakage Check**: The 40 test examples were strictly held out during training. Training was performed only on `train_exp17_v1.jsonl` (320 items) and validated on `val_exp17_v1.jsonl` (40 items). Test examples never entered the gradient updates or checkpoint selection.

---

## 4. Conclusion on Evaluation Claims

1. **What is Scientifically Confirmed**:
   - The numerical metrics reported in `EVALUATION_REPORT.md` (EOS rate 45.0% on Test, 60.0% on Gate v0.1; Perplexity 7.10; Best Val Loss 1.9602) are **completely verified and mathematically exact**.
2. **What Must Be Explicitly Clarified**:
   - The improvement is primarily **structural and grammatical (turn-taking and termination learning)**.
   - It is **not** evidence of general instruction-following intelligence, factual accuracy, or JSON schema generation capability.
