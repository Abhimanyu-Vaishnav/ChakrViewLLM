# Stage 4 & 5 — Pilot Reproduction and English/Arithmetic Discrepancy Audit Report

**Date:** 2026-11-18  
**Audit Mission:** UCIP-09S-R1 Independent Evidence Verification & Evaluator Audit  
**Artifacts Examined:**
- `experiments/ucip09s/scripts/train_ucip09s_pilot.py`
- `experiments/ucip09s/scripts/eval_pilot_reproduction.py`
- `experiments/ucip09s/checkpoints/ucip09s_balanced_pilot.pt` (SHA-256: `c99dcc0a229310ec682857349c06e1a46957b559a8d180a33a91ea05b225d830`)
- `experiments/ucip09s/evaluations/pilot_reproduction_results.json`

---

## 1. Pilot Reproduction & Evaluation Pipeline Verification

### 1.1 SHA-256 Checkpoint Verification
- **Starting Checkpoint:** `releases/ucip_09_candidates/chakrview_ucip09_medium_model.pt`  
  SHA-256: `5513fefe878ca85a637aa8401b3bebf5b8e3ca93ab362e07722d629e34294499` (MATCH)
- **Saved Pilot Checkpoint:** `experiments/ucip09s/checkpoints/ucip09s_balanced_pilot.pt`  
  SHA-256: `c99dcc0a229310ec682857349c06e1a46957b559a8d180a33a91ea05b225d830` (MATCH)

### 1.2 Decoding & Evaluation Parity Check
We verified whether the pilot evaluation in `eval_pilot_reproduction.py` used the exact same prompt wrapper and decoding settings as the baseline runner in `run_owner_reproduction.py`:
- **Prompt Wrapper:** `f"User: {p['input']}\nAssistant:"` (Identical across all 3 checkpoints).
- **Tokenizer:** `data/experiments/vocab_4096` (Identical vocab and merges).
- **Max New Tokens:** 48 (Identical).
- **Repetition Penalty:** 1.05 (Identical).
- **No Repeat N-Gram:** 3 (Identical).
- **Greedy Argmax:** `int(torch.argmax(logits).item())` (Identical).

**Finding:** The evaluation environment, prompt format, and inference parameters were **100% identical and directly comparable across all three models**.

---

## 2. Investigation of English Scoring Discrepancy (Requirement 5)

The previous report claimed **4/4 (100.0%)** for Condition C in English conversation, yet qualitative notes highlighted:
1. `REPRO_ENG_03` ("What does a CPU do?"):  
   `"A CPU operates through the fetch-decode-"` *(Truncated mid-sentence!)*
2. `REPRO_ENG_04` ("Explain RAM vs storage"):  
   `"RAM is fast, volatile primary memory used to hold active data for the CPU, whereas storage that holatier holy."` *(Incoherent trailing clause!)*
3. `REPRO_ENG_02` ("Who created you?"):  
   `"ChakrView ChakrView, an independent neural conversational assistant designed for efficient local execution."` *(Duplicated name prefix!)*

### Origin of the Discrepancy:
All three examples belong directly to the scored 20 probes (`REPRO_ENG_01`–`04`).
They passed in UCIP-09S solely because the evaluator script had **no semantic rubric** for English conversation and defaulted to:
```python
passed = len(cleaned.split()) >= 3 and not ("?" in cleaned and len(cleaned.split()) < 5)
```
Since `"A CPU operates through the fetch-decode-"` has 6 words and no question mark, the naive script awarded a `PASS`. Under strict semantic criteria:
- `REPRO_ENG_01` (Greeting): **PASS** (Fluent, coherent identity).
- `REPRO_ENG_02` (Creator): **PASS** (Mentions ChakrView neural conversational assistant design; acceptable given small capacity).
- `REPRO_ENG_03` (CPU): **FAIL** (Incomplete truncation at token limit).
- `REPRO_ENG_04` (RAM vs Storage): **FAIL** (Semantic breakdown in storage explanation).

**Strict Score for English Conversation:** **2 / 4 (50.0%)**, NOT 4/4 (100.0%).

---

## 3. Investigation of Arithmetic Scoring Discrepancy (Requirement 6)

The previous report claimed UCIP-09 arithmetic was **2/5 (40.0%)**.

### Probe-by-Probe Truth Table:

| Probe | Problem | Expected | Model Output | Evaluator Passed? | Actual Reality |
|---|---|:---:|---|:---:|:---|
| `REPRO_MATH_01` | $2 + 2$ | 4 | `"2 + 1 = 3."` | FAIL | Correctly failed (Asserts 3) |
| `REPRO_MATH_02` | $5 - 2$ | 3 | `"5 - 3 = 4."` | **PASS (False Positive)** | **WRONG:** Evaluator found operand `3` in `[5, 3, 4]`. The model asserted $5 - 3 = 4$! |
| `REPRO_MATH_03` | 5 apples - 2 | 3 | `"You will have 3 apple remaining (37 application dowild."` | PASS | **CORRECT:** Model correctly computed $5 - 2 = 3$ |
| `REPRO_MATH_04` | 15 books - 7 | 8 | `"You pening, and I have 5 apples remaining (3 = 36)."` | FAIL | Correctly failed |
| `REPRO_MATH_05` | $10 + 5$ | 15 | `"10 + 5 = 3."` | FAIL | Correctly failed (Asserts 3) |

**Actual Genuine Arithmetic Accuracy:**
- **UCIP-09 Baseline:** **1 / 5 (20.0%)** (Only `REPRO_MATH_03` passed).
- **UCIP-09R Degraded:** **0 / 5 (0.0%)** (Collapsed into `21` and `22`).
- **UCIP-09S Condition C Pilot:** **1 / 5 (20.0%)** (Answered `REPRO_MATH_03` as 3; no `21`/`22` collapse).
