# NR-019: Argument Extraction Forensics & Failure Taxonomy

## 1. Overview
In NR-018, System C (Neural Intent Router Direct) failed on **336 out of 500 tasks** (67.2% overall failure rate). A forensic classification of every individual failure was conducted across the 12 required error categories.

---

## 2. Quantitative Failure Taxonomy (336 Failed Cases)

| Category # | Error Classification | Cases | % of Failures | Primary Origin | Representative Example |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **01** | Wrong operation | 0 | 0.0% | Neural Model | In all parsed cases, operation name was matched correctly. |
| **02** | Missing argument | 0 | 0.0% | Neural Model | Schema keys (`a`, `b`, `list`, `start`) were present. |
| **03** | Wrong numeric / entity value | **292** | **86.9%** | **Neural Model** | Prompt: `"Multiply 15 by 9."` $\to$ Extracted: `{"a": 12, "b": 9}`. Prompt: `"[21, 22, 16, 12] list ko reverse..."` $\to$ Extracted: `{"list": [28, 20, 18]}`. |
| **04** | Transposed arguments | Included in #03 | - | Neural Model | Prompt: `"Subtract 11 from 33."` $\to$ Extracted `{"a": 32, "b": 21}` (hallucinated values rather than pure transposition). |
| **05** | Wrong variable binding | Included in #03 | - | Neural Model | Variable assignment tasks scrambled stored variable names. |
| **06** | Incorrect operation order | 0 | 0.0% | Neural Model | Sequential steps preserved order (`['add', ...], ['sub', ...]`). |
| **07** | Number-word interpretation failure | Included in #03 | - | Neural Model | Prompt: `"पंद्रह को बारह से गुना करो"` $\to$ hallucinated ASCII digits or failed. |
| **08** | Unicode / Numeral normalization failure | Included in #03 | - | Normalization Layer | Devanagari numerals (`१५`) fragmented into raw bytes and distorted. |
| **09** | Unsupported/ambiguous incorrectly accepted/rejected | 0 | 0.0% | Neural Model | Unsupported injection attacks (`os.system`) were correctly rejected. |
| **10** | JSON formatting / truncation failure | **43** | **12.8%** | **Neural Model** | Long conditional branch JSON payloads exceeded generation window (truncated at `"stat...`). |
| **11** | Execution failure (malformed types) | **1** | **0.3%** | **Validation Layer** | Prompt: `"10 को 11 से गुणा करें।"` $\to$ Generated `{"a": 18, "b": [10, 10, 10]}` (passed array to scalar multiplication). |
| **12** | Final response mismatch | 0 | 0.0% | Runtime Engine | When correct JSON was executed, runtime engine was 100% accurate. |

---

## 3. Core Forensic Insights
1. **The Root Cause is Generative Hallucination, NOT Semantic Misunderstanding**:
   - The 17M foundation model has robust semantic comprehension: it almost never confused multiplication with addition or string reversal with sorting.
   - However, when asked to **generate numbers free-form**, the autoregressive decoder samples from its language pretraining frequency distribution rather than copying tokens from the prompt.
2. **The Solution: Exact Source-Span Grounding**:
   - The neural model must NEVER generate argument values freely.
   - Instead, arguments must be grounded directly in the user's input text via **exact character span pointers** `(start_idx, end_idx)`.
   - A deterministic parser/normalizer then extracts the exact substring from the source text and parses it into typed numeric values.
