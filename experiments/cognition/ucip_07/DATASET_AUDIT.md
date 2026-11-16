# UCIP-07: Dataset Representation & Tokenizer Alignment Audit

## 1. Corpus Structure & Inventory

The UCIP-07 training corpus consolidated data across three primary repositories in `data/experiments/`:
1. `cbr_stage_c_canonical.jsonl` (Canonical conversational pairs)
2. `cbr_language_pairs.jsonl` (Bilingual English-Hindi parallel texts)
3. `cbr_reasoning_arithmetic.jsonl` (Step-by-step arithmetic and logic traces)

Total curated examples: 2,450 conversations / instruction pairs.

---

## 2. Category Distribution Audit

| Category Domain | Pair Count | Percentage | Token Count (approx.) |
|---|---|---|---|
| English Conversational & QA | 650 | 26.5% | 42,000 |
| Devanagari Hindi Dialogue | 480 | 19.6% | 38,000 |
| Roman Hindi & Hinglish | 520 | 21.2% | 36,000 |
| Concept Explanations | 350 | 14.3% | 29,000 |
| Elementary Arithmetic & Logic | 250 | 10.2% | 15,000 |
| Instruction Following & Exact Copy | 200 | 8.2% | 12,000 |
| **Total** | **2,450** | **100.0%** | **~172,000** |

---

## 3. Tokenizer Coverage & Hindi Fragmentation Analysis

### 3.1 Vocabulary Parameters
- **Vocabulary Size**: 4,096 tokens (BPE)
- **Token Artifact Path**: `data/experiments/vocab_4096/`
- **Special Tokens**: `<unk>` (0), `<pad>` (1), `<s>` (2), `</s>` (3)

### 3.2 Token Fragmentation Metrics

A critical source of generation degradation in Hindi and arithmetic is subword token fragmentation:

| Text Type | Sample String | Char Count | Token Count | Compression Ratio (Chars/Token) |
|---|---|---|---|---|
| **English Prose** | `"The CPU is the central processing unit."` | 39 | 9 | **4.33** |
| **Roman Hindi** | `"Bhai machine learning ko simple mein samjhao"` | 45 | 11 | **4.09** |
| **Devanagari Hindi** | `"नमस्ते! आप क्या कर सकते हैं?"` | 29 | 16 | **1.81** |
| **Exact Copying** | `"The sky is blue."` | 17 | 6 | **2.83** |
| **Arithmetic** | `"5 + 3 = 8"` | 9 | 7 | **1.28** |

### 3.3 Key Findings:
1. **Devanagari Expansion**: Due to the small vocabulary (4,096), Devanagari characters require 2 to 3 subword tokens per conjunct. This doubles the effective sequence length for Hindi, forcing the attention mechanism to maintain coherence across twice as many autoregressive rollout steps.
2. **Arithmetic Digit Splitting**: Numbers are not tokenized as atomic digits; multi-digit numbers often split into erratic prefix fragments, impairing arithmetic reasoning.
3. **Exact Copying Failure**: In exact copy prompts (`"Say exactly: The sky is blue"`), models repeatedly fail (0% pass rate) because the training corpus heavily biases the model towards conversational completions rather than pure identity mappings.
