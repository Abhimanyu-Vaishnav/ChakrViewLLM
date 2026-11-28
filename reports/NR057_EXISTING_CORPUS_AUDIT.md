# NR-057 EXISTING CORPUS FORENSIC AUDIT REPORT

**Date:** November 28, 2026  
**Mission:** NR-057 Clean Corpus Expansion & Training Readiness Engineering  
**Artifact ID:** `NR057_EXISTING_CORPUS_AUDIT`  
**JSON Inventory:** [existing_corpus_inventory.json](file:///d:/Project/ChakrView/experiments/corpus_nr057/existing_corpus_inventory.json)

---

## 1. Executive Summary

This forensic audit inspected all historical training and validation datasets in the ChakrView repository across Corpus V1, V2, V3, V4, and NR-051 Pretraining splits. Every file was audited using canonical Byte-Level BPE tokenizer (`tokenizer/bpe_v2_4096.json`) tokenization, exact SHA-256 text fingerprinting, and NFC/NFKC text normalization.

### Key Forensic Findings:
1. **Total Available Historical Pretraining Volume:** The absolute maximum unique token count in any individual training set was **592,824 tokens** in Corpus V4 Train (and 1,026,382 total tokens in NR-051 Pretrain Train, which had 15.82% exact duplicates).
2. **Corpus V4 Composition:** Corpus V4 Train consists of 5,828 unique documents, 0% exact internal duplicates, totaling 592,824 tokens. Corpus V4 Val consists of 306 unique documents totaling 31,244 tokens. The combined clean V4 pool is **624,068 tokens**.
3. **Severe Under-Exposure Proven:** Standard pretraining regimes for models of 17M to 38M parameters require between 340M and 760M tokens (Chinchilla compute-optimal: 20 tokens per parameter). The historical ChakrView training exposure of ~624K tokens represents less than **0.2%** of optimal exposure for 17M, and less than **0.08%** for 38M.
4. **Duplication in Historical Sets:**
   - Corpus V3 Train had 315 exact duplicate documents (7.73% duplication rate).
   - NR-051 Pretrain Train had 1,337 duplicate documents (15.82% duplication rate).
   - Corpus V1, V2, and V4 had 0.0% internal exact duplicates within their respective splits.

---

## 2. Quantitative Inventory

| Dataset Split | File Path | Raw Docs | Unique Docs | Dup Count | Dup Rate (%) | Raw Bytes | Normalized Bytes | Total Tokens | Avg Tok/Doc | Median Tok/Doc |
|---|---|---|---|---|---|---|---|---|---|---|
| **Corpus V1 Train** | `data/corpus_v1/master_corpus_train.jsonl` | 5,826 | 5,826 | 0 | 0.00% | 2,345,706 | 1,389,242 | 483,937 | 83.1 | 72 |
| **Corpus V1 Val** | `data/corpus_v1/master_corpus_val.jsonl` | 539 | 539 | 0 | 0.00% | 196,022 | 108,201 | 38,274 | 71.0 | 64 |
| **Corpus V2 Train** | `data/corpus_v2/corpus_v2_train.jsonl` | 5,089 | 5,089 | 0 | 0.00% | 2,952,401 | 1,128,135 | 370,218 | 72.7 | 61 |
| **Corpus V2 Val** | `data/corpus_v2/corpus_v2_val.jsonl` | 435 | 435 | 0 | 0.00% | 228,858 | 75,084 | 24,028 | 55.2 | 54 |
| **Corpus V3 Train** | `data/corpus_v3/corpus_v3_train.jsonl` | 4,075 | 3,760 | 315 | 7.73% | 2,478,196 | 1,011,743 | 326,312 | 80.1 | 69 |
| **Corpus V3 Val** | `data/corpus_v3/corpus_v3_val.jsonl` | 435 | 435 | 0 | 0.00% | 228,858 | 75,084 | 24,028 | 55.2 | 54 |
| **Corpus V4 Train** | `data/corpus_v4/corpus_v4_train.jsonl` | 5,828 | 5,828 | 0 | 0.00% | 2,823,514 | 1,829,251 | 592,824 | 101.7 | 119 |
| **Corpus V4 Val** | `data/corpus_v4/corpus_v4_val.jsonl` | 306 | 306 | 0 | 0.00% | 153,426 | 101,383 | 31,244 | 102.1 | 115 |
| **NR-051 Pretrain Train** | `data/pretraining_nr051/pretrain_train.jsonl` | 8,450 | 7,113 | 1,337 | 15.82% | 4,225,741 | 3,189,251 | 1,026,382 | 121.5 | 61 |
| **NR-051 Pretrain Val** | `data/pretraining_nr051/pretrain_val.jsonl` | 444 | 444 | 0 | 0.00% | 142,409 | 86,984 | 27,955 | 63.0 | 56 |

---

## 3. Forensic Analysis of Data Deficiencies

### 3.1 Severe Short-Context Truncation
Across all historical corpora, the median document length is only **54 to 119 tokens**. 
For a causal language model with context window of 512 tokens:
- Over 90% of training samples occupied less than 25% of the attention context window.
- The model was predominantly trained on isolated sentences, short Q&A pairs, or conversational snippets without discourse structure, paragraphs, or narrative transitions.

### 3.2 Repetitive Over-Sampling vs Genuine Knowledge Acquisition
In NR-051, attempting to achieve 1M tokens resulted in 1,337 duplicated documents (15.82% duplicate rate). In multiple historical training runs, models were trained for 3 to 10 epochs over the same 500k-token pool. This produced rapid loss decay and synthetic overfitting without exposing the network to genuine lexical variety or broad syntactic patterns.

### 3.3 Domain Imbalance
Historical datasets were heavily skewed toward synthetic conversational templates and instruction pairs rather than rich, expository, structured pretraining text (encyclopedic, literary, technical, mathematical, or long-form prose).

---

## 4. Architectural Scaling Implications for Corpus V5

According to established scaling laws (Kaplan et al., Chinchilla / Hoffmann et al.):
- **ChakrGenesis17M** (17,063,808 params): Compute-optimal training tokens = $17.06\text{M} \times 20 \approx 341\text{M}$ tokens. Minimum viable threshold = $50\text{M}$ tokens (~3 tokens/param).
- **ChakrGenesis38M** (37,499,392 params): Compute-optimal training tokens = $37.5\text{M} \times 20 \approx 750\text{M}$ tokens. Minimum viable threshold = $75\text{M}$ tokens.

### Target for NR-057 Corpus Expansion:
Building a clean, verified, deduplicated pretraining corpus (`Corpus V5`) spanning:
- English clean public domain literature / expository texts (Gutenberg, Wikipedia clean extracts).
- Python and technical algorithmic code / documentation.
- Math and science expository concepts.
- Devanagari Hindi and Hinglish clean open texts.
- Verified absence of test prompts from NR-053 fresh benchmark, NR-055 Benchmark V2, and NR-056 Benchmark V3.
