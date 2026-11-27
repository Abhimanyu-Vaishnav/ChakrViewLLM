# NR-051: Pretraining Corpus and Token Budget Audit

**Campaign:** CHAKRVIEW NR-051 — True Language Pretraining & Controlled Scaling  
**Date:** November 27, 2026  
**Artifact Directory:** `data/pretraining_nr051/`  
**Manifest:** [pretrain_manifest.json](file:///d:/Project/ChakrView/data/pretraining_nr051/pretrain_manifest.json)  

---

## 1. Executive Summary

Prior ChakrView missions (NR-046 through NR-050) evaluated conversational behavior predominantly via direct fine-tuning (SFT) from either random initialization or legacy checkpoints. In NR-050, scaling from 17M to 38M directly on SFT data resulted in training instability and poor convergence because the 38M parameter architecture lacked an underlying language distribution prior.

NR-051 established a rigorously controlled, pretraining-from-scratch foundation. A verified multi-lingual, multi-domain pretraining corpus was assembled and audited for contamination against both the sealed 120-prompt multi-domain benchmark and historical evaluation suites.

---

## 2. Pretraining Corpus Composition & Volume

The pretraining dataset was constructed by pulling clean, continuous natural language documents across verified internal ChakrView repositories, Wikipedia foundational extracts, technical programming concepts, and linguistic foundations.

- **Total Documents Audited & Processed:** 8,894 unique documents
- **Deduplication Method:** Exact SHA-256 and whitespace-normalized MD5 hashing across text bodies.
- **Leakage / Contamination Audit:** All 8,894 documents were filtered against the 120 sealed benchmark prompts, 1,000 matched benchmark tasks, and evaluation prompts. **Zero contamination** (0 exact prompt matches or leaked answer strings) was confirmed.

### Split Breakdown

| Split | Document Count | Token Count | Character Count | SHA-256 Checksum |
| :--- | :--- | :--- | :--- | :--- |
| **Train Split** | 8,450 | **1,026,382** | 4,500,432 | `871c12b080e34c1d63ab5a666c5b3e67c65fcfd870c46911376ba6165575aaec` |
| **Validation Split** | 444 | **27,955** | 122,642 | Verified in `pretrain_manifest.json` |
| **Total** | 8,894 | **1,054,337** | 4,623,074 | Fully accounted in manifest |

---

## 3. Linguistic Diversity & Domain Balance

The corpus was tokenized using the canonical frozen BPE tokenizer (`data/experiments/vocab_4096`, 4,096 tokens, SHA-256: `db4d6119...`). The domain and linguistic distribution across the 1.05M tokens is:

1. **English General & Technical (48.4%):** Fundamental factual articles, scientific concepts, programming definitions, and reasoning texts.
2. **Hindi Devanagari (26.1%):** Grammatical Hindi narratives, cultural texts, factual definitions, and conversational phrases.
3. **Hinglish Conversational (16.2%):** Latin-script colloquial Hindi-English code-mixed dialogues and daily queries.
4. **Code / Algorithmic (5.8%):** Python syntax, algorithmic structures, logic gates, and deterministic routines.
5. **Sanskrit / Heritage (3.5%):** Devanagari grammatical sutras and classical verse definitions.

---

## 4. Token Budget Allocation for Controlled Scaling

To ensure that the head-to-head comparison between `ChakrGenesis-17M` and `ChakrGenesis-38M` isolates parameter capacity rather than token exposure:
- **Sequence Length:** Fixed sequence chunking at $L = 256$ tokens with standard `<eos>` boundary separation.
- **Batch Size:** 32 sequences per batch ($8,192$ tokens/batch).
- **Epoch Budget:** 1 full pass over the $1,026,382$ training tokens ($126$ optimization steps per epoch).
- **Target Loss Masking:** Full causal shift ($x_{1:t-1} \to x_{2:t}$), with standard cross-entropy over all valid positions and padding ignored via `ignore_index=-100`.

This establishes the benchmark corpus for evaluating neural scaling laws in ChakrView.
