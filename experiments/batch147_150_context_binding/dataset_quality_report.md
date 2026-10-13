# Dataset Quality & Split Isolation Report: Batch 147–150

**Date:** 2026-10-13  
**Milestone:** Batch 147–150 Context Binding & Extractive Learning  
**Dataset:** `ChakrView Context Binding & Extractive Learning v0.7`  
**Location:** `experiments/batch147_150_context_binding/data/`  

---

## 1. Dataset Design & Grounded Specification

To test whether ChakrView's neural core can attend to and extract values directly from an input context (rather than recalling general pretraining templates), we designed a controlled key-value and natural language binding dataset:

### Category Breakdown:
1. **Direct Single Key-Value Bindings:** E.g., `Context: color is blue. Question: What is color?` (Tests primary span attention).
2. **Distractor Key-Value Bindings (Target First & Second):** E.g., `Context: fruit is apple. animal is tiger. Question: What is fruit?` and `Context: city is London. country is UK. Question: What is country?` (Tests selective attention against distractor noise).
3. **Natural Sentence Span Bindings:** E.g., `Context: The package arrived on Monday and contained books. Question: What day did the package arrive?` (Tests unstructured sentence extraction).
4. **Hindi / Indic Bindings:** E.g., `Context: रंग: नीला। Question: रंग क्या है?` (Tests multilingual extractive attention).
5. **Negative / Abstention Cases:** E.g., `Context: mineral is quartz. continent is Asia. Question: What is president?` (Tests abstention when requested key is missing from context).

---

## 2. Split Quantities & Integrity Verification

| Split | Count | SHA-256 | Purpose | Leakage Status |
|:---|:---:|:---|:---|:---|
| **Train** | 24 | `1f34ec...` | Supervised training on structured bindings | Strictly Disjoint |
| **Validation** | 8 | `6a438b...` | Hyperparameter & checkpoint tuning | Zero hash / prompt overlap |
| **Held-Out Test** | 8 | `82136e...` | Final capability assessment on unseen keys/values | **100% Unseen Keys & Combinations** |

- **Exact Content Hash Overlap:** 0 records ($0.0\%$).
- **Prompt String Overlap:** 0 records ($0.0\%$).
- **Held-Out Test Novelty:** The test set evaluates keys that never appeared in the training set (e.g. `material is diamond`, `language is Sanskrit`, `flower is lotus`, `author is Tagore`, `crater named Shackleton`, `धातु: तांबा`).
