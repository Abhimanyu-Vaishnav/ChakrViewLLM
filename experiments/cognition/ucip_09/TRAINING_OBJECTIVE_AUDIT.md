# UCIP-09: Training Objective & Corpus Forensics Audit

**Project:** ChakrView  
**Experiment Branch:** `exp/cbr-01-conversational-breakthrough`  
**Date:** November 17, 2026  

---

## 1. Corpus Quantity and Target Diversity

An empirical audit of all cognitive training corpora in the repository revealed:

| Corpus Name | Record Count | Unique Prompts | Unique Targets | Duplicate Target Ratio | Core Finding |
|---|---|---|---|---|---|
| `ucip06_train.jsonl` | 1,562 | 1,562 | 692 | **55.7% Target Duplication** | Re-used 692 response templates across 1,562 synthetically permuted prompts. |
| `ucip07` Training Run | *None Created* | *N/A* | *N/A* | *N/A* | Re-used `ucip06_train.jsonl` directly without adding new data. |
| `ucip08_train.jsonl` | 734 | 734 | 528 | **28.0% Target Duplication** | Cleaned prompt stems and removed benchmark overlap, but total dataset size remained compact (734 samples). |

### Key Corpus Findings:
1. **Severe Data Volume Deficit:** Total training exposure across UCIP-06, 07, and 08 was limited to 700–1,500 conversational pairs. At 12.9M parameters, 1,500 samples are insufficient to internalize broad factual world knowledge and robust grammar.
2. **Template Entrainment:** The high ratio of duplicated targets in `ucip06_train.jsonl` (only 692 unique targets across 1,562 rows) entrained the model on fixed template completions (e.g., repeatedly generating `"coins in total"`, `"Alpha-Omega"`), resulting in catastrophic template hallucination when presented with unseen entities.

---

## 2. Training Objective Forensics

### 2.1 Prompt Masking & Loss Calculation
In `train_ucip08.py`:
- User prompt tokens are correctly assigned target `-100`, masking them out of cross-entropy loss computation.
- Loss is computed strictly over the Assistant turn tokens + `[EOS]`.
- Label smoothing ($\epsilon = 0.05$) was successfully applied, preventing the model from collapsing logits into extreme overconfidence.

### 2.2 Teacher-Forced Loss vs Autoregressive Generalization
- In `train_ucip08.py`, teacher-forced validation loss dropped steadily from **1.9126 to 1.4452**.
- However, as demonstrated in Phase B, teacher-forced token prediction does not immunize the model against autoregressive divergence if the dataset lacks syntactic breadth.
- **Exposure Bias:** In training, the model always conditions on ground-truth prior assistant tokens. During inference, any slightly sub-optimal token immediately feeds into subsequent steps, triggering a compounding drift.

---

## 3. Required Training & Objective Improvements for UCIP-09

1. **Substantial Corpus Scaling:** Expand training dataset from 734 samples to **3,000–5,000 clean, diverse conversational and multi-capability samples** across English, Devanagari Hindi, Roman Hindi, Hinglish, and Arithmetic.
2. **Entity & Template Diversity:** Eradicate repetitive answer templates; ensure every arithmetic problem, technical explanation, and conversational reply has a distinct grammatical structure and variable entities.
3. **Turn Consistency:** Standardize all training records on the strict `User: {prompt}\nAssistant: {target}` dialogue convention with explicit `[EOS]` termination.
