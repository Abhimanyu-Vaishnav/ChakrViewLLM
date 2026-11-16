# UCIP-05: Model Card — ChakrView Foundational Neural Generation Candidate

**Release Candidate:** `releases/ucip_05_candidate/`  
**Date:** 2026-11-16  
**License:** Project Internal / Sovereign Local-First  

---

## 1. Model Overview

The UCIP-05 release candidate series packages two foundational autoregressive checkpoints trained to recover coherent language generation and eliminate repetitive attractor basins:
- **ChakrMedium Candidate (`chakrview_ucip05_medium_model.pt`)**: Primary 12.93M parameter candidate.
- **ChakrMicro Candidate (`chakrview_ucip05_micro_model.pt`)**: Compact 3.44M parameter candidate.

---

## 2. Architectural Specifications

| Parameter | ChakrMedium Candidate | ChakrMicro Candidate |
|---|---|---|
| Total Parameters | 12,932,736 | 3,443,136 |
| Hidden Dimension ($d_{\text{model}}$) | 384 | 192 |
| Transformer Layers | 7 | 6 |
| Attention Heads | 6 | 6 |
| Feed-Forward Dimension ($d_{\text{ffn}}$) | 896 | 512 |
| Context Window | 512 tokens | 512 tokens |
| Vocabulary Size | 4,096 tokens (BPE) | 4,096 tokens (BPE) |
| Weight Tying | True (Embedding & Output) | True (Embedding & Output) |
| SHA-256 Checksum | `089335f838c3e9999600144edccbb1caf1a4a741f43f17d493625b457e90e234` | `25933cc577a3c956d5359e9924eba22e403ef738df8bcac1880a212f8cb5b077` |

---

## 3. Training Details

- **Dataset:** `experiments/cognition/ucip_05/data/ucip05_train.jsonl` (SHA-256 `4b69c71d...`, 822 curated balanced dialogue records).
- **Optimizer:** AdamW ($\text{lr} = 7 \times 10^{-5}$ for Medium, $6 \times 10^{-5}$ for Micro; $\beta_1 = 0.9, \beta_2 = 0.98$, weight decay = 0.01).
- **Gradient Clipping:** 1.0.
- **Batch Size:** 4 with gradient accumulation steps 2 (effective batch size 8).
- **Parent Checkpoints:**
  - ChakrMedium: `experiments/cognition/ucip_01r/training/ucip01r_13m_recovered_best.pt`
  - ChakrMicro: `releases/v1.0_unified_conversational_intelligence/chakrview_v011_model.pt`

---

## 4. Benchmark Performance & Capabilities

Evaluated against the frozen 300-probe benchmark (`experiments/cognition/ucip_05/baseline_benchmark.json`):

- **Owner-Observed Failures Recovered:**
  - `Hello ChakrView. How are you?` $\to$ Generates fluent greeting without corrupt blends.
  - `What does a CPU do?` $\to$ Generates accurate fetch-decode-execute explanation.
  - `Say exactly: The sky is blue.` $\to$ 100% exact copy reproduction.
  - `नमस्ते! आप क्या कर सकते हैं?` $\to$ Coherent Devanagari Hindi persona description.
  - Attractor basin (`"Your project is a computer..."`) completely eliminated.
- **Overall Benchmark Accuracy:** 22.0% across 300 unseen probes (vs 9.3% in UCIP-04 Micro).

---

## 5. Limitations & Future Work

- **Arithmetic Precision:** Performs elementary arithmetic structures, but exact multi-step calculation requires external tools or further token supervision.
- **Capacity Limits:** Out-of-domain generalisation on complex questions is bounded by compact parameter scale (12.9M).
- **Experimental Classification:** Preserved under `releases/ucip_05_candidate/` for controlled owner verification.
