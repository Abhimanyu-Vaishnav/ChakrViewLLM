# ChakrView Conversational Alpha — Experimental Research Preview (`v0.10.0-alpha.1`)

## Overview

This is an experimental research preview of ChakrView's from-scratch neural conversational model: **ChakrGenesis38M Conversational Alpha**.

> [!WARNING]
> **Scientific Integrity & Boundary Notice:** This is an **experimental research artifact and engineering preview**, NOT a reliable general-purpose conversational assistant. The model has been trained from scratch without external pretrained weights or hosted commercial LLM APIs. While it demonstrates local grammatical patterns, greeting templates, and code formatting, it exhibits expected semantic drift, factual deficits, and context decay resulting from small-scale training (~4.17M tokens).

---

## 1. Model & Architecture Specifications

- **Model Name:** ChakrGenesis38M-Conversational-Alpha
- **Version:** `v0.10.0-alpha.1`
- **Architecture:** Autoregressive Decoder-only Transformer
- **Parameter Count:** 37,499,392 (37.5M unique parameters)
- **Layers:** 12
- **Attention Heads:** 8 Query heads, 4 Key/Value heads (Grouped Query Attention)
- **Hidden Dimension ($d_{\text{model}}$):** 512
- **FFN Hidden Dimension:** 1,408 (SwiGLU)
- **Vocabulary Size:** 4,096 (Canonical Byte-Level BPE)
- **Context Window:** 256 tokens
- **Positional Encoding:** RoPE (Rotary Position Embeddings)
- **Target Git Commit:** `851f45da3e7a78fe00c97d06b5ba076ff1656f1a`

---

## 2. Training Methodology & Data Provenance

1. **Pretraining (Corpus V6):**
   - 1 complete epoch (1,305 optimizer steps) on 4,170,667 active tokens across 9,843 clean documents (Python standard library code/docs, STEM & humanities expository prose, Hindi Devanagari text, and curated dialogues).
   - Validation loss dropped from 8.4240 (PPL: 4,554.99) to **2.9562** (PPL: **19.22**, Accuracy: **41.84%**).
   - Pretrained Base Checkpoint SHA-256: `92f2aa7487361ec112f1e40a8760993f167733a1943a32e50078338e77ecfcea`.

2. **Conversational Supervised Fine-Tuning (SFT):**
   - Ingested 1,612 clean, deduplicated dialogues (English 90.0%, Hindi 7.0%, Hinglish 3.0%).
   - Contamination Firewall: Filtered 75 candidates against 26,590 protected 8-grams from sealed Benchmark V3, V2, and test sets.
   - Response-only loss masking (`ignore_index=-100` on prompt tokens).
   - 3 epochs (273 steps) with AdamW and cosine decay ($5 \times 10^{-5}$).
   - SFT validation loss improved from 4.1777 (27.04% accuracy) to **2.8924** (**49.70% accuracy**).

---

## 3. Conversational Evaluation & Quality Metrics

Evaluated across 12 distinct capability dimensions (24 held-out test prompts):
- **Valid UTF-8 Rate:** **100.0%**
- **Empty Output Rate:** **0.0%**
- **Average 4-gram Repetition:** **0.15%** (no degenerative looping)
- **Observed Strengths:** Natural greeting phrasing (*"Hello! Ready to help..."*), valid Python function syntax headers.
- **Documented Limitations:**
  - *Semantic Drift:* Deep multi-sentence logic drifts into tangential phrasing.
  - *Factual Deficit:* Cannot answer exact factual or historical questions without retrieval.
  - *Context Decay:* Multi-turn entity recall decays over turns.
  - Full taxonomy documented in `MODEL_LIMITATIONS.md`.

---

## 4. Verified Checksums

| Artifact | File Name | Size | SHA-256 Checksum |
|:---|:---|:---:|:---|
| **Conversational SFT Model** | `chakr_38m_conversational_sft.pt` | 143.09 MB | `e4beba3ae20f66eccc2cd06a45582087a059eb9117ed15b8ed9ab8e5500ff106` |
| **Canonical BPE Vocabulary** | `vocab.json` | 110.0 KB | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` |

---

## 5. Local Standalone Inference

Run locally on CPU or GPU without hosted APIs:

```bash
# Clone and checkout release tag
git clone https://github.com/Abhimanyu-Vaishnav/ChakrViewLLM.git
cd ChakrViewLLM
git checkout v0.10.0-alpha.1

# Launch local interactive chat CLI
python chat_cli.py
```

Or via Python:
```python
from chakrview.inference_package import ChakrViewGenerator, ChakrChatSession

generator = ChakrViewGenerator("experiments/cv24h_release/chakr_38m_conversational_sft.pt")
session = ChakrChatSession(generator)
print(session.chat("Hello!"))
```
