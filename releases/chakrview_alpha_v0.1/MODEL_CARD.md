# ChakrView Model Card: ChakrMicro Alpha v0.1

## Model Overview
- **Model Name:** ChakrMicro Alpha v0.1
- **Architecture:** Decoder-only Causal Transformer (RoPE, RMSNorm, SwiGLU, Tied Embeddings)
- **Parameters:** 3,443,136 unique parameters
- **Layers:** 6
- **Hidden Width ($d_{model}$):** 192
- **Attention Heads:** 6 ($d_{head} = 32$)
- **Feedforward Hidden Dim:** 512
- **Context Length:** 512 tokens
- **Vocabulary:** 4,096 tokens (Byte-backed fallback, 724 Devanagari subword merges)
- **Base Pre-Training Checkpoint:** `checkpoint_step_800.pt` (Seed 42)
- **Checkpoint SHA-256:** `8759ccb4a70989b79c0b4b26876da1d47989ebc61f39831750b6acf0455b9943`

---

## Pre-Training Lineage & Evidence
The base model was pretrained from scratch in **UCIP-13** on the Stage C clean multilingual corpus:
- **Corpus Coverage:** English, Hindi, Hinglish, Sanskrit, and Python code across 27 clean shards.
- **Exposure:** 800 optimization steps (batch token exposure: 6,553,600 tokens).
- **Initial Validation Loss:** 8.3084 (Perplexity: 4057.8)
- **Final Validation Loss:** 4.6859 (Perplexity: 108.41)
- **Final Test Loss:** 4.6595 (Perplexity: 105.59)
- **Reproducibility Audit:** Independently audited and verified in UCIP-13R (zero loss discrepancy, bit-exact parameter replication).

---

## Conversational Alignment Status (Readiness for SFT)
- **Current State:** Base autoregressive language model.
- **SFT Dataset Prepared:** 15,000 instruction-response records constructed in UCIP-14R across 18 balanced categories and 4 languages.
- **Causal Loss Alignment:** Mathematically verified in UCIP-14C-R and UCIP-14C-R2. Prompt tokens are strictly masked (`-100`), with zero loss and zero logit gradient leakage. Assistant tokens and EOS receive active loss.
- **Next Stage:** Authorized SFT training (UCIP-15) using the 12,000-sample training partition.

---

## Intended Use & Boundaries
- **Target Platform:** Local CPU execution (Windows, Linux, macOS).
- **Hardware Footprint:** <500MB RAM, no GPU required.
- **Explicit Limitations:**
  - Base checkpoint is not instruction-tuned; it predicts raw autoregressive text continuations.
  - Multi-subword entity binding and factual recall across long multi-turn contexts are constrained at 3.44M parameters.
  - Open-ended reasoning and factual QA without retrieval are prone to hallucination.
