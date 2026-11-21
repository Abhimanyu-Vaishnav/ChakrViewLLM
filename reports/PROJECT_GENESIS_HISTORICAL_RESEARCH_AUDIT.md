# Project Genesis: Historical Research Audit & Consolidation Map

**Date:** 21 November 2026  
**Project:** ChakrView (ULTIMATE-BRAIN-003)  
**Evaluator:** Principal Foundation-Model Architect & Cognitive Systems Researcher  
**Git HEAD:** `48d52ff`  

---

## 1. Historical Architecture & Experiment Lineage

Across the ChakrView research campaigns (v0.9, ChakrMicro, ChakrConv, ChakrUltra, UCIP-01 through UCIP-29, Ultimate Brain 001/002), models have repeatedly suffered from generation failure, token collapse, attractor loops, and lack of genuine linguistic generalization.

| Model / Milestone | Parameters | Training Data | Primary Failure Mode | Root Cause Classification |
| :--- | :---: | :---: | :--- | :--- |
| **ChakrMicro** | ~1.2M | Synthetic arithmetic & small texts | Immediate repetition loop; no syntax | **Data & Scale:** <500K tokens, shallow capacity |
| **ChakrConv v0.1** | 3.44M | 680 conversational pairs | 1.67% exact match; gibberish | **Optimization & Pretraining:** Jumped directly to SFT without pretraining on natural language |
| **Stage-C Baseline** | 4.88M | 6.65M tokens (27 shards) | Low validation loss (1.89 NLL) but brittle generation | **Tokenizer & Damping:** High residual norms in late FFN layers; training/inference mismatch |
| **UCIP-20 to 28** | 4.88M | Curricula & Ablations | Identified Layers 4 & 5 FFN attractor circuits | **Architecture & Residuals:** SwiGLU gating dynamics concentrated attractors in late layers |
| **ChakrUltra (UB-001)** | 5.18M | 545 SFT examples | Severe generation collapse (`'Gou I'`, `'The ker'`) | **Severe Under-Training:** Trained on only 545 examples for 8 epochs on CPU; pure neural generation was completely untrained |
| **Ultimate Brain 002** | 5.18M + Hybrid | 545 SFT + Rule Router | Hybrid scored 100%, but Pure Neural was 0% | **Evaluation & Scaffolding Defect:** Deterministic response policies masked neural failure |

---

## 2. Root Cause Classification Matrix

1. **Pretraining Void (The SFT-From-Scratch Fallacy):**
   - ChakrConv and ChakrUltra attempted conversational fine-tuning directly on 545–680 Q&A pairs without foundational autoregressive language pretraining. A 5M parameter neural network cannot learn English, Hindi, syntax, grammar, and reasoning from 500 conversation snippets.
2. **Hardware Under-Utilization (CPU-Bound Bottleneck):**
   - Previous runs trained exclusively on CPU using `.venv` (Torch 2.14.0+cpu), achieving only ~200–500 tokens/sec. Consequently, training was arbitrarily truncated to 8 epochs or ~500 steps.
   - An audited NVIDIA RTX A1000 6GB Laptop GPU is available via `.venv_gpu` (Torch CUDA 12.6), providing **24,396 tokens/sec (87.8M tokens/hr)**, enabling comprehensive multi-million token pretraining within minutes to hours.
3. **Tokenizer Artifacts & Multi-Lingual Consistency:**
   - The frozen `v4096` BPE tokenizer (`experiments/tokenizer/candidates/v4096`) contains full byte-level coverage, lossless Devanagari/Hindi encoding, code tokens, and special tokens (`<pad>`, `<bos>`, `<eos>`).
   - Training corpora (such as the 6.65M token Stage-C dataset) were pre-tokenized against this exact vocabulary.
4. **Attractor Gating in Late FFN Layers:**
   - UCIP-27 and UCIP-28 demonstrated that late SwiGLU layers (layers 4–5) can become attractor sinks without proper residual scaling and pre-layer normalization.
   - Genesis architecture introduces pre-RMSNorm, RoPE, Grouped-Query Attention (GQA), and stabilized SwiGLU with residual damping ($\frac{1}{\sqrt{2N}}$).

---

## 3. Engineering Decisions: Retain, Redesign, Abandon

- **ABANDON:**
  - Training neural models from scratch on only 500 conversational examples.
  - Relying on CPU for autoregressive language pretraining when a CUDA GPU is available.
  - Using hardcoded response templates to artificially claim neural conversational capability.
- **RETAIN:**
  - Frozen `v4096` Byte-level BPE tokenizer specification (`vocab_size=4096`).
  - Audited Stage-C multi-domain pretraining dataset (6.65M tokens across 27 binary shards).
  - Clean cognitive hybrid components (relational persistent memory, intent dispatch router, multi-container arithmetic solver, principled personal abstention) for System B.
- **REDESIGN:**
  - **System A Neural Architecture (`ChakrGenesis`):** Modern pre-RMSNorm decoder with RoPE positional embeddings, Grouped-Query Attention (GQA), SwiGLU FFN with residual scaling, and unified KV cache.
  - **Two-Stage Training Pipeline:**
    - *Stage 1 (Language Foundation Pretraining):* Autoregressive next-token prediction over multi-million token general language corpus (Stage-C).
    - *Stage 2 (Conversational & Reasoning SFT):* Assistant-masked instruction tuning with pretraining replay.
  - **Transparent Dual-Track Evaluation:** Independent Track A (Pure Neural) vs. Track B (Full Cognitive Hybrid).
