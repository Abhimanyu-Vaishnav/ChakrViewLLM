# UCIP-11 Phase F: Controlled Experiment Design & Configuration

**Mission ID:** UCIP-11  
**Architecture:** 12.9M Parameter ChakrMedium (7 layers, width 384, 6 heads, FF 896, vocab 4096)  
**Execution Environment:** CPU-first (Intel i9-13900H, 32 GB RAM, ~2,800 tokens/sec training throughput)  
**Compute Allocation:** Balanced training, continuous validation, and independent benchmark evaluation  

---

## 1. Executive Summary

To systematically resolve whether ChakrView can acquire robust natural-language generation and generalization on held-out tasks without template memorization, we establish 5 strictly controlled experimental conditions:

1. **Condition A — Baseline Control (UCIP-09 Medium):**
   - No additional training.
   - Purpose: Reference baseline for all capability, perplexity, and generalization metrics.
2. **Condition B — Clean Natural-Language Continued Pretraining (CPT):**
   - Starting Checkpoint: UCIP-09 Medium.
   - Dataset: Stage B clean multi-domain corpus (English systems, Hindi computing, Hinglish, Python, logic).
   - Training Objective: Pure causal language modeling (predicting full token sequences).
   - Purpose: Test whether exposure to diverse natural text expands linguistic representations and eliminates repetitive degeneration.
3. **Condition C — Clean Supervised Instruction Tuning (SFT):**
   - Starting Checkpoint: UCIP-09 Medium.
   - Dataset: Exp34 v3 Instruction Mixture (900 clean, diverse instruction-response pairs; CC0-1.0).
   - Training Objective: Supervised autoregressive instruction tuning with loss masking on prompt tokens.
   - Purpose: Measure whether instruction tuning alone produces conversational competence and instruction adherence.
4. **Condition D — Two-Stage Pretraining Followed by Instruction Tuning (CPT + SFT):**
   - Starting Checkpoint: Checkpoint produced by Condition B.
   - Fine-tuned on the Exp34 v3 Instruction Mixture.
   - Purpose: Test whether a strong natural-language pretraining foundation synergizes with instruction tuning to surpass SFT alone.
5. **Condition E — Random-Initialization Language Learning Control (Scratch):**
   - Architecture: Same 12.9M ChakrMedium architecture initialized from random Gaussian weights (seed 42).
   - Trained on Stage B clean corpus.
   - Purpose: Establish the lower bound of genuine language acquisition from scratch vs. inherited checkpoint representations.

---

## 2. Experimental Parameters & Hyperparameter Invariants

| Hyperparameter | Condition B (CPT) | Condition C (SFT) | Condition D (CPT+SFT) | Condition E (Scratch) |
|---|---|---|---|---|
| **Parent Checkpoint** | UCIP-09 (`5513fefe...`) | UCIP-09 (`5513fefe...`) | Condition B Best Ckpt | Random Init (Seed 42) |
| **Training Steps** | 100 steps | 100 steps | 60 steps | 100 steps |
| **Batch Size** | 4 sequences | 4 sequences | 4 sequences | 4 sequences |
| **Max Sequence Length** | 128 tokens | 128 tokens | 128 tokens | 128 tokens |
| **Learning Rate** | $2.0 \times 10^{-5}$ | $1.8 \times 10^{-5}$ | $1.2 \times 10^{-5}$ | $5.0 \times 10^{-4}$ |
| **Weight Decay** | 0.01 | 0.01 | 0.01 | 0.01 |
| **Grad Clip Norm** | 1.0 | 1.0 | 1.0 | 1.0 |
| **Label Smoothing** | 0.00 (Pure LM) | 0.05 (Prevents collapse) | 0.05 | 0.00 |
| **Random Seed** | 42 | 42 | 42 | 42 |
| **Tokens Processed** | ~51,200 tokens | ~51,200 tokens | ~30,720 tokens | ~51,200 tokens |
| **Execution Time (Est.)**| ~20–25 seconds | ~20–25 seconds | ~15 seconds | ~20–25 seconds |

---

## 3. Strict Scientific Controls

- **Zero Evaluation Contamination:** Neither the development suite nor the sealed suite will ever be observed during training.
- **Identical Decoding Protocol:** All evaluations use `ModelAdapter` with greedy argmax decoding (`temperature = 0.0`, `repetition_penalty = 1.05`, `no_repeat_ngram_size = 3`).
- **Comprehensive Logging:** Step-by-step training loss, validation cross-entropy, elapsed time, and sample generation outputs will be logged at regular 20-step evaluation intervals.
