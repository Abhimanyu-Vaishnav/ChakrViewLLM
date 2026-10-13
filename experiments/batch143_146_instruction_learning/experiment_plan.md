# Controlled Experiment Plan: Batch 143–146 Instruction Learning

**Date:** 2026-10-13  
**Milestone:** Batch 143–146  
**Parent Checkpoint:** Canonical Step 10,000 (`experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`)  
**SHA-256:** `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`  

---

## 1. Objectives

1. Determine whether supervised fine-tuning (SFT) with exact prompt loss-masking can instill basic instruction-following behavior in the small 3.44M parameter ChakrMicro neural core without destroying base language comprehension.
2. Compare the canonical baseline against a minimal SFT recipe and a conservative warmup + cosine annealed SFT recipe.
3. Evaluate both teacher-forced cross-entropy loss and actual free-running generation across all 5 domains (English, Code, Math, Hindi, Grounded Science).
4. Measure catastrophic forgetting on the held-out Stage C raw language test split.

---

## 2. Experimental Arms

| Parameter | Arm A (Baseline) | Arm B (Constant LR SFT) | Arm C (Warmup + Cosine SFT) | Arm D (Conservative Low-LR SFT) |
|:---|:---|:---|:---|:---|
| **Base Weights** | Canonical Step 10,000 | Canonical Step 10,000 | Canonical Step 10,000 | Canonical Step 10,000 |
| **Training Steps** | 0 | 50 | 50 | 50 |
| **Batch Size** | N/A | 2 | 2 | 2 |
| **Peak LR** | N/A | $5.0\times 10^{-5}$ | $5.0\times 10^{-5}$ | $2.0\times 10^{-5}$ |
| **Min LR** | N/A | $5.0\times 10^{-5}$ | $5.0\times 10^{-6}$ | $2.0\times 10^{-6}$ |
| **Warmup Steps** | N/A | 0 | 10 (20%) | 10 (20%) |
| **LR Schedule** | None | Constant | Linear Warmup + Cosine Decay | Linear Warmup + Cosine Decay |
| **Loss Masking** | N/A | Strict Prompt Masking | Strict Prompt Masking | Strict Prompt Masking |
| **Optimizer** | None | AdamW ($wd=0.01$) | AdamW ($wd=0.01$) | AdamW ($wd=0.01$) |
| **Grad Clip** | None | 1.0 | 1.0 | 1.0 |

---

## 3. Evaluation Dimensions

1. **Instruction Validation Loss & Perplexity:** Evaluated on the 5 held-out validation instruction examples.
2. **Instruction Held-Out Test Loss & Perplexity:** Evaluated on the 6 held-out test instruction examples (completely unseen prompts and tasks).
3. **Stage C Raw Language Retention:** Evaluated on 10 batches of held-out Stage C test tokens (5,120 tokens) to detect catastrophic forgetting of pretraining distribution.
4. **Free-Running Response Quality:** Generation from user prompts (max 32 tokens, greedy decoding) assessing response formatting, repetition ratio, and answer grounding.
