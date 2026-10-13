# Controlled Experiment Plan: Batch 139–142 Neural-Core Recovery

**Date:** 2026-10-13  
**Milestone:** Batch 139–142  
**Parent Checkpoint:** Canonical Step 10,000 (`experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`)  
**SHA-256:** `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`  

---

## 1. Experimental Goals

1. **Verify Baseline Reference (Arm A):** Quantify canonical Step 10,000 baseline loss and perplexity on global splits and individual domain shards (English, Hindi, Code).
2. **Replicate Previous Recipe (Arm B):** Confirm the Batch 135–138 regression under exact evaluation conditions (100 steps, $lr=3\times 10^{-4}$ constant, balanced sampling).
3. **Conservative Warmup + Cosine Annealing (Arm C):** Test hypothesis that learning rate shock caused the regression. Run 250 steps starting with 50 steps warmup ($0 \to 1.5\times 10^{-4}$) followed by cosine decay down to $2.0\times 10^{-5}$ on natural sampling.
4. **Controlled Domain-Balanced Sampling (Arm D):** Isolate the effect of domain weighting. Run 250 steps under the identical warmup + cosine schedule as Arm C, but with balanced Indic/English domain sampling.

---

## 2. Experimental Arms Specification

| Parameter | Arm A (Baseline) | Arm B (Replicated Pilot) | Arm C (Warmup + Cosine Natural) | Arm D (Warmup + Cosine Balanced) |
|:---|:---|:---|:---|:---|
| **Base Checkpoint** | Step 10,000 | Step 10,000 | Step 10,000 | Step 10,000 |
| **Training Steps** | 0 | 100 | 250 | 250 |
| **Tokens Processed** | 0 | 51,200 | 128,000 | 128,000 |
| **Batch Size** | N/A | 8 | 8 | 8 |
| **Sequence Length** | 64 | 64 | 64 | 64 |
| **Peak Learning Rate** | N/A | $3.0\times 10^{-4}$ | $1.5\times 10^{-4}$ | $1.5\times 10^{-4}$ |
| **Min Learning Rate** | N/A | $3.0\times 10^{-4}$ | $2.0\times 10^{-5}$ | $2.0\times 10^{-5}$ |
| **Warmup Steps** | N/A | 0 | 50 (20%) | 50 (20%) |
| **LR Schedule** | None | Constant | Linear Warmup + Cosine Decay | Linear Warmup + Cosine Decay |
| **Optimizer** | None | AdamW ($wd=0.01$) | AdamW ($wd=0.01$) | AdamW ($wd=0.01$) |
| **Grad Clip Norm** | None | 1.0 | 1.0 | 1.0 |
| **Sampling Mode** | N/A | Balanced (50% Indic) | Natural Uniform | Balanced (50% Indic) |
| **Seed** | 42 | 42 | 42 | 42 |

---

## 3. Evaluation Protocol

All models are evaluated under identical conditions:
1. **Global Validation:** 20 batches of batch_size=8, seq_len=64 (10,240 tokens), seed 101.
2. **Global Test:** 20 batches of batch_size=8, seq_len=64 (10,240 tokens), seed 202.
3. **Domain Breakdown (Held-out Test Shards):**
   - **English:** `data/tokenized/stage_c/test/shard_00000.bin` (20 batches, seed 301)
   - **Hindi:** `data/tokenized/stage_c/test/shard_00001.bin` (20 batches, seed 302)
   - **Code / Reasoning:** `data/tokenized/stage_c/test/shard_00002.bin` (20 batches, seed 303)
4. **Qualitative Generation:** Greedy autoregressive continuation (32 new tokens) from 4 standard prompts:
   - *English:* `"The development of modern science began with"`
   - *Code:* `"def binary_search(arr, target):"`
   - *Reasoning:* `"If all mammals are warm-blooded and a whale is a mammal, then"`
   - *Hindi:* `"भारत एक विशाल और विविधतापूर्ण"`
5. **Generation Diagnostics:** Repetition ratio (fraction of repeated 3-grams) and token diversity (unique tokens / total tokens).
