# Controlled Experiment Plan: Batch 147–150 Context Binding

**Date:** 2026-10-13  
**Milestone:** Batch 147–150 Context Binding & Grounded Answers  
**Base Checkpoint:** Canonical Step 10,000 (`experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`)  
**SHA-256:** `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`  

---

## 1. Experimental Objectives

1. Quantify the canonical baseline model's raw context-binding capability on unseen key-value prompts.
2. Compare direct SFT context-binding training across conservative learning rates with linear warmup and cosine decay.
3. Compare neural generation against a deterministic regex/key-value baseline extractor.
4. Measure retention on raw Stage C pretraining text to verify non-catastrophic forgetting.
5. Evaluate exact-match accuracy, selective distractor resistance, and negative abstention behavior.

---

## 2. Experimental Arms

| Arm | Description | Steps | Peak LR | Min LR | Warmup | Schedule |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| **Arm A (Baseline)** | Canonical Step 10,000 (no task training) | 0 | N/A | N/A | N/A | None |
| **Arm B (Direct SFT)** | Constant LR Binding SFT | 60 | $5.0\times 10^{-5}$ | $5.0\times 10^{-5}$ | 0 | Constant |
| **Arm C (Warmup+Cosine SFT)** | Warmup 12 steps + Cosine Decay Binding SFT | 60 | $5.0\times 10^{-5}$ | $5.0\times 10^{-6}$ | 12 (20%) | Warmup + Cosine Decay |
| **Arm D (Conservative SFT)** | Conservative Low-LR SFT | 60 | $2.0\times 10^{-5}$ | $2.0\times 10^{-6}$ | 12 (20%) | Warmup + Cosine Decay |
| **Arm E (Deterministic)** | Deterministic Key-Value Parser Baseline | N/A | N/A | N/A | N/A | Exact Regex / Heuristic |

---

## 3. Evaluation Protocol

All models are evaluated on:
1. **Validation Binding Loss & PPL:** 8 unseen validation binding examples.
2. **Held-Out Test Binding Loss & PPL:** 8 held-out test binding examples.
3. **Exact Answer Match Rate (Neural):** Fraction of held-out test questions where the neural model's greedy generation matches the target value.
4. **Distractor Resistance:** Accuracy on multi-key examples where distractors appear before or after the target key.
5. **Abstention Accuracy:** Correct output on queries asking for keys missing from the context.
6. **Stage C Raw Language Loss:** Evaluated on 10 batches (5,120 tokens) of held-out Stage C test shards.
