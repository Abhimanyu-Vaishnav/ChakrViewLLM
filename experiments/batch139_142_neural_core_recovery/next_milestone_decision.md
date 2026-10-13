# Next Milestone Decision: Batch 139–142 to Batch 143+

**Date:** 2026-10-13  
**Decision Authority:** Senior ML Systems Engineer & Neural-Network Researcher  
**Status:** EVIDENCE-BASED DECISION RATIFIED  

---

## 1. Experimental Evidence Summary

1. **Diagnosis Confirmed:** The regression in Batch 135–138 was caused by learning rate shock (abrupt $3.0\times 10^{-4}$ LR without warmup) and cold optimizer state on converged Step 10,000 weights.
2. **Warmup + Cosine Recovery:** Applying a 50-step linear warmup ($0 \to 1.5\times 10^{-4}$) followed by cosine decay down to $2.0\times 10^{-5}$ eliminated 90%+ of the loss regression across all domains.
3. **No False Promotion:** None of the 250-step candidate runs outperformed the canonical baseline on global held-out test perplexity ($21.59$ baseline vs $21.71$ Arm D). Therefore, **no candidate checkpoint is promoted**. The canonical baseline checkpoint remains `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`.
4. **Structural Limit of Short-Horizon Pretraining:** Repeated short-burst training (100–250 steps) on pretraining data does not yield general intelligence or eliminate repetition loops.

---

## 2. Decision: Milestone Roadmap

### Option 1: Blindly Scale Unsupervised Pretraining to 2,500 Steps
- *Pros:* More tokens might eventually achieve a slight reduction in cross-entropy.
- *Cons:* CPU throughput (~1,200 tok/s) means 2,500 steps takes ~20–30 minutes without addressing the core repetition attractor or lack of instruction grounding. Stage C data is already familiar to the baseline.
- *Verdict:* **REJECTED as primary strategy.**

### Option 2: Curated Multi-Turn SFT / Instruction Grounding on Neural Core (Recommended)
- *Rationale:* The neural core has basic language primitives (learned in 10,000 pretraining steps), but lacks conversational framing and question-answering structure. As observed in the user's `session.py` diff, conversation templates (ChatML `<|im_start|>`) are being integrated into the runtime.
- *Next Step:* Transition to a controlled, multi-domain instruction-tuning (SFT) curriculum using structured prompt-response pairs (reasoning, code, QA, Hindi/English) with loss masking on prompt tokens.
- *Verdict:* **ACCEPTED.**

---

## 3. Actionable Prerequisites for Batch 143+

1. **Retain the Resumable Warmup Engine:** The `get_lr_scheduler` and `train_candidate` harness developed in Batch 139–142 will serve as the engine foundation for future learning experiments.
2. **Incorporate Prompt Loss Masking:** Ensure future training scripts apply loss masks to prompt tokens so that gradients are computed exclusively on target completions.
3. **Multi-Domain Benchmark Expansion:** Maintain the multi-shard test harness (English, Hindi, Code) established in Batch 139–142 as the standard gate before accepting any checkpoint.
