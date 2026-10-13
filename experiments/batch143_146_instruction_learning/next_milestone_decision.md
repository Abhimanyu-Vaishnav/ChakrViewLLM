# Next Milestone Decision: Batch 143–146 to Batch 147+

**Date:** 2026-10-13  
**Status:** EVIDENCE-BASED DECISION RATIFIED  

---

## 1. Experimental Evidence Summary

1. **SFT Dramatically Reduces Instruction Cross-Entropy:**
   Prompt-masked SFT reduced held-out test instruction loss from $5.0877 \to 2.7641$ (Arm B) and $3.1184$ (Arm C), eliminating wikitext table artifacts and instilling ChatML role formatting.
2. **Warmup + Cosine Protects Pretraining Memory:**
   Arm C ($lr=5\times 10^{-5} \to 5\times 10^{-6}$, 10 warmup steps) retained 99.5% of raw Stage C pretraining perplexity ($27.26 \to 27.74$).
3. **Capacity Threshold Identified:**
   At 3.44M parameters, the neural core can learn *format syntax* (ChatML tags, function signatures, prompt-response turn boundaries), but cannot reliably synthesize factual world knowledge or multi-step reasoning from parametric weights alone.
4. **Hybrid Retrieval Is Stable:**
   `HybridRetriever` successfully performs hybrid lexical/semantic retrieval and passes grounded contexts into the cognitive prompt envelope.

---

## 2. Checkpoint Promotion Decision

- **Canonical Checkpoint Status:** **FROZEN & IMMUTABLE.**
- As with Batch 139–142, the canonical checkpoint remains `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`.
- Candidate SFT checkpoints (`arm_b_sft_constant.pt`, `arm_c_sft_warmup_cosine.pt`, `arm_d_sft_conservative.pt`) are preserved with their exact hashes in `experiments/batch143_146_instruction_learning/checkpoints/` as candidate models for conversational runtime experiments.

---

## 3. Decision for Batch 147+: Contextual Span-Copy & Grounded Answering

Now that the model responds to ChatML prompts without wikitext artifacts, the next logical capability gate is:
1. **Contextual Span-Extraction SFT:** Train the model specifically on short-context extractive QA (e.g., `"Context: X is Y. Question: What is X? Answer: Y."`) to teach the self-attention heads to bind to tokens in the prompt context.
2. **Integrate Runtime Session ChatML:** Finalize and test the uncommitted `chakrview/runtime/session.py` ChatML formatting integration with the newly validated conversation template.
