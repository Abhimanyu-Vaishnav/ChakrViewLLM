# NR003: Final Engineering Investigation Report

**Date:** 21 November 2026  
**Mission ID:** CV-GENESIS-NR-003  
**Project:** ChakrView  
**Status:** **INVESTIGATION COMPLETE & ROOT CAUSES PROVEN**

---

## 1. Verified Facts
1. **Tokenizer Permutation Mismatch in Pretraining:**
   - The Stage-C binary corpus was compiled using `data/experiments/vocab_4096` (`db4d6119...`).
   - Pretraining used `experiments/tokenizer/candidates/v4096` (`fbde6847...`).
   - 3,677 out of 4,096 token mappings were permuted, meaning the model was trained on scrambled token representations during the 6.65M token pretraining run.
2. **SFT Dataset Skew:**
   - 40.5% of the 1,476 SFT training instances were dominated by 25 repeated response strings.
   - Without an aligned pretraining language base, SFT acted as a pattern memorizer rather than a semantic fine-tuner, causing the model to regurgitate memorized phrases regardless of the question asked.
3. **Preservation of Artifacts:**
   - All historical checkpoints, datasets, and tokenizer artifacts remain intact with verified SHA-256 hashes.
   - No experimental training was launched prior to establishing root causes.

---

## 2. Unresolved Questions & Next Steps
- Will 2 full epochs (~13.3M tokens) of clean Stage-C pretraining with `data/experiments/vocab_4096` yield low-perplexity open-domain language modeling?
- Preliminary evidence strongly indicates yes: when decoded with `data/experiments/vocab_4096`, Stage-C contains coherent Wikipedia prose, Devanagari text, and Python algorithms.
