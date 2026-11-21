# GENESIS-NR-001: Logits & Generation Forensics Report

**Date:** 21 November 2026  
**Mission ID:** CV-GENESIS-NR-001  
**Project:** ChakrView  

---

## 1. Origin of `Gou I hoon! I hoon!` Forensics

Step-by-step logit inspection of `chakr_ultra_best.pt` vs `genesis_sft_best.pt` revealed the exact mechanism:

1. **In `chakr_ultra_best.pt` (Old Ultimate Brain):**
   - Lacking pretraining, the model has high entropy and no semantic language foundation.
   - For prompt `"User: hi chakra\nAssistant:"`, step 1 predicts:
     - Top token: `'The '` (p=0.0154), `'A'` (0.0147), `'G'` (0.0138).
   - In greedy/temperature sampling with un-normalized prompts (containing trailing spaces `Assistant: `), token 74 (`'G'`) + token 120 (`'ou'`) + token 35 (`' '`) + token 76 (`'I'`) trigger a positive feedback loop into token 441 (`' hoon!'`).
   - Entropy drops to near zero ($0.02$), causing the infinite loop: `'Gou I hoon! I hoon! I hoon!'`.

2. **In `genesis_sft_best.pt` (Project Genesis):**
   - Grounded in 6.65M tokens of autoregressive language pretraining.
   - For prompt `"User: Hello! Who are you?\nAssistant:"`:
     - Step 1: Predicts `' '` ($p=0.9902$, entropy $0.10$).
     - Step 2: Predicts `'He'` ($p=0.6498$).
     - Steps 3–5: Completes `'Hello!'` with high confidence.
     - Final Output: `"Hello! I am doing well and ready to assist you. How can I help you today?"`
   - **Zero instances of `Gou I hoon!` occur in `genesis_sft_best.pt`.**
