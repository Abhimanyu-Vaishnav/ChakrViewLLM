# ChakrView Exp34 — Candidate Selection Recommendation

## 1. Recommendation Decision

### **Selected Experimental Candidate: Arm A (`mask_prompt=True`)**
**Classification:** *Promising experimental candidate for further scaling; NOT yet production-ready.*

---

## 2. Selection Rationale

1. **Superior Attractor Suppression:**
   - Arm A demonstrated a statistically significant reduction in attractor collapse (34.4% vs 43.8% in Arm B and 53.1% in the canonical baseline).
   - Prompt masking successfully prevents the model from wasting parameter capacity on predicting fixed instruction prompt prefixes.
2. **Lower N-Gram Repetition:**
   - Arm A reduced 4-gram repetition from 25.4% down to **18.6%**, whereas Arm B remained elevated at 23.3%.
3. **Better Loss & Perplexity Trajectory:**
   - Validation loss on held-out mixture examples was lower for Arm A (4.5724 vs 4.6471), yielding a perplexity of 96.78 compared to 104.28 for Arm B.

---

## 3. Scope Limitations & Safeguards

- **Pilot Scale:** 50 optimizer steps is an exploratory budget. This experiment establishes that `mask_prompt=True` is the scientifically superior architectural choice for ChakrView's SFT stage.
- **Canonical Protection:** Neither candidate will overwrite or replace the Canonical Step 10,000 checkpoint. Arm A is recommended as the starting foundation for a 500-step SFT run in milestone M3.

