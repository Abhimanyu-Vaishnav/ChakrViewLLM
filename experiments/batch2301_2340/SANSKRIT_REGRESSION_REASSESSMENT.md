# SANSKRIT REGRESSION REASSESSMENT

**Batch:** 2301–2340  
**Focus:** Re-evaluation of Classical Sanskrit Drift on `wikipedia_sa_bhagavad_gita`  
**Status:** REGRESSION INDEPENDENTLY CONFIRMED; ROOT CAUSE VERIFIED  

---

## 1. Verified Evidence of Held-Out Sanskrit Regression

Direct evaluation on the 61 held-out documents (19,577 tokens) of `wikipedia_sa_bhagavad_gita` demonstrates:
- **Baseline Token-Weighted Cross-Entropy:** `4.3202` (Perplexity: `75.20`)
- **Seed 42 Token-Weighted Cross-Entropy:** `4.3978` (Perplexity: `81.27`)
- **Regression Magnitude ($\Delta$):** `+0.0776` cross-entropy (+6.07 perplexity)
- **Macro Document Cross-Entropy:** Baseline `4.2906` $\to$ Candidate `4.3718` ($\Delta = +0.0812$)

In contrast, Hindi literature text (`wikipedia_hi_sanskrit_sahitya`, 9 documents, 4,798 tokens) did not regress:
- Baseline CE: `3.9203` $\to$ Candidate CE: `3.9057` ($\Delta = -0.0146$)

This proves that the regression is strictly localized to pure Devanagari Classical Sanskrit text.

---

## 2. Characterization: "Held-Out Source-Family Regression", Not Catastrophic Collapse

The term "catastrophic forgetting" typically connotes total loss of representational competence (e.g. perplexity exploding into thousands). 
Here, Sanskrit perplexity increased from 75.20 to 81.27 (+8.07% relative increase). 
Therefore, this phenomenon is accurately characterized as **held-out source-family regression due to zero training coverage**, rather than catastrophic collapse.

### Verified Root Cause
In `SPLIT_V2_MANIFEST.json` and `CORPUS_V2_CLEANED_MANIFEST.json`:
- Pure Sanskrit documents were allocated **100% to the held-out validation and test partitions**.
- The CPT training split contained English geometry, English information theory, Hindi literature, and technical code, but **0 tokens of Classical Sanskrit**.
- During 125 steps of CPT, the model updated weights exclusively toward English and Hindi objective functions, allowing unconstrained latent drift on pure Sanskrit transition probabilities.

---

## 3. Pre-Registered Follow-Up Intervention Protocol (Proposed Only — No Training Authorized)

When model training is authorized in future missions, the following controlled curriculum intervention should be executed:

1. **Source Disjoint Sanskrit Ingestion:**
   - Ingest an independent Classical Sanskrit corpus (e.g. *Hitopadesha* or *Panchatantra*, strictly non-overlapping with *Bhagavad Gita*).
2. **Stratified Split Allocation:**
   - Partition into 60% Train, 20% Val, 20% Test, ensuring zero leakage into `wikipedia_sa_bhagavad_gita`.
3. **Balanced CPT Blend:**
   - Blend 15% Sanskrit, 45% English math/tech, 40% Hindi literature.
4. **Verification Gates:**
   - Baseline Sanskrit preservation gate: `wikipedia_sa_bhagavad_gita` CE $\le 4.3202$.
   - Cross-domain preservation gate: `gutenberg_loney` CE $\le 4.1495$ and `wikipedia_en_information_theory` CE $\le 4.4737$.
