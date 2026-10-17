# SANSKRIT REGRESSION FINAL GATE

**Batch:** 2341–2380  
**Focus:** Definitive Evidence Assessment of Classical Sanskrit Drift on `wikipedia_sa_bhagavad_gita`  
**Status:** REGRESSION CONFIRMED; CAUSAL ATTRIBUTION BOUNDED  

---

## 1. Verified Quantitative Metrics

Based on the verified 61 held-out test documents (19,577 tokens) of `wikipedia_sa_bhagavad_gita`:

| Metric | Canonical Baseline (Step 10,000) | CPT Seed 42 Candidate (Step 125) | Difference ($\Delta$) | Direction |
| :--- | :---: | :---: | :---: | :---: |
| **Exact Unrounded NLL Sum** | 84,575.7604 | 86,095.3015 | +1,519.5411 | Regression |
| **Exact Token-Weighted CE** | **4.320159** | **4.397778** | **+0.077619** | **Regression (+0.0776)** |
| **Perplexity (PPL)** | **75.20** | **81.27** | **+6.07** | **Degradation (+8.07% rel)** |
| **Macro Document Mean CE** | 4.2906 | 4.3718 | +0.0812 | Regression |

---

## 2. Source-Family Specificity

- **Classical Sanskrit Literature Commentary (Hindi):**
  - Source: `wikipedia_hi_sanskrit_sahitya` (9 documents, 4,798 tokens)
  - Baseline CE: `3.9203` $\to$ Seed 42 CE: `3.9057` ($\Delta = \mathbf{-0.0146}$)
- **Finding:** The regression is strictly confined to pure Classical Sanskrit in Devanagari script (`wikipedia_sa_bhagavad_gita`). Hindi text discussing Sanskrit improved slightly.

---

## 3. Causal Assessment & Evidence Boundaries

### Directly Observed Evidence:
1. `wikipedia_sa_bhagavad_gita` exhibits a reproducible regression of $+0.0776$ CE.
2. The CPT training manifest (`experiments/batch2141_2180/cpt/cpt_train_manifest.json`) contained English geometry, English information theory, Hindi literature, Hindi chemistry, Python algorithms, and RFC 793, but **0 tokens of Classical Sanskrit**.
3. All Classical Sanskrit text was allocated 100% to the held-out validation and test partitions.

### Causal Attribution vs Hypothesis:
- **Observed Fact:** Zero training coverage coincided with a $+0.0776$ CE regression over 125 steps of CPT.
- **Scientific Boundary:** While zero training coverage is the most plausible explanation for representation drift on unreinforced token transitions, declaring it as the sole causal mechanism remains a working hypothesis until validated by a controlled intervention experiment.
- **Classification:** Properly termed **held-out source-family regression due to zero training coverage**, not "catastrophic forgetting".

---

## 4. Preregistered Intervention Protocol (Proposed Only — No Training Authorized)

When training is authorized in a future mission:
1. Ingest an independent Classical Sanskrit corpus (e.g. *Hitopadesha*, non-overlapping with *Bhagavad Gita*).
2. Stratify splits: 60% Train, 20% Val, 20% Test.
3. Blend ratio: 15% Sanskrit, 45% English math/tech, 40% Hindi literature.
4. Validation stop criteria: Sanskrit test CE $\le 4.3202$ while geometry and tech domains remain $\Delta < 0.0$.
