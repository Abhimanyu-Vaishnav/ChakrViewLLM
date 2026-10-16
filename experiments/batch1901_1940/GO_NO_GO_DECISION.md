# CHAKRVIEW — Batches 1901–1940
## Formal Go/No-Go Decision Gate

### Mission Mandate

Determine the formal status of the Batch 1861–1900 language modeling and continued pretraining claims based on independent audit, forensic leakage investigation, and evaluation on an unshared source-disjoint benchmark.

Pre-declared decision options:
- **A. Independently replicated language-modeling improvement**
- **B. Improvement confirmed only on the existing corpus**
- **C. Inconclusive**
- **D. Previous claim invalidated or materially overstated**

---

### Formal Verdicts

#### 1. Candidate 1: 150-Step Balanced Continued Pretraining (Condition B)
**VERDICT: D. Previous claim invalidated or materially overstated**
*(Secondary Finding: B. Modest real improvement confirmed, but magnitude materially overstated)*

- **Rationale:**
  1. The reported gain of **-1.7750 cross-entropy** (from 5.7817 down to 4.0067) was claimed to represent robust generalization across held-out domains.
  2. Our audit discovered that **100% of the prose test split consisted of repeated boilerplate placeholder sentences**, and all remaining domains shared identical synthetic templates with the training set.
  3. When evaluated on the clean, source-disjoint independent benchmark, the true cross-entropy improvement was only **-0.2038** (from 5.9254 down to 5.7216).
  4. The claimed improvement was inflated by **~8.7x** due to evaluation corpus contamination and synthetic template memorization.
  5. While the model does exhibit a modest real learning signal (-0.2038 CE, 95% CI `[-0.3915, -0.2279]`), the primary claim that generalization was validated at the reported scale is **materially overstated**.

---

#### 2. Candidate 2: 300-Step Extended Pretraining (Condition D)
**VERDICT: D. Previous claim invalidated or materially overstated**
*(Secondary Finding: C. Inconclusive due to unpersisted artifact and single-seed execution in original sprint)*

- **Rationale:**
  1. Batch 1861 reported an extraordinary reduction in cross-entropy to **3.0989** (perplexity 22.17) after 300 steps, claiming evidence of continuous token-budget scaling.
  2. The checkpoint was not even serialized to disk in Batch 1861, existing solely as an unpersisted, single-seed in-memory run.
  3. Upon independent reproduction and evaluation on the source-disjoint benchmark, the 300-step model achieved a cross-entropy of **5.8420** (perplexity 344.46).
  4. This represents an **overfitting regression of +0.1178 cross-entropy** relative to the 150-step model, and recovers only **0.0834** CE compared to the canonical baseline.
  5. The claim that scaling to 300 steps improves held-out capability is invalidated; extended training on repetitive synthetic templates actively degrades external generalization.

---

### Comparison Summary Matrix

| Candidate | Claimed CE (Batch 1861) | Claimed Delta | Clean Independent CE (Batch 1901) | Clean Delta | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Canonical Baseline** | 5.7817 | Baseline | 5.9254 | Baseline | Baseline |
| **150-Step CPT (Condition B)** | 4.0067 | -1.7750 | 5.7216 | -0.2038 | **Decision D** (Overstated) |
| **300-Step CPT (Condition D)** | 3.0989 | -2.6828 | 5.8420 | -0.0834 | **Decision D** (Overstated & Regressed) |

---

### Non-Negotiable Gate Directives

1. **Do NOT Promote Either Checkpoint to Production:**
   Neither the 150-step nor the 300-step checkpoint shall be promoted to canonical status or integrated into the runtime execution engine.
2. **Decommission Synthetic Template Corpus Generation:**
   The stage-B procedural template generation pattern (`stage_b_derivations.txt`, `stage_b_algorithms.txt`, `stage_b_dialogues.txt`) that caused cross-split template leakage must be replaced with diverse, natural corpora.
3. **Preserve Checkpoint Integrity:**
   Canonical (`96357705...`) and Adapted (`15d48176...`) checkpoints remain permanently preserved and unchanged.
