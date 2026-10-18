# PREREGISTRATION FINAL REVIEW: AUDIT WITHOUT OVERCLAIMING
**Batch:** 2581–2620  
**Repository:** ChakrViewLLM (`dev/v0.6-v0.7-validation`)  
**Audited Documents:**
- `experiments/batch2541_2580/SANSKRIT_CURRICULUM_PREREGISTRATION_AMENDED.md`
- `experiments/batch2541_2580/SANSKRIT_CURRICULUM_PREREGISTRATION_AMENDED.json`
- `experiments/batch2541_2580/PREREGISTRATION_AMENDMENT_DIFF.md`

---

### 1. Verification of Preregistered Experimental Invariants

We conducted a line-by-line verification against the 11 Phase 4 audit requirements:

1. **Unconfounded Lineage:**
   - Arms 1, 2, and 3 are strictly anchored to canonical Step 10,000 checkpoint (`experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`, SHA-256 `96357705...`), eliminating the step-disparity confound.
2. **Role of Existing Seed 42 Candidate:**
   - Explicitly categorized as an external reference comparator (Arm 1) rather than an intervention candidate.
3. **Training Specifications:**
   - All training arms specify data provenance, token budget (500 steps, $146.7\text{k}$ tokens), effective batch size 8, AdamW optimizer (`lr=1.5e-4`, cosine decay to `1.5e-5`), seeds `[42, 43, 44]`, and early stopping rules.
4. **Leakage Screening Limitations:**
   - Explicitly acknowledged: Multi-tier 8-gram exact match and MinHash Jaccard thresholds (0.70) are **screening heuristics**, not mathematical proof of zero semantic leakage. Any flagged item requires human linguistic review.
5. **Held-Out Test Set Quarantine:**
   - The 290 held-out test documents are strictly isolated; they cannot be used for training, hyperparameter tuning, or checkpoint selection.
6. **Tokenizer Preflight Integrity:**
   - The preflight specification is declared prior to data inspection. Crucially: **No candidate corpus has yet been ingested or tokenized in this audit mission**. Therefore, the tokenizer preflight is declared as a **preregistered provisional specification**, not an empirical pass.
7. **Empirical Character/Token Threshold Rationale:**
   - The threshold of $\ge 1.80\text{ characters/token}$ is formally designated as a provisional acceptance criterion pending validation against the candidate corpus.
8. **Prespecified Primary Metric & Promotion Decision:**
   - Primary metric: Token-weighted Cross-Entropy on Classical Sanskrit (`wikipedia_sa_bhagavad_gita`).
   - Decision rule: Mean Arm 2 Sanskrit CE must beat Arm 1 by $\ge 1.96 \times \sigma$ while global CE $\le 4.2900$ and Hindi CE $\le 3.9250$.
9. **Statistical Plan for Multi-Seed Variance:**
   - Evaluates variance across seeds `[42, 43, 44]` with paired difference tests, guarding against single-seed noise.
10. **Early Stopping Guardrails:**
    - Explicitly halts training if Hindi regression $> +0.05\text{ CE}$ or overall validation CE increases $> +0.20\text{ CE}$.
11. **Mandatory Reporting of All Planned Runs:**
    - Protocol requires reporting all planned runs across seeds 42, 43, 44, including failed or aborted runs.

---

### 2. Disambiguation of Protocol States

To prevent overclaiming, we decouple four key protocol readiness states:

| Protocol State | Status | Rationale |
| :--- | :--- | :--- |
| **`STRUCTURE_COMPLETE`** | **`VERIFIED`** | All required sections, schemas, tables, and machine-readable JSON keys are present. |
| **`SCIENTIFIC_PROTOCOL_REVIEWABLE`** | **`VERIFIED`** | Checkpoint lineage unconfounded, controls matched, leakage screening tiered, multi-seed variance planned, and decision rules prespecified. |
| **`EMPIRICAL_PREFLIGHT_COMPLETED`** | **`UNVERIFIED`** | **Pending.** No candidate Sanskrit training corpus has been downloaded or evaluated in this batch per non-negotiable rules. |
| **`TRAINING_AUTHORIZED`** | **`UNVERIFIED`** | **Pending.** Training requires separate explicit human review and authorization after preflight passes. |
