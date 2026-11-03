# CHAKRVIEW — CAND_01 v1.2 CORRECTION & DECISION REGISTER
## Traceable Resolution of Independent Audit Findings & Architectural Decisions

**Document ID:** `REG-EXP-CAND01-COR-001`  
**Location:** `docs/experiments/CAND_01_V1.2_CORRECTION_AND_DECISION_REGISTER.md`  
**Candidate ID:** `CAND_01_CONTROLLED_ENTITY_COPYING`  
**Parent Governance Standard:** `docs/experiments/EXPERIMENT_GOVERNANCE_PROTOCOL_v1.0.md`  
**Status:** `LIVING_REGISTER`  
**Execution Authorization:** **`UNAPPROVED`**  
**Date:** 2026-10-31  

---

## 1. Register Purpose & Scope

This document provides a formal, traceable record of all issues, critique vectors, and vulnerabilities identified during the Independent Scientific Audit of CAND_01 v1.1. It details the exact scientific risks, the architectural and statistical resolutions implemented in the v1.2 suite, the specific document sections updated, and the remaining uncertainties.

---

## 2. Audit Findings Resolution Matrix (Findings F-01 through F-07)

### Finding F-01: Validation vs. Test Split Distributional Asymmetry
- **Original Issue:** In v1.1, the validation split ($N=500$) was drawn from training entity combinations, while the test split ($N=500$) was drawn from disjoint unseen entities. Early stopping on validation loss optimizes seen-token binding rather than OOD token generalization.
- **Scientific Risk:** High risk of false confidence during training and invalid generalization claims.
- **Resolution Implemented:** Replaced the 3-split design with an explicit **4-partition architecture** (Total: 5,500 records):
  - Partition A: Train (4,000 records, $\mathcal{E}_{\text{train}}$)
  - Partition B: Validation Recombination (500 records, $\mathcal{E}_{\text{train}}$, held-out tuples)
  - Partition C: Test Recombination (500 records, $\mathcal{E}_{\text{train}}$, held-out tuples) $\to A_{\text{test,recomb}}$
  - Partition D: Test Disjoint OOD (500 records, $\mathcal{E}_{\text{test}}$, disjoint tokens) $\to A_{\text{test,ood}}$
- **Documents & Sections Updated:**
  - `docs/experiments/CAND_01_SCIENTIFIC_DESIGN_v1.2_REVIEWED.md` §6
  - `docs/experiments/CAND_01_DATASET_SPECIFICATION_v1.2_REVIEWED.md` §3, §4
  - `docs/experiments/CAND_01_CONTROLLED_ENTITY_COPYING_REGISTRATION_v1.2_REVIEWED.md` §1
- **Remaining Uncertainty:** Generating 5,500 records requires verifying that $\mathcal{E}_{\text{train}}$ has sufficient combinatorial capacity for 5,000 unique prompts without repetition.
- **Verification Method:** Combinatorial cardinality verification in `scripts/generate_cand01_dataset.py`.
- **Status:** **`RESOLVED`**

---

### Finding F-02: Cluster Dependence Across Shared Token IDs
- **Original Issue:** $N=500$ evaluation instances are constructed from only 150 unique entity tokens across 8 task families. Prompts sharing the same key or value are not statistically independent.
- **Scientific Risk:** Standard binomial standard errors underestimate true parameter variance, producing artificially narrow confidence intervals.
- **Resolution Implemented:**
  1. Mandated reporting of **clustered standard errors** grouped by `target_val_token_id`.
  2. Mandated a $B = 2,000$ **cluster bootstrap** resampled at the token ID level.
  3. Formulated a hierarchical mixed-effects logistic regression model with random intercepts for token IDs.
- **Documents & Sections Updated:**
  - `docs/experiments/CAND_01_SCIENTIFIC_DESIGN_v1.2_REVIEWED.md` §7
  - `docs/experiments/CAND_01_EVALUATOR_SPECIFICATION_v1.2_REVIEWED.md` §6
  - `docs/experiments/CAND_01_ADVERSARIAL_DESIGN_REVIEW_v1.2_REVIEWED.md` §2 (Inquiry 9)
- **Remaining Uncertainty:** Cluster correlation magnitude ($\rho$) cannot be measured empirically until benchmark predictions are generated.
- **Verification Method:** Intraclass correlation coefficient (ICC) calculation on output residuals.
- **Status:** **`RESOLVED`**

---

### Finding F-03: Confounded Factorial Comparison in Arm D
- **Original Issue:** Arm D used conversational natural language framing, re-introducing multi-token subword entities, while Arms A, B, and C used single-token entities.
- **Scientific Risk:** Comparing Arm D to Arm C conflated natural language syntax with subword tokenization fragmentation.
- **Resolution Implemented:**
  1. Mandated that Arm C use the **exact same single-token entity pool** as Arms A and B, cleanly isolating syntax framing.
  2. Explicitly designated Arm D as an **exploratory historical reference arm**, permanently barring it from being cited as a clean causal comparator.
- **Documents & Sections Updated:**
  - `docs/experiments/CAND_01_SCIENTIFIC_DESIGN_v1.2_REVIEWED.md` §5.1
  - `docs/experiments/CAND_01_CONTROLLED_ENTITY_COPYING_REGISTRATION_v1.2_REVIEWED.md` §3
- **Remaining Uncertainty:** None. Epistemic boundaries are formally codified.
- **Verification Method:** Token count inspection on serialized Arm C JSONL inputs.
- **Status:** **`RESOLVED`**

---

### Finding F-04: Uncalibrated Hyperparameters Under Target-Only Loss
- **Original Issue:** Learning rate ($1.0 \times 10^{-4}$), 50-step warmup, and 500-step budget were carried forward from historical pretraining or unmasked SFT runs.
- **Scientific Risk:** Concentrating 100% of loss on target tokens increases per-token gradient magnitude by $>4\times$, risking gradient explosion or numerical instability.
- **Resolution Implemented:**
  1. Reclassified learning rate and step counts strictly as `PROVISIONAL`.
  2. Formulated a **pre-flight calibration protocol**: an isolated 20-step gradient telemetry run on a disposable model copy to measure gradient $L_2$ norm, clipping frequency, and logit drift before confirmatory launch.
- **Documents & Sections Updated:**
  - `docs/experiments/CAND_01_SCIENTIFIC_DESIGN_v1.2_REVIEWED.md` §8
  - `docs/experiments/CAND_01_CONTROLLED_ENTITY_COPYING_REGISTRATION_v1.2_REVIEWED.md` §2
  - `docs/experiments/CAND_01_IMPLEMENTATION_READINESS_AUDIT_v1.2_REVIEWED.md` §3 (GAP-07)
- **Remaining Uncertainty:** Calibration cannot be executed until authorized.
- **Verification Method:** Gradient norm logging script during authorized calibration phase.
- **Status:** **`RESOLVED`**

---

### Finding F-05: First-Token Exact Match Ambiguity vs. Full Completion
- **Original Issue:** Primary metric $A_{\text{FT}}$ checks only `generated_tokens[0] == target_token_id`. If the model emits the target token followed by 50 tokens of repetitive garbage before `<EOS>`, $A_{\text{FT}}$ scores it as 100% correct.
- **Scientific Risk:** A model suffering from repetition collapse would be represented as completely successful.
- **Resolution Implemented:**
  1. Retained $A_{\text{FT}}$ as primary target emission metric.
  2. Elevated Full Completion Match with clean termination ($A_{\text{FC}}$) to a mandatory co-primary promotion gate:
     $$A_{\text{FC}} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}\left(\hat{t}_{i,0} = y_i^* \land \hat{t}_{i,1} = \text{eos\_id} \land \text{len}(\hat{\mathbf{t}}_i) = 2\right)$$
  3. Pre-registered promotion mandates satisfying **both** $A_{\text{FT}} \ge 75.0\%$ **and** $A_{\text{FC}} \ge 70.0\%$.
- **Documents & Sections Updated:**
  - `docs/experiments/CAND_01_SCIENTIFIC_DESIGN_v1.2_REVIEWED.md` §3.2, §10
  - `docs/experiments/CAND_01_EVALUATOR_SPECIFICATION_v1.2_REVIEWED.md` §2, §3
  - `docs/experiments/CAND_01_CONTROLLED_ENTITY_COPYING_REGISTRATION_v1.2_REVIEWED.md` §1
- **Remaining Uncertainty:** Whether a 5.0% tolerance between $A_{\text{FT}}$ (75%) and $A_{\text{FC}}$ (70%) is appropriate requires empirical baseline calibration.
- **Verification Method:** Benchmark evaluation output audit on synthetic test splits.
- **Status:** **`RESOLVED`**

---

### Finding F-06: Equal-Count Frequency Tiers vs. Power-Law Vocabulary Distribution
- **Original Issue:** Candidate pool $\mathcal{E}$ was split into 3 equal-count tiers (33% each), ignoring the Zipfian distribution of token frequencies in natural corpora.
- **Scientific Risk:** High embedding magnitude variance within tiers could confound retrieval accuracy.
- **Resolution Implemented:**
  Redefined stratification boundaries based on **log-frequency cutoffs** derived from base Stage C pretraining validation corpus counts:
  - Tier 1 (High): $\log_{10} C(t) \ge 4.0$
  - Tier 2 (Medium): $2.5 \le \log_{10} C(t) < 4.0$
  - Tier 3 (Low): $\log_{10} C(t) < 2.5$
- **Documents & Sections Updated:**
  - `docs/experiments/CAND_01_DATASET_SPECIFICATION_v1.2_REVIEWED.md` §2.2
  - `docs/experiments/CAND_01_ADVERSARIAL_DESIGN_REVIEW_v1.2_REVIEWED.md` §2 (Inquiry 2)
- **Remaining Uncertainty:** Exact token frequency distribution in `vocab_4096` requires empirical histogram extraction during dataset generator implementation.
- **Verification Method:** Histogram verification in `generate_cand01_dataset.py`.
- **Status:** **`RESOLVED`**

---

### Finding F-07: Autoregressive Primacy Over Teacher Forcing
- **Original Issue:** Teacher-forced target rank was elevated to a primary success criterion, creating a risk that teacher-forced logit scores could override free generation failure.
- **Scientific Risk:** In M5, teacher-forcing showed 72% accuracy while free generation failed completely (0/200).
- **Resolution Implemented:**
  1. Demoted teacher-forced target rank and pre-softmax probability strictly to **secondary diagnostic telemetry**.
  2. Mandated that free autoregressive greedy generation ($T=0.0$) is the sole determinant of behavioral retrieval success.
- **Documents & Sections Updated:**
  - `docs/experiments/CAND_01_SCIENTIFIC_DESIGN_v1.2_REVIEWED.md` §3.2, §10
  - `docs/experiments/CAND_01_EVALUATOR_SPECIFICATION_v1.2_REVIEWED.md` §2, §3
- **Remaining Uncertainty:** None. Epistemic hierarchy is locked.
- **Verification Method:** Free generation assertion in evaluator unit tests.
- **Status:** **`RESOLVED`**

---

## 3. Register Summary & Pre-Execution Status

| Metric | Count |
|---|---|
| **Total Audit Findings Processed** | 7 |
| **Findings Classified as RESOLVED** | 7 (100.0%) |
| **New Governance Documents Created (v1.2)** | 7 |
| **Execution Authorization Status** | **`UNAPPROVED`** |
| **Candidate Readiness Status** | **`READY_FOR_PRE_EXECUTION_REVIEW`** |

*(This register is permanently preserved as part of the CAND_01 experiment audit lineage.)*

