# CHAKRVIEW — CAND_01 v1.3 CORRECTION & DECISION REGISTER
## Traceable Disposition of Findings ADV-01 Through ADV-07 & Architectural Decisions

**Document ID:** `REG-EXP-CAND01-COR-002`  
**Location:** `docs/experiments/CAND_01_V1.3_CORRECTION_AND_DECISION_REGISTER.md`  
**Candidate ID:** `CAND_01_CONTROLLED_ENTITY_COPYING`  
**Parent Governance Standard:** `docs/experiments/EXPERIMENT_GOVERNANCE_PROTOCOL_v1.0.md`  
**Status:** `CANONICAL_REGISTER`  
**Execution Authorization:** **`UNAPPROVED`**  
**Date:** 2026-10-31  

---

## 1. Register Purpose & Scope

This document provides a formal, traceable record of all issues, critique vectors, and vulnerabilities identified during the Independent Adversarial Verification of CAND_01 v1.2 (Findings ADV-01 through ADV-07). It details the exact scientific risks, the architectural and statistical resolutions implemented in the v1.3 suite, the specific document sections updated, and the remaining uncertainties.

---

## 2. Adversarial Findings Disposition Matrix (Findings ADV-01 through ADV-07)

### Finding ADV-01: TF-01 Combinatorial Starvation & Split Leakage
- **Audit Finding:** In v1.2, TF-01 required 924 examples from $\mathcal{E}_{\text{train}}$, but there were only 450 unique training tokens. Because direct copying prompts contain no keys, prompts had to be duplicated across Train, Val, and Test Recomb splits, violating the cryptographic prompt-disjointness invariant ($\text{SHA256}(\text{Prompt}_i) \neq \text{SHA256}(\text{Prompt}_j)$).
- **Scientific Risk:** Training leakage directly into validation and test splits, artificially inflating copy scores.
- **Resolution Implemented:**
  1. Excluded TF-01 from Partitions B and C (Validation and Test Recombination). Direct copying ($X \to X$) contains no relational bindings to recombine.
  2. Allocated exactly 350 examples to TF-01 in Partition A (Train) and 50 examples in Partition D (Disjoint-Token).
  3. Because $350 \le 450$ and $50 \le 150$, prompt-disjointness is **100% mathematically guaranteed** with zero duplication.
- **Documents & Sections Updated:**
  - `docs/experiments/CAND_01_SCIENTIFIC_DESIGN_v1.3_REVIEWED.md` §6.1
  - `docs/experiments/CAND_01_DATASET_SPECIFICATION_v1.3_REVIEWED.md` §3.1, §3.2
  - `docs/experiments/CAND_01_CONTROLLED_ENTITY_COPYING_REGISTRATION_v1.3_REVIEWED.md` §1
- **Remaining Uncertainty:** None. Combinatorial feasibility is proved by construction.
- **Status:** **`RESOLVED`**

---

### Finding ADV-02: Fabricated Logarithmic Frequency Cutoffs
- **Audit Finding:** v1.2 mandated $\log_{10} C(t) \ge 4.0$ ($C(t) \ge 10,000$) on a 465k token corpus after excluding stopwords. Non-stopword single tokens virtually never reach 10,000 occurrences in a 465k corpus; Tier 1 would be empty.
- **Scientific Risk:** Generator failure or arbitrary empty frequency tiers.
- **Resolution Implemented:**
  1. Discarded arbitrary logarithmic cutoffs.
  2. Specified an empirical percentile ranking protocol (top 33%, middle 33%, bottom 34% of filtered candidate tokens).
  3. Formally classified frequency tier boundaries as **`BLOCKED_PENDING_DATA`** until empirical histogram calculation is authorized and executed.
- **Documents & Sections Updated:**
  - `docs/experiments/CAND_01_DATASET_SPECIFICATION_v1.3_REVIEWED.md` §2.2
  - `docs/experiments/CAND_01_ADVERSARIAL_DESIGN_REVIEW_v1.3_REVIEWED.md` §2 (Inquiry 10)
- **Remaining Uncertainty:** The exact empirical count boundaries will be established when the generator extracts token counts from pretraining shards.
- **Status:** **`RESOLVED`**

---

### Finding ADV-03: Mathematically Impossible Seed-Level Permutation Test
- **Audit Finding:** v1.2 prescribed a paired permutation test across $n=3$ seeds demanding $p < 0.01$. The minimum possible two-sided p-value for $n=3$ is $p = 0.25$, making statistical significance mathematically impossible.
- **Scientific Risk:** Incoherent statistical testing protocol that could never reject the null hypothesis.
- **Resolution Implemented:**
  1. Formally removed the paired seed-level permutation NHST requirement.
  2. Acknowledged that $n=3$ seeds provide **descriptive uncertainty** only ($\text{mean} \pm \text{range}$, SEM).
  3. Confined inferential statistical testing to within-model evaluations using two-way cluster bootstraps over key and target token IDs.
- **Documents & Sections Updated:**
  - `docs/experiments/CAND_01_SCIENTIFIC_DESIGN_v1.3_REVIEWED.md` §7.1
  - `docs/experiments/CAND_01_EVALUATOR_SPECIFICATION_v1.3_REVIEWED.md` §6
  - `docs/experiments/CAND_01_ADVERSARIAL_DESIGN_REVIEW_v1.3_REVIEWED.md` §2 (Inquiry 6)
- **Remaining Uncertainty:** None. Statistical reporting is mathematically aligned with sample size.
- **Status:** **`RESOLVED`**

---

### Finding ADV-04: Unidentifiable Prompt Random Effect in GLMM
- **Audit Finding:** v1.2 specified a mixed-effects logistic regression with a random intercept for prompt instances ($v_{\text{prompt}(k)}$) where each prompt had only a single observation per arm, rendering the prompt variance unidentifiable.
- **Scientific Risk:** Convergence failure, singular fit errors, or software crashes during benchmark evaluation.
- **Resolution Implemented:**
  1. Removed $v_{\text{prompt}}$ from the GLMM formula.
  2. Retained crossed random intercepts strictly for repeated structural units: keys ($u_{\text{key}}$) and values ($w_{\text{val}}$):
     $$\text{logit}(P(Y_{ij} = 1)) = \beta_0 + \beta_{\text{arm}} + u_{\text{key}(i)} + w_{\text{val}(j)} + \epsilon_{ij}$$
- **Documents & Sections Updated:**
  - `docs/experiments/CAND_01_SCIENTIFIC_DESIGN_v1.3_REVIEWED.md` §7.2
  - `docs/experiments/CAND_01_EVALUATOR_SPECIFICATION_v1.3_REVIEWED.md` §6
- **Remaining Uncertainty:** GLMM estimation runtime on CPU across 500 prompts; fallback to two-way clustered bootstrap is pre-registered if quadrature fails.
- **Status:** **`RESOLVED`**

---

### Finding ADV-05: Confounded Factorial Comparison (Arm A vs. Arm C)
- **Audit Finding:** Natural language conversational framing in Arm C increases context length $4\times$, increases RoPE positional distance between key and query, and dilutes softmax attention mass. Calling this an isolated test of syntax framing was an overclaim.
- **Scientific Risk:** Attributing performance drops solely to syntax framing when length and positional attenuation contributed.
- **Resolution Implemented:**
  1. Formally classified Arm A vs. Arm C as a **composite packaging intervention** (syntax + length + distance + softmax dilution).
  2. Barred citing Arm A vs. Arm C as an isolated test of syntax framing.
- **Documents & Sections Updated:**
  - `docs/experiments/CAND_01_SCIENTIFIC_DESIGN_v1.3_REVIEWED.md` §5.1
  - `docs/experiments/CAND_01_CONTROLLED_ENTITY_COPYING_REGISTRATION_v1.3_REVIEWED.md` §3
  - `docs/experiments/CAND_01_ADVERSARIAL_DESIGN_REVIEW_v1.3_REVIEWED.md` §2 (Inquiry 8)
- **Remaining Uncertainty:** Isolating syntax independently of length is deferred to Track 4 (Format Robustness).
- **Status:** **`RESOLVED`**

---

### Finding ADV-06: Pre-Flight Calibration Scope & Gradient Norm Ambiguity
- **Audit Finding:** v1.2 specified norm $< 5.0$ without stating raw vs. clipped norm (if clipped, norm is trivially $\le 1.0$), and restricted duration to 20 steps (never reaching the step 50 warmup peak).
- **Scientific Risk:** Meaningless telemetry check that misses gradient explosion at peak learning rate.
- **Resolution Implemented:**
  1. Specified the metric explicitly as the **unclipped global gradient $L_2$ norm** prior to clipping.
  2. Extended the calibration check to **60 steps**, traversing the step 50 peak.
  3. Labeled all training hyperparameters strictly as `PROVISIONAL`.
- **Documents & Sections Updated:**
  - `docs/experiments/CAND_01_SCIENTIFIC_DESIGN_v1.3_REVIEWED.md` §8.1
  - `docs/experiments/CAND_01_IMPLEMENTATION_READINESS_AUDIT_v1.3_REVIEWED.md` §2, §3 (GAP-07)
  - `docs/experiments/CAND_01_ADVERSARIAL_DESIGN_REVIEW_v1.3_REVIEWED.md` §2 (Inquiry 9)
- **Remaining Uncertainty:** Numerical outcome of calibration cannot be known until authorized and run.
- **Status:** **`RESOLVED`**

---

### Finding ADV-07: Two-Way Clustered Uncertainty vs. Target-Only Resampling
- **Audit Finding:** Target-only cluster bootstrap underestimated variance caused by query key embedding degeneration.
- **Scientific Risk:** Overconfident confidence intervals failing to account for key token representation variance.
- **Resolution Implemented:**
  Mandated a **two-way cluster bootstrap** resampled simultaneously over unique key token IDs and unique target token IDs ($B = 2,000$ resamples).
- **Documents & Sections Updated:**
  - `docs/experiments/CAND_01_SCIENTIFIC_DESIGN_v1.3_REVIEWED.md` §7.2
  - `docs/experiments/CAND_01_EVALUATOR_SPECIFICATION_v1.3_REVIEWED.md` §6
  - `docs/experiments/CAND_01_ADVERSARIAL_DESIGN_REVIEW_v1.3_REVIEWED.md` §3
- **Remaining Uncertainty:** None. Two-way resampling protocol is locked.
- **Status:** **`RESOLVED`**

---

## 3. Register Summary & Pre-Execution Status

| Metric | Count |
|---|---|
| **Total Adversarial Findings Processed** | 7 (ADV-01 through ADV-07) |
| **Findings Classified as RESOLVED** | 7 (100.0%) |
| **New Governance Documents Created (v1.3)** | 7 |
| **Execution Authorization Status** | **`UNAPPROVED`** |
| **Candidate Readiness Status** | **`READY_FOR_PRE_EXECUTION_REVIEW`** |

*(This register is permanently preserved as part of the CAND_01 experiment audit lineage.)*

