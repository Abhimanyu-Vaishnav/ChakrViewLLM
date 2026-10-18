# SANSKRIT CURRICULUM EXPERIMENT PREREGISTRATION PROTOCOL

**Batch:** 2461–2500  
**Phase:** Phase 5 — Controlled Curriculum Experiment Preregistration  
**Document Status:** FROZEN PREREGISTRATION SPECIFICATION (NO MODEL TRAINING IN THIS BATCH)  
**Machine-Readable Spec:** [`experiments/batch2461_2500/SANSKRIT_CURRICULUM_PREREGISTRATION.json`](file:///d:/Project/ChakrView/experiments/batch2461_2500/SANSKRIT_CURRICULUM_PREREGISTRATION.json)  

---

## 1. Research Question & Primary Hypothesis

### Core Research Question:
Does adding a controlled, non-leaking curriculum of independent Classical Sanskrit training data to Continued Pretraining (CPT) reverse the observed +0.0776 CE (+8.07% PPL) held-out regression on Classical Sanskrit without causing negative transfer, perplexity degradation, or catastrophic forgetting in existing English, Hindi, or Code domains?

### Null Hypothesis ($H_0$):
Incorporating Classical Sanskrit into the CPT curriculum does not produce a statistically meaningful improvement on the held-out Sanskrit evaluation set ($\Delta_{\text{sa}} \ge 0.000$ CE relative to the baseline), or it causes an unacceptable regression ($\Delta \ge +0.050$ CE) in English or Hindi control domains.

### Alternative Hypothesis ($H_1$):
Incorporating Classical Sanskrit into the CPT curriculum reduces cross-entropy on the held-out Sanskrit benchmark ($\Delta_{\text{sa}} \le -0.050$ CE relative to Candidate Seed 42, with recovery towards or below baseline 4.3202 CE) while preserving overall test cross-entropy ($\Delta_{\text{global}} \le -0.150$ CE) and maintaining stable performance across English, Code, and Hindi controls ($\Delta \le +0.020$ CE on each control).

---

## 2. Experimental Arms & Controlled Architecture

To isolate the specific effect of Classical Sanskrit tokens from generic compute scaling or data volume effects, the experiment will evaluate four strictly controlled arms:

| Arm ID | Arm Name | Description | Compute / Steps | Data Allocation |
| :--- | :--- | :--- | :---: | :--- |
| **Arm 0** | **Canonical Baseline** | Frozen checkpoint at step 10,000 (`training_checkpoint_step_10000.pt`). | 0 steps (Reference) | Unmodified pretraining state. |
| **Arm 1** | **CPT Seed 42 Reference** | Existing CPT model trained on 6 sources (no Sanskrit). | 500 steps (Batch size 8, lr 2e-4) | 146,694 target tokens (en, hi, code). |
| **Arm 2** | **Sanskrit-Augmented CPT** | CPT trained on balanced mixture: 85% existing CPT corpus + 15% independent Classical Sanskrit. | 500 steps (Identical budget) | Identical total tokens trained; 15% Sanskrit token substitution. |
| **Arm 3** | **Volume-Matched Control** | CPT trained on existing CPT corpus + 15% additional Hindi/English text (no Sanskrit). | 500 steps (Identical budget) | Distinguishes Sanskrit-specific semantic transfer from general data expansion. |

### Controlled Invariants Across All Arms:
- **Base Architecture:** ChakrMicro (3,443,136 parameters, 6 layers, $d_{model}=192$, $n_{heads}=6$, tied embeddings, RoPE, SwiGLU).
- **Starting Weights:** Initialized strictly from `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`.
- **Optimizer & Schedule:** AdamW ($\beta_1=0.9, \beta_2=0.98, \epsilon=10^{-8}$, weight decay 0.01), cosine decay with linear warmup (50 warmup steps).
- **Seed Policy:** Pre-registered 3-seed evaluation (`seed=42, 43, 44`) to calculate empirical standard deviation across all metrics.
- **Hardware & Device Selection:** CPU execution with deterministic PyTorch operations (`torch.use_deterministic_algorithms(True)`).

---

## 3. Data Integrity & Leakage Prevention Protocol

### Strict Sourcing Requirements for Proposed Sanskrit Curriculum:
1. **Independent Provenance:** Data must originate from public domain Classical Sanskrit corpora completely disjoint from Wikipedia/Bhagavad Gita (e.g., Classical Sanskrit fables like *Hitopadesha* or *Panchatantra*).
2. **Leakage Gate (Mandatory Preflight):**
   - **Exact Match Check:** 0 identical 8-gram or longer sequences with the 61 documents in `wikipedia_sa_bhagavad_gita`.
   - **Jaccard / 13-gram MinHash Similarity:** Maximum n-gram Jaccard similarity $< 0.15$ against any test set document.
   - **Manifest Freezing:** Before training commences, the augmented manifest must be hashed and committed to Git with an immutable SHA-256 digest.
3. **No Substitution:** If compliant independent Sanskrit data cannot be obtained or fails the leakage gate, no training will take place.

---

## 4. Evaluation Protocol & Predefined Decision Boundaries

### Outcome Metrics (Derived Strictly from Raw Document Records):
1. **Primary Metric:** $\text{TW-CE}_{\text{sa}}$ on held-out `wikipedia_sa_bhagavad_gita` (19,577 target tokens).
2. **Secondary Metrics:**
   - Global test cross-entropy across all 290 documents (64,769 tokens).
   - English literature & math CE (`gutenberg_loney_coordinate_geometry`, `wikipedia_en_information_theory`).
   - Hindi control CE (`wikipedia_hi_sanskrit_sahitya`).
   - Document-level win rate against Canonical Baseline (%).
   - Token-weighted perplexity across all splits.

### Predefined Decision Rules:

| Outcome Category | Quantitative Criteria | Formal Project Decision |
| :--- | :--- | :--- |
| **SUCCESSFUL HYPOTHESIS CONFIRMATION** | - $\text{TW-CE}_{\text{sa}} \le 4.3400$ ($\Delta_{\text{sa}} \le -0.057$ vs Seed 42)<br>- Global $\text{TW-CE} \le 4.2900$<br>- Hindi control $\text{TW-CE} \le 3.9250$ (no regression)<br>- Win rate $\ge 70.0\%$ | **ACCEPT CURRICULUM AS VALIDATED**<br>Candidate eligible for release review. |
| **PARTIAL REPAIR WITH NEGATIVE TRANSFER** | - $\text{TW-CE}_{\text{sa}}$ improves ($\le 4.3600$)<br>- BUT Hindi or English degrades ($\Delta > +0.030$ CE) | **REVISE CURRICULUM WEIGHTING**<br>Reduce Sanskrit proportion and test regularization. |
| **INCONCLUSIVE / NO EFFECT** | - Sanskrit change within seed noise ($|\Delta_{\text{sa}}| < 0.020$ CE)<br>- Or seed variance exceeds effect size | **REJECT HYPOTHESIS**<br>Token absence was not the primary driver. |
| **CATASTROPHIC REGRESSION** | - Global $\text{TW-CE} > 4.4789$ (worse than baseline step 10,000) | **REJECT CANDIDATE IMMEDIATELY**<br>Retain model in research-only status. |

---

## 5. Preregistration Commitment

This protocol is frozen prior to dataset ingestion or execution. No parameters, thresholds, or decision gates may be altered retroactively. Training execution remains prohibited until explicit authorization in a subsequent sprint.
