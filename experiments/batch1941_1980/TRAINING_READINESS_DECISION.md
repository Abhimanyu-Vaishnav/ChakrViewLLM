# CHAKRVIEW — Batches 1941–1980
## Training Readiness Decision Gate

### Mission Mandate

Establish whether ChakrView is scientifically, statistically, and methodologically ready to resume controlled continued-pretraining experiments based on:
1. Corrected and reconciled statistical confidence intervals.
2. Verified checkpoint provenance.
3. Verification of candidate natural corpora.
4. Elimination of synthetic template generators and boilerplate leakage.
5. Strict source-level split isolation.

---

### Formal Readiness Verdict

**DECISION: CONDITIONAL TRAINING READINESS (CORPUS STAGED & FROZEN)**

ChakrView's evaluation framework and experimental discipline have successfully earned statistical trustworthiness. However, before launching next-generation continued pretraining, the newly created natural corpus must remain **frozen as a staged candidate dataset** while the training pipeline is configured for source-group validation.

---

### 1. Prerequisite Readiness Checklist

| Readiness Criterion | Target Standard | Status | Evidence |
| :--- | :--- | :--- | :--- |
| **Statistical Integrity** | Point estimates must lie inside their confidence intervals; estimands clearly separated. | **PASS** | Evaluated in [STATISTICAL_AUDIT.md](file:///d:/Project/ChakrView/experiments/batch1941_1980/STATISTICAL_AUDIT.md) & [CORRECTED_CONFIDENCE_INTERVALS.json](file:///d:/Project/ChakrView/experiments/batch1941_1980/CORRECTED_CONFIDENCE_INTERVALS.json). |
| **Checkpoint Lineage** | Canonical and adapted checkpoints immutable; 150s and 300s persisted. | **PASS** | Hashes verified bit-exact against baseline manifests. |
| **Decommissioning of Procedural Templates** | No procedural `while len(dom_chunks) < 22` loops; no synthetic template repetition. | **PASS** | Decommissioned; zero synthetic generation in natural corpus. |
| **Source Provenance & Permissible Usage** | Documented origins, public domain or open licenses, file hashes recorded. | **PASS** | Cataloged in [NATURAL_CORPUS_PROVENANCE.md](file:///d:/Project/ChakrView/experiments/batch1941_1980/NATURAL_CORPUS_PROVENANCE.md). |
| **Strict Split Isolation** | No shared source files across train/val/test splits. | **PASS** | Enforced in [SPLIT_MANIFEST.json](file:///d:/Project/ChakrView/experiments/batch1941_1980/data/SPLIT_MANIFEST.json). |
| **Token Scale** | Meaningful target token volume for pretraining (>100k tokens in train). | **PASS** | 115,913 train tokens, 334,092 test tokens across 2,619 deduplicated documents. |

---

### 2. Candidate Checkpoint Status

1. **150-Step CPT Checkpoints (`cpt_balanced_seed_42`, `123`, `999`):**
   - **Status:** **RETAINED AS EXPERIMENTAL CANDIDATES ONLY.**
   - *Directives:* May serve as comparative baselines in future experiments. Under no circumstances should they be promoted to canonical status or integrated into the production runtime, as their transfer accuracy remains 7.69% (failing arithmetic and reasoning).
2. **300-Step CPT Checkpoint (`cpt_balanced_seed_42_300steps`):**
   - **Status:** **DEPRECATED (FAILED GENERALIZATION GATE).**
   - *Directives:* Permanently archived as an experimental artifact demonstrating template overfitting. Re-training on synthetic templates beyond 150 steps caused confirmed regression on independent benchmarks.

---

### 3. Immediate Next Steps for Subsequent Batches

1. **Keep Natural Text Corpus v1 Frozen:**
   Do not modify [CORPUS_MANIFEST.json](file:///d:/Project/ChakrView/experiments/batch1941_1980/data/CORPUS_MANIFEST.json) until pretraining hyperparameters (learning rate, context length, batch size) are formally audited.
2. **Configure Multi-Epoch Learning on Genuine Natural Prose:**
   Formulate a continued-pretraining schedule utilizing the 115k natural tokens (Tagore, RFC 793, CPython, Premchand) with validation monitoring against the source-disjoint validation set (`bisect.py`, Ganit).
3. **Maintain Strict Separation of Neural and Deterministic Authority:**
   Continue strict architectural enforcement that neural model predictions do not bypass `SafeCalculator` or `GovernedToolRouter`.
