# ChakrView Training Readiness Decision — Batches 2021–2060
**Milestone:** Batches 2021–2060  
**Branch:** `dev/v0.6-v0.7-validation`  
**Date:** October 2026  

---

## 1. Formal Readiness Decision

```
================================================================================
FINAL VERDICT: READY_FOR_CONTROLLED_CPT
================================================================================
```

ChakrView has satisfied all ten pre-requisite engineering, data integrity, and statistical gates required to conduct a controlled continued-pretraining experiment.

---

## 2. Gate-by-Gate Evaluation & Evidence

| Gate # | Gate Name | Target Requirement | Measured Status | Verdict |
|:---:|:---|:---|:---|:---:|
| **Gate 1** | **Source Provenance & Licensing** | Every source family must possess a verifiable URL, author/org identity, explicit license (Public Domain, CC BY-SA 4.0, Apache-2.0, or PSF), and content hash. | 13/13 source families fully cataloged in `SOURCE_PROVENANCE_MANIFEST.json` with legally documented permissive licenses. | **PASSED** |
| **Gate 2** | **Data Integrity & Ingestion** | Deterministic extraction, boilerplate removal, no synthetic filler or generated templates. | Clean extraction across all 1,258 documents. Zero template contamination or artificial backfilling. | **PASSED** |
| **Gate 3** | **Source-Family Leakage Isolation** | Strict source-family disjointness across Train, Validation, and Test partitions. | $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$. Zero overlapping 8-grams across partitions. | **PASSED** |
| **Gate 4** | **Evaluation Diversity** | Test benchmark must span multiple independent source families and domains without monopoly. | Test benchmark expanded to 4 distinct source families across English Math, Hindi Literature, Classical Sanskrit, and Information Theory. | **PASSED** |
| **Gate 5** | **Validation Sufficiency** | Validation set must possess adequate size and diversity to guide early stopping and checkpoint selection. | 186 documents, 32,617 tokens across conversational Hinglish, algorithm code, and Hindi mathematics. | **PASSED** |
| **Gate 6** | **Hindi / Hinglish / Sanskrit Labeling** | Accurate linguistic demarcation between Hindi prose, authentic Latin-script Hinglish, and classical Sanskrit text. | Distinct sources established: `wikipedia_hi_*` (Hindi), `google_hinglish_top_dialogues` (Hinglish), `wikipedia_sa_bhagavad_gita` (authentic Sanskrit). | **PASSED** |
| **Gate 7** | **Tokenizer Compatibility** | Lossless roundtrip decoding, verified fertility across scripts, zero modification to canonical `vocab_4096`. | 100.0% roundtrip fidelity across all scripts. Fertility measured (1.4–4.6 tokens/word). Canonical embedding table preserved. | **PASSED** |
| **Gate 8** | **Frozen Benchmark Integrity** | Historical 88-document benchmark preserved untouched; new frozen test split clearly versioned. | Historical benchmark preserved; `SPLIT_V2_MANIFEST.json` established as versioned evaluation manifest. | **PASSED** |
| **Gate 9** | **Compute Feasibility** | Training budget executable on standard CPU hardware within realistic time and memory limits. | 200 steps at batch size 8 (context 256) requires ~12–15 minutes on standard CPU ($\le 2.0$ GB RAM). | **PASSED** |
| **Gate 10** | **Baseline Reproducibility** | Canonical checkpoint verified bit-exact; direct comparison between 150-step and 300-step models statistically proven. | Canonical SHA-256 `96357705...` verified bit-exact. Regression of 300-step model over 150-step model proven with $p < 0.0001$. | **PASSED** |

---

## 3. Scope of Readiness & Boundaries

- **Eligible Scope:** Execution of the pre-registered controlled continued-pretraining plan (`CONTROLLED_CPT_EXPERIMENT_PLAN.md`) on Corpus v2 under CPU hardware using seeds 42, 1337, and 2026.
- **Excluded Scope:**
  - Runtime integration into `chakrview/runtime/session.py` (remains prohibited until post-training evaluation gates pass).
  - Production model replacement (canonical checkpoint remains the primary runtime core).
  - External pretrained model integration or API calling (prohibited by architectural invariants).

---

## 4. Remaining Operational Recommendations for Future Execution

1. **Deterministic Random Seeds:** When launching the training runner in the subsequent milestone, ensure seed initialization encompasses PyTorch, NumPy, and Python standard `random`.
2. **Periodic Checkpoint Archival:** Ensure all intermediate checkpoints (steps 25, 50, 75, 100, 125, 150, 175, 200) are saved with accompanying SHA-256 manifests.
3. **Automated Probe Evaluation:** Run the arithmetic, code syntax, and factual transfer probes immediately upon training completion to detect potential semantic regressions.
