# Phase 1 — Benchmark Independence & Contamination Forensic Audit

**Date:** 2026-11-18  
**Audit Mission:** UCIP-10E-R1 Independent Benchmark, Evaluator & Reproducibility Validation  
**Datasets Audited:**
- `experiments/ucip10e/datasets/development/benchmark_dev.json` (840 Probes)
- `experiments/ucip10e/datasets/sealed/benchmark_sealed.json` (840 Probes)
- `experiments/ucip10e/datasets/sealed/benchmark_sealed_core100.json` (100 Probes)
**Training Corpora Audited for Contamination:**
- `experiments/cognition/ucip_08/data/ucip08_train.jsonl`
- `experiments/cognition/ucip_09/data/ucip09_train.jsonl`
- `experiments/cognition/ucip_09r/data/ucip09r_train.jsonl`
- `experiments/ucip09s/datasets/ucip09s_balanced_train.jsonl`

---

## 1. Executive Summary & Critical Findings

A forensic audit of the benchmark creation code (`build_benchmark_suites.py`) and dataset contents revealed **major contamination and redundancy issues** that challenge the claim of an independent 840-probe sealed suite:

1. **Massive Internal Probe Duplication (Artificial Inflation):**
   - In both the Development and Sealed suites (840 probes each), there are **only ~340 unique prompts!**
   - Development Suite: 840 total probes, but only **336 unique prompts** (**504 duplicate probes!**).
   - Sealed Suite: 840 total probes, but only **347 unique prompts** (**493 duplicate probes!**).
   - *Cause:* In `build_benchmark_suites.py`, small fixed lists of templates (e.g. 8 English prompts, 5 Hindi prompts, 10 copying sentences, 5 logic syllogisms, 4 coding tasks) were multiplied by loop multipliers (e.g. `eng_samples * 5`, `hin_samples * 8`, `sentences[i % len(sentences)]`).
   - Consequently, prompts like `"Hello ChakrView. How are you?"` and `"Who created you?"` are repeated 5 times within the exact same test suite under different probe IDs (`SEA_LANG_ENG_00`, `SEA_LANG_ENG_08`, etc.).
2. **Direct Training Data Contamination (Not Genuinely Held Out):**
   - The sealed suite contains **81 prompts that were directly present in the training datasets (`ucip09_train.jsonl` / `ucip09s_balanced_train.jsonl`)**!
   - Examples of direct training contamination:
     - `"Hello ChakrView. How are you?"` (Trained in UCIP-09)
     - `"Who created you?"` (Trained in UCIP-09)
     - `"What does a CPU do?"` (Trained in UCIP-09)
     - `"Explain the difference between RAM and storage."` (Trained in UCIP-09)
     - `"What is 1 + 2?"`, `"What is 5 + 9?"` (Trained in arithmetic subsets)
   - *Finding:* A benchmark cannot be considered genuinely "held out" or "sealed" when its language probes are verbatim copies of the SFT training set.
3. **Significant Overlap Between Development and Sealed Suites:**
   - There are **81 exact prompt overlaps** between `benchmark_dev.json` and `benchmark_sealed.json`.
   - The fixed templates for language, logic, and coding were reused identically across both splits; only the random number seeds for arithmetic and generalization differed.

---

## 2. Quantitative Contamination & Duplication Matrix

| Benchmark Suite | Total Probes | Unique Prompts | Duplicate Prompt Instances | Direct Training Contamination | Unique Held-Out Prompts |
|---|:---:|:---:|:---:|:---:|:---:|
| **Development Suite** | 840 | 336 (40.0%) | 504 (60.0%) | 76 probes | 260 |
| **Sealed Suite** | 840 | 347 (41.3%) | 493 (58.7%) | 81 probes | 266 |
| **Sealed Core100** | 100 | 72 (72.0%) | 28 (28.0%) | 14 probes | 58 |

---

## 3. Scientific Recommendation

The UCIP-10E 840-probe benchmark suite cannot be represented as 840 independent test samples:
- It represents **~266 genuinely unique held-out test scenarios** combined with repeated template instances and 81 training-contaminated prompts.
- **Action for R1:** We will evaluate the full 840-probe suite to maintain parity and verify reproducibility, but in our scientific analysis, we will explicitly separate:
  1. Performance on In-Distribution / Contaminated Prompts.
  2. Performance on Genuinely Unique Out-of-Distribution / Held-Out Prompts.
