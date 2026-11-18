# Phase B — Clean Benchmark Reconstruction & Provenance Report

**Date:** 2026-11-18  
**Mission:** UCIP-10E-R2 Clean Benchmark Reconstruction, Evaluator Repair & Baseline Re-evaluation  
**New Datasets Created:**
- `experiments/ucip10e_r2/datasets/development/benchmark_r2_development.json` (460 Probes)
- `experiments/ucip10e_r2/datasets/sealed/benchmark_r2_sealed.json` (460 Probes)
- **Total Combined Benchmark Universe:** 920 Probes
**Cryptographic Manifest:** `experiments/ucip10e_r2/manifests/manifest_registry_r2.json`

---

## 1. Zero-Contamination Guarantees & Verification

Every prompt in the R2 benchmark suites was checked against the complete corpus of known training sets:
- `experiments/cognition/ucip_08/data/ucip08_train.jsonl`
- `experiments/cognition/ucip_09/data/ucip09_train.jsonl`
- `experiments/cognition/ucip_09r/data/ucip09r_train.jsonl`
- `experiments/ucip09s/datasets/ucip09s_balanced_train.jsonl`
- `experiments/ucip10e/datasets/sealed/benchmark_sealed.json` (Old contaminated suite)

### Provenance Audit Findings:
- **Training Contamination Count:** **0 probes (0.0% contamination)** in both Development and Sealed suites.
- **Development vs Sealed Overlap:** **0 probes (100% disjoint)**.
- All prompts are independently authored systems programming queries, novel multi-digit additions/subtractions, verified word problems, and epistemic queries.

---

## 2. Domain Distribution Across R2 Benchmark Suites

| Capability Domain | Subdomains | Dev Probes | Sealed Probes | Total Probes | Scoring Type |
|---|---|:---:|:---:|:---:|---|
| **1. Language Intelligence** | Systems English, Hindi, Hinglish | 75 | 75 | 150 | Semantic rubric + script verification + loop filter |
| **2. Mathematics** | Multi-digit addition, subtraction with borrow, word problems | 75 | 75 | 150 | Deterministic numeric oracle |
| **3. Exact Copying** | Long sentences, technical clauses | 50 | 50 | 100 | Exact verbatim vs normalized match |
| **4. Logical Reasoning** | Deductive syllogisms, conditionality | 50 | 50 | 100 | Deductive conclusion oracle |
| **5. Generalization (OOD)** | Novel vessels, fictional gadgets, large numbers | 60 | 60 | 120 | Combinatorial OOD oracle |
| **6. Multi-Turn Reasoning** | Entity and waypoint context retention | 40 | 40 | 80 | Contextual state adherence |
| **7. Self-Correction** | Error detection & independent revision | 30 | 30 | 60 | Independent recalculation oracle |
| **8. Uncertainty / Calibration** | Historical & physical unanswerable questions | 30 | 30 | 60 | Epistemic abstention detector |
| **9. Coding Intelligence** | Python functions & arithmetic algorithms | 30 | 30 | 60 | Sandboxed subprocess test runner |
| **10. Runtime Awareness** | Identity, CPU-first, local boundaries | 20 | 20 | 40 | Metadata & boundary verification |
| **TOTAL** | — | **460** | **460** | **920** | — |
