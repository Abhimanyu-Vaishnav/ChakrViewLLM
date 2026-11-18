# Phase E — Sealed Benchmark Integrity & Security Report

**Date:** 2026-11-18  
**Mission:** UCIP-10E-R2 Clean Benchmark Reconstruction, Evaluator Repair & Baseline Re-evaluation  
**Registry File:** `experiments/ucip10e_r2/manifests/manifest_registry_r2.json`

---

## 1. Cryptographic Manifest & Dataset Hashes

All R2 benchmark datasets are versioned under v2.0.0 and tracked with cryptographic SHA-256 hashes:

| Dataset Suite | File Path | Total Probes | SHA-256 Hash |
|---|---|:---:|---|
| **R2 Development Suite** | `experiments/ucip10e_r2/datasets/development/benchmark_r2_development.json` | 460 | `365db40c946e3e57f2aa2515ec23528b49704e9c704fa77317e13c8f8b8a5b28` |
| **R2 Sealed Suite** | `experiments/ucip10e_r2/datasets/sealed/benchmark_r2_sealed.json` | 460 | `3c873ddcf0c29f64921b6a3806fb7692c89fbfd251d7c35a6b0c201889fc665f` |

---

## 2. Integrity Controls Implemented

1. **Separation of Splits:** Development and Sealed splits share **0 identical prompts**.
2. **Immutable Results:** Raw evaluation files under `experiments/ucip10e_r2/results/` record exact timestamps, prompt strings, raw outputs, and failure reasons.
3. **No Target Leaks in Execution Logs:** Test prompts are delivered to model adapters without expected answers or rubrics embedded in prompt prefixes.
