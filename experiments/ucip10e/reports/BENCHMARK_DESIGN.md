# UCIP-10E Benchmark Design & Domain Specifications

**Date:** 2026-11-18  
**Mission:** UCIP-10E Intelligence Evaluation Infrastructure  

---

## 1. Domain Coverage & Statistical Design

The UCIP-10E benchmark suite covers 10 orthogonal intelligence capability domains across development (840 probes) and sealed (840 probes) test splits:

| Capability Domain | Subdomains Covered | Development Probes | Sealed Probes | Scoring Type |
|---|---|:---:|:---:|---|
| **1. Language Intelligence** | English, Devanagari Hindi, Hinglish | 120 | 120 | Semantic rubric & script consistency |
| **2. Mathematics** | 1D/2D addition, subtraction, carry, borrow, word problems | 150 | 150 | Deterministic numeric oracle |
| **3. Exact Copying** | Sentences, alphanumeric, punctuation | 100 | 100 | Verbatim vs normalized match |
| **4. Logical Reasoning** | Syllogisms, conditional logic, contradictions | 100 | 100 | Deductive answer extraction |
| **5. Generalization** | Unseen entities, novel objects, large operands | 100 | 100 | Out-of-distribution composition |
| **6. Multi-Turn Reasoning** | Entity tracking, context retention | 60 | 60 | Contextual state adherence |
| **7. Self-Correction** | Error verification cycle, counter-prompt revision | 60 | 60 | Two-turn correction accuracy |
| **8. Uncertainty & Calibration** | Epistemic abstention, unanswerable questions | 60 | 60 | Abstention vs false confidence |
| **9. Coding Intelligence** | Python functions, algorithms, bug fixes | 50 | 50 | Sandboxed subprocess test runner |
| **10. Self-Modeling** | Architecture awareness, identity, tool boundaries | 40 | 40 | Metadata accuracy verification |
| **TOTAL** | — | **840** | **840** | — |

---

## 2. Core 100 Sealed Benchmark Subset (`sealed_core100`)

To facilitate rapid regression auditing and multi-checkpoint comparison without sacrificing domain coverage, the infrastructure provides `benchmark_sealed_core100.json`, containing exactly **10 independent probes per domain** (100 total probes).

### Security & Provenance:
- All probes are tagged with unique IDs, explicit evaluation types, difficulty ratings, and provenance tags.
- Development and sealed suites were generated from distinct random seeds (`1001` vs `2002`).
- Checkpoint training code is isolated and has never seen the sealed probes.
