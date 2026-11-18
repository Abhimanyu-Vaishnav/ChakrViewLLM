# UCIP-11 Phase E: Independent Benchmark Design

**Mission ID:** UCIP-11  
**Benchmark Directory:** `experiments/ucip11/benchmarks/`  
**Manifest Artifact:** `experiments/ucip11/benchmarks/manifest.json`  

---

## 1. Executive Summary

To measure genuine foundational language acquisition and generalization without circularity or memorization, an entirely new benchmark infrastructure was created.

**Guarantees:**
1. **Zero Contamination:** Completely disjoint from previous sealed suites (`benchmark_r2_sealed.json` and UCIP-10E) and disjoint from all Stage B and Exp34 training corpora.
2. **De-duplicated Deductive Logic:** Replaced the previous 10 repeated square/rectangle templates with balanced affirmative and negative syllogisms across diverse biological, physical, and geometric properties.
3. **Multilingual Balance:** Includes dedicated subsets for English technical explanations, Devanagari Hindi, and code-switched Hinglish.
4. **Compositional Generalization:** Evaluates multi-hop entity relations involving novel, unseen proper nouns and technical terms.
5. **Sealed Suite Cryptographic Freeze:**
   - Development Suite: `benchmarks/development/ucip11_dev.json` (100 probes) — SHA-256: `f2e6693938884ddac20922f8d51a38ad80331bd84d380364008fbdbb0b83017b`
   - Sealed Suite: `benchmarks/sealed/ucip11_sealed.json` (100 probes) — SHA-256: `dce6954ab6db4aa00bfcf1405a47243d4698e9e0cd084ad63ff3e5281b678fe4`

---

## 2. Benchmark Composition Matrix

| Evaluation Domain | Dev Probes | Sealed Probes | Total Probes | Evaluation Method |
|---|---|---|---|---|
| **English Technical & Science** | 50 | 50 | 100 | Semantic concept rubric (presence of core explanatory concepts without degenerate loops) |
| **Hindi Devanagari** | 15 | 20 | 35 | Devanagari keyword rubric & grammatical coherence |
| **Hinglish Code-Switched** | 15 | 20 | 35 | Bilingual marker & technical terminology check |
| **Instruction Following & Formatting**| 5 | 5 | 10 | Strict formatting compliance & negative constraint checking |
| **Compositional Generalization**| 5 | 2 | 7 | Exact novel entity extraction & multi-hop binding |
| **Arithmetic Diagnostic** | 5 | 4 | 9 | Exact numerical answer extraction |
| **Deductive Logic Diagnostic**| 4 | 4 | 8 | Balanced binary truth assertion (affirmative and negative) |
| **Verbatim Copying Diagnostic** | 1 | 1 | 2 | Exact character/token match |
| **Total** | **100** | **100** | **200** | Multi-dimensional evaluation |

---

## 3. Cryptographic Invariants

- Sealed Suite SHA-256: `dce6954ab6db4aa00bfcf1405a47243d4698e9e0cd084ad63ff3e5281b678fe4`
- Development Suite SHA-256: `f2e6693938884ddac20922f8d51a38ad80331bd84d380364008fbdbb0b83017b`
- Both suites are strictly frozen prior to running any training steps.
