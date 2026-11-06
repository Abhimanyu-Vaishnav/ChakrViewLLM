# CR-EXP-10 BENCHMARK SPECIFICATION & CONTAMINATION AUDIT
**Date:** 2026-11-07T00:08:35+05:30  
**Status:** Frozen Immutable Evaluation Suites  

---

## 1. SUITE MANIFEST & CRYPTOGRAPHIC HASHES

| Suite Name | Target File | Probes | SHA-256 Digest |
|---|---|---|---|
| **Multi-Dimensional Cognition Suite (800 Probes)** | `benchmark_data/benchmark_800_suite.json` | 800 (100 per category) | `8d6f31523a69734c39e521bdad5366a6dd4813e9451a05b73b3d1d6d2179dd26` |
| **Conversational & Working Memory Suite (300 Probes)** | `benchmark_data/conversational_300_suite.json` | 300 (150 persona, 150 memory) | `3d1cf5e694e8e2bf5d8e8357345d93c518aba7dd168cc252d0f8e61bac57d509` |

---

## 2. DISJOINT ENTITY SET & LEAKAGE SAFEGUARDS
- **Entity Names**: Utilizes 20 strictly out-of-distribution Indian names (e.g., *Kavita, Manish, Divya, Suresh, Lakshmi, Gautam, Swati, Nikhil, Sunita, Deepak*) absent from the training split.
- **Locations**: Utilizes 16 distinct secondary cities (*Ahmedabad, Chandigarh, Bhopal, Indore, Surat, Nagpur, Visakhapatnam, Patna, Vadodara, Coimbatore, Kochi, Varanasi, Agra, Amritsar, Dehradun, Ranchi*).
- **Zero Answer-Key Exposure**: Answer templates are decoupled from training generators. Evaluators read answers strictly post-generation.

