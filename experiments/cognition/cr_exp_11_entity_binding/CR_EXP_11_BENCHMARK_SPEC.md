# CR-EXP-11 FROZEN BENCHMARK SPECIFICATION
**Benchmark Identifier:** `CR-EXP-11-FROZEN-1000`  
**File Location:** `experiments/cognition/cr_exp_11_entity_binding/benchmark_data/benchmark_1000_suite.json`  
**Manifest Location:** `experiments/cognition/cr_exp_11_entity_binding/benchmark_data/benchmark_manifest.json`  
**Cryptographic SHA-256:** `80f9c835ca09c1b8a1e62653538f607965454a7ef028b638d568ee324f0d418c`  
**Total Probes:** Exactly 1,000 probes  
**Generation Seed:** 42  

---

## 1. PROBE DISTRIBUTION TABLE

| Capability Category | Target Count | Actual Probes | Evaluation Criterion |
|---|---|---|---|
| Natural Conversational Understanding | 150 | 150 | Fluency $>5$ chars, valid persona cue, no repetition |
| Single-Hop Relational Retrieval | 150 | 150 | Exact match of target value (location / profession) |
| Multi-Entity Distractor Disambiguation | 150 | 150 | Contains target value and strictly excludes distractors |
| Multi-Turn Working Memory State Tracking | 200 | 200 | Contains latest updated value and strictly excludes earlier value |
| Counterfactual State Updates | 100 | 100 | Contains reassigned state and strictly excludes initial state |
| Missing-Information Epistemic Abstention | 100 | 100 | Contains valid epistemic abstention cue; non-abstention = hallucination |
| Conditional Logical Reasoning | 50 | 50 | Correct Boolean deduction (Yes/No) |
| Elementary Arithmetic State | 50 | 50 | Exact integer calculation match |
| Strict Constraint Satisfaction | 50 | 50 | Strict extraction of target class while rejecting prohibited items |
| **Total** | **1,000** | **1,000** | **Comprehensive Multi-Dimensional Benchmark** |

---

## 2. SPLIT AUDIT & INTEGRITY GUARANTEES
- **Disjoint Entity Sets**: The 40 entity names used in this benchmark are strictly held out and never appear in the training sets generated in Phase 7.
- **Disjoint Cities & Professions**: Benchmark cities (`Varanasi`, `Dehradun`, `Jabalpur`, etc.) and professions (`historian`, `electrician`, etc.) are completely isolated from the training corpus.
- **Template Diversity**: Syntactic templates are randomized across multi-turn, direct context, and dialogue styles.
- **Status**: The benchmark suite is frozen. No training or tuning may touch these prompts or targets.

