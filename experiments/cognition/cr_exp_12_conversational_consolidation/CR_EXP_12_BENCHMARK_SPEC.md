# CR-EXP-12 FROZEN BENCHMARK SPECIFICATION
**Benchmark Identifier:** `CR-EXP-12-FROZEN-1500`  
**File Location:** `experiments/cognition/cr_exp_12_conversational_consolidation/benchmark_data/benchmark_1500_suite.json`  
**Manifest Location:** `experiments/cognition/cr_exp_12_conversational_consolidation/benchmark_data/benchmark_manifest.json`  
**Split Audit:** `experiments/cognition/cr_exp_12_conversational_consolidation/benchmark_data/benchmark_split_audit.json`  
**Cryptographic SHA-256:** `19565749c65dbe91552caa337997eaa4052e6ed9c46a8bfa8825e70cfd65184c`  
**Total Probes:** Exactly 1,500 probes  
**Generation Seed:** 202612  

---

## 1. PROBE DISTRIBUTION TABLE

| Capability Category | Target Count | Actual Probes | Evaluation Criterion |
|---|---|---|---|
| Natural Conversation | 200 | 200 | Fluency $>5$ chars, valid persona cue, no repetition |
| Single-Hop Entity Retrieval | 200 | 200 | Exact match of target value (location / profession) |
| Multi-Entity Distractor Disambiguation | 200 | 200 | Contains target value and strictly excludes distractors |
| Multi-Turn Working Memory | 300 | 300 | Contains latest updated value and strictly excludes earlier value |
| Counterfactual State Updates | 200 | 200 | Contains reassigned state and strictly excludes initial state |
| Answerability Calibration | 150 | 150 | Correct answer on answerable, correct abstention on unanswerable |
| Conditional Logical Reasoning | 100 | 100 | Correct Boolean deduction (Yes/No) |
| Elementary Arithmetic State | 75 | 75 | Exact integer calculation match |
| Constraint Satisfaction | 75 | 75 | Strict extraction of target class while rejecting prohibited items |
| **Total** | **1,500** | **1,500** | **Comprehensive Multi-Dimensional Benchmark** |

---

## 2. SPLIT AUDIT & ZERO-LEAKAGE GUARANTEE
- **Disjoint Entity Names**: The 40 entity names (`Aalok`, `Brijesh`, `Charu`, etc.) are completely disjoint from the training sets of CR-09, CR-10, CR-11, and CR-12.
- **Disjoint Vocabulary Targets**: Evaluates on holdout cities (`Shimla`, `Kozhikode`, etc.) and professions (`biologist`, `economist`, etc.).
- **Multiple Constraint Types**: Extends beyond fruit/color to animal, vehicle, and furniture constraints.
- **Frozen Status**: Cryptographic hash is locked. No training script has read access to this benchmark.

