# CR-EXP-08: DATASET INTEGRITY REPORT
## Comprehensive Split Discipline, Contamination & Leakage Analysis

- **Date**: 2026-11-06
- **Dataset Location**: `experiments/cognition/cr_exp_08_contextual_assignment/data/`
- **Total Samples Generated**:
  - `train.json`: 1,600 samples (960 relational [60.0%], 640 conversational [40.0%])
  - `val.json`: 200 samples (120 relational [60.0%], 80 conversational [40.0%])
  - `test.json`: 400 frozen samples across 7 distinct evaluation categories

---

### 1. Subject & Entity Disjointness Verification

To rigorously test generalization, subject entities are split into strictly disjoint sets:
- **Training Subjects (10)**: `Aman`, `Amit`, `Anjali`, `Neha`, `Pooja`, `Priya`, `Rahul`, `Ravi`, `Rohit`, `Sneha`
- **Validation Subjects (4)**: `Karan`, `Simran`, `Deepak`, `Kavita`
- **Strict OOD Test Subjects (8)**: `Aditya`, `Arjun`, `Meera`, `Ritu`, `Suresh`, `Tanya`, `Vikram`, `Zoya`

**Cross-Split Contamination Audit**:
- $\text{Train Subjects} \cap \text{Val Subjects} = \emptyset$ (0 overlap)
- $\text{Train Subjects} \cap \text{OOD Test Subjects} = \emptyset$ (0 overlap)
- $\text{Val Subjects} \cap \text{OOD Test Subjects} = \emptyset$ (0 overlap)
- **Status**: **STRICTLY ZERO SUBJECT LEAKAGE**.

---

### 2. Category-by-Category Test Split Leakage Analysis

The 400-sample test suite was audited for exact prompt leakage into `train.json`:

| Test Category | Sample Count | Prompts in `train.json` | Leakage Rate | Evaluation Role |
|---|---|---|---|---|
| **Familiar Relational Retrieval** | 60 | 43 | 71.7% | **In-Distribution Memorization & Familiar Pattern Benchmark** (Expected overlap on familiar subjects) |
| **OOD Entity Retrieval** | 60 | 0 | **0.0%** | **Strict Out-of-Distribution Generalization** (Completely unseen subjects) |
| **OOD Template Generalization** | 60 | 0 | **0.0%** | **Strict Phrasing Generalization** (Unseen dossier/datasheet templates) |
| **Counterfactual Reassignment** | 60 | 0 | **0.0%** | **Dynamic Contextual Override** (Entities moved from earlier to later cities) |
| **Distractor Resistance** | 60 | 0 | **0.0%** | **Multi-Entity Disambiguation** (Contexts containing 2 or 3 entity mappings) |
| **Missing-Information Abstention**| 50 | 0 | **0.0%** | **Epistemic Abstention** (Queries about unmentioned entities) |
| **Conversational Preservation** | 50 | 50 | 100.0% | **Persona & Dialogue Retention** (Core conversational safety check) |

**Conclusion on Split Discipline**:
- 290 out of 400 test samples (all 5 core cognitive generalization categories) have **0.0% prompt leakage into training data**.
- In-distribution familiar retrieval and conversational preservation intentionally evaluate training distribution retention.

---

### 3. Arm C Negative Candidate Construction & Length Matching

For all 960 relational training samples, negative candidate pairs were generated in compliance with `negative_candidate_construction.md`:
- Every negative candidate is a semantically valid Indian city.
- The negative candidate is strictly incorrect for the given subject mapping.
- **Exact BPE Length Matching**:
  - Positive 3-token cities $\to$ Negative chosen strictly from 3-token pool (`Delhi`, `Jaipur`, `Pune`, `Surat`, `Kanpur`, `Nagpur`, `Indore`, `Patna`).
  - Positive 4-token cities $\to$ Negative chosen strictly from 4-token pool (`Mumbai`, `Bengaluru`, `Bhopal`, `Nashik`).
  - Positive 5-token cities $\to$ Negative chosen strictly from 5-token pool (`Chennai`, `Kolkata`, `Hyderabad`, `Lucknow`).
- **Audit Verification**:
  $$\text{Length Mismatch Count} = 0 \quad (100.0\% \text{ length-matched})$$
  $$\text{Missing Negative Candidate Count} = 0$$

All dataset files have been serialized and frozen.

