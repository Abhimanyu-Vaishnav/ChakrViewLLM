# CR-EXP-15A Capability Scorecard
## Granular Capability Decomposition Across All Models and Conditions

**Mission Identifier:** CR-EXP-15A  
**Auditor:** Autonomous Scientific Auditor & Cognitive Benchmark Engineer  
**Benchmark Suite:** 1,200 Probes (8 Decomposed Cognitive Capabilities)  

---

## 1. Full Capability Decomposition Matrix (Numerator / Denominator / Percentage)

| Capability / Sub-Task | Baseline Parent | Candidate A (Standard LM) | Candidate B (Copy & Select) | Candidate C (Static Binding) | Candidate D (Dynamic Updates) | Arch Arm B ($d_{\text{head}}=64$) | Arch Arm C ($256\text{-dim}$) |
|---|---|---|---|---|---|---|---|
| **Literal Entity Copying** | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) |
| ↳ *Single-Token Entities* (Dev, etc.) | 0/8 (0.00%) | 0/8 (0.00%) | 0/8 (0.00%) | 0/8 (0.00%) | 0/8 (0.00%) | 0/8 (0.00%) | 0/8 (0.00%) |
| ↳ *Multi-Token Entities* | 0/142 (0.00%) | 0/142 (0.00%) | 0/142 (0.00%) | 0/142 (0.00%) | 0/142 (0.00%) | 0/142 (0.00%) | 0/142 (0.00%) |
| **In-Context Entity Selection** | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) |
| **Static Subject-Value Binding**| 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) |
| **Dynamic State Updates (Single)**| 0/150 (0.00%)| 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) |
| **Multi-Entity State Tracking** | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) |
| **Interleaved State Updates** | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) |
| **Delayed Distractor Retrieval**| 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) |
| **Calibrated Abstention** | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) |
| **Overall Strict Accuracy** | **0/1200 (0.00%)**| **0/1200 (0.00%)**| **0/1200 (0.00%)**| **0/1200 (0.00%)**| **0/1200 (0.00%)**| **0/1200 (0.00%)**| **0/1200 (0.00%)**|

---

## 2. Separate Answerable vs Abstention Metric Isolation

As mandated by scientific integrity guidelines:
- **Answerable Probes Denominator ($N=1,050$):**
  - Strict Accuracy: $0 / 1,050 \quad (0.00\%)$
  - Loose Accuracy: $0 / 1,050 \quad (0.00\%)$ across Candidate B, C, D, Arm B, Arm C; $1 / 1,050 \quad (0.10\%)$ on Parent and Candidate A.
  - Subject Match Rate: $1 / 1,050 \quad (0.10\%)$ on Candidate B; $0 / 1,050 \quad (0.00\%)$ on all other models.
  - False Abstention Rate: $0 / 1,050 \quad (0.00\%)$ across all models. (Models never spontaneously abstained on answerable questions).
- **Unanswerable Probes Denominator ($N=150$):**
  - Abstention Accuracy: $0 / 150 \quad (0.00\%)$ across all models.
  - Hallucination Rate: $150 / 150 \quad (100.00\%)$ across all models. (Models hallucinated an entity/location statement on every unanswerable query).

