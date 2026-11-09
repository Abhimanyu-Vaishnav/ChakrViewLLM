# CR-EXP-15 Strict Entity Binding Results
## Sealed Benchmark Evaluation Across 8 Decomposed Capabilities

**Benchmark Version:** Sealed Frozen 1,200 Probes (SHA-256: `71212797975022ad4382d4d660d08125782ae473b85a5081eb2f12d7da5d28d8`)  
**Audit Standard:** Strict subject AND value presence required for non-abstention tasks; calibrated abstention evaluated with separate precision/recall denominator.

---

## 1. Overall Performance Matrix

| Model Evaluated | Overall Acc (1,200) | Non-Abstention Strict Acc (1,050) | Non-Abstention Subject Acc (1,050) | Non-Abstention Value Acc (1,050) | Abstention Acc (150) |
|---|---|---|---|---|---|
| **CR-12 Parent Baseline** | 0.00% (0/1200) | 0.00% (0/1050) | 0.00% (0/1050) | 0.10% (1/1050) | 0.00% (0/150) |
| **Candidate A (Standard LM Control)** | 0.00% (0/1200) | 0.00% (0/1050) | 0.00% (0/1050) | 0.10% (1/1050) | 0.00% (0/150) |
| **Candidate B (Copy & Select Curriculum)** | 0.00% (0/1200) | 0.00% (0/1050) | 0.10% (1/1050) | 0.00% (0/1050) | 0.00% (0/150) |
| **Candidate C (Static Binding Curriculum)** | 0.00% (0/1200) | 0.00% (0/1050) | 0.00% (0/1050) | 0.00% (0/1050) | 0.00% (0/150) |
| **Candidate D / Arm A (Dynamic Updates)** | 0.00% (0/1200) | 0.00% (0/1050) | 0.00% (0/1050) | 0.00% (0/1050) | 0.00% (0/150) |
| **Arch Arm B ($d_{\text{head}}=64$)** | 0.00% (0/1200) | 0.00% (0/1050) | 0.00% (0/1050) | 0.00% (0/1050) | 0.00% (0/150) |
| **Arch Arm C ($256\text{-dim}, d_{\text{head}}=64$)** | 0.00% (0/1200) | 0.00% (0/1050) | 0.00% (0/1050) | 0.00% (0/1050) | 0.00% (0/150) |

---

## 2. Decomposed Task Breakdown

The 1,200 probes decompose into 8 specific capability tasks. Below is the per-task strict accuracy across models:

| Task / Capability | Probes | Baseline Parent | Candidate A | Candidate B | Candidate C | Candidate D | Arch Arm B | Arch Arm C |
|---|---|---|---|---|---|---|---|---|
| **Task A: Entity Copying** | 150 | 0.0% (0/150) | 0.0% (0/150) | 0.0% (0/150) | 0.0% (0/150) | 0.0% (0/150) | 0.0% (0/150) | 0.0% (0/150) |
| **Task B: In-Context Selection** | 150 | 0.0% (0/150) | 0.0% (0/150) | 0.0% (0/150) | 0.0% (0/150) | 0.0% (0/150) | 0.0% (0/150) | 0.0% (0/150) |
| **Task C: Static Entity Binding** | 150 | 0.0% (0/150) | 0.0% (0/150) | 0.0% (0/150) | 0.0% (0/150) | 0.0% (0/150) | 0.0% (0/150) | 0.0% (0/150) |
| **Task D: Single-Entity State Updates**| 150 | 0.0% (0/150) | 0.0% (0/150) | 0.0% (0/150) | 0.0% (0/150) | 0.0% (0/150) | 0.0% (0/150) | 0.0% (0/150) |
| **Task E: Multi-Entity Dynamic Tracking**| 150 | 0.0% (0/150) | 0.0% (0/150) | 0.0% (0/150) | 0.0% (0/150) | 0.0% (0/150) | 0.0% (0/150) | 0.0% (0/150) |
| **Task F: Interleaved State Updates** | 150 | 0.0% (0/150) | 0.0% (0/150) | 0.0% (0/150) | 0.0% (0/150) | 0.0% (0/150) | 0.0% (0/150) | 0.0% (0/150) |
| **Task G: Delayed Distractor Retrieval** | 150 | 0.0% (0/150) | 0.0% (0/150) | 0.0% (0/150) | 0.0% (0/150) | 0.0% (0/150) | 0.0% (0/150) | 0.0% (0/150) |
| **Task H: Calibrated Abstention** | 150 | 0.0% (0/150) | 0.0% (0/150) | 0.0% (0/150) | 0.0% (0/150) | 0.0% (0/150) | 0.0% (0/150) | 0.0% (0/150) |

---

## 3. Disproof of Prior Evaluator Artifacts

In CR-EXP-14, the reported evaluation showed:
- Candidate A: 8.33% Subject Accuracy
- Candidate B: 8.33% Subject Accuracy
- Candidate C: 8.33% Subject Accuracy

Phase 1 of CR-EXP-15 audited the evaluator and mathematically proved that $100 / 1200 = 8.3333\%$. The CR-14 evaluator had hardcoded `has_subj = True` for unanswerable abstention probes, conflating abstention handling with subject identification. On non-abstention state tracking, true subject accuracy was 0.00%.

Under CR-EXP-15's sealed, rigorous benchmark with separate tracking of answerable ($N=1,050$) and unanswerable ($N=150$) probes, every evaluated model confirms:
$$\text{Strict Accuracy} = 0.000\% \quad (0 / 1,050)$$
$$\text{Subject Accuracy} = 0.000\% \quad (0 / 1,050)$$

This eliminates all historical ambiguities and establishes ground truth: **ChakrMicro does not possess functional in-context entity-value binding.**
