# CR-EXP-15A Raw Prediction Independent Rescoring
## Systematic Rescoring of 8,400 Model Generations Using the Calibrated Evaluator

**Mission Identifier:** CR-EXP-15A  
**Auditor:** Autonomous Scientific Auditor & Benchmark Engineer  
**Date:** November 2026  
**Evaluation Scope:** 1,200 probes $\times$ 7 models = 8,400 raw generated strings.  

---

## 1. Independent Rescoring Summary Table

Evaluated against the sealed 1,200-probe benchmark (`benchmark_1200_sealed.json`):

| Model Evaluated | Total Probes | Answerable Strict Acc ($N=1,050$) | Answerable Loose Acc ($N=1,050$) | Subject Match Rate ($N=1,050$) | Target Value Match Rate ($N=1,050$) | Calibrated Abstention Acc ($N=150$) | False Abstention Rate ($N=1,050$) |
|---|---|---|---|---|---|---|---|
| **Parent Baseline** | 1,200 | **0.00%** (0/1050) | **0.10%** (1/1050) | 0.00% (0/1050) | 0.10% (1/1050) | **0.00%** (0/150) | 0.00% (0/1050) |
| **Candidate A (Standard LM)** | 1,200 | **0.00%** (0/1050) | **0.10%** (1/1050) | 0.00% (0/1050) | 0.10% (1/1050) | **0.00%** (0/150) | 0.00% (0/1050) |
| **Candidate B (Copy & Select)** | 1,200 | **0.00%** (0/1050) | **0.00%** (0/1050) | 0.10% (1/1050) | 0.00% (0/1050) | **0.00%** (0/150) | 0.00% (0/1050) |
| **Candidate C (Static Binding)**| 1,200 | **0.00%** (0/1050) | **0.00%** (0/1050) | 0.00% (0/1050) | 0.00% (0/1050) | **0.00%** (0/150) | 0.00% (0/1050) |
| **Candidate D (Dynamic Updates)**| 1,200 | **0.00%** (0/1050) | **0.00%** (0/1050) | 0.00% (0/1050) | 0.00% (0/1050) | **0.00%** (0/150) | 0.00% (0/1050) |
| **Arch Arm B ($d_{\text{head}}=64$)** | 1,200 | **0.00%** (0/1050) | **0.00%** (0/1050) | 0.00% (0/1050) | 0.00% (0/1050) | **0.00%** (0/150) | 0.00% (0/1050) |
| **Arch Arm C ($256\text{-dim}, d_{\text{head}}=64$)** | 1,200 | **0.00%** (0/1050) | **0.00%** (0/1050) | 0.00% (0/1050) | 0.00% (0/1050) | **0.00%** (0/150) | 0.00% (0/1050) |

---

## 2. In-Depth Analysis of Rescoring Output

1. **Why Loose Accuracy Remains at 0% / 0.1%:**
   - Under loose evaluation, a model gets credit if it outputs the expected location anywhere in its response, regardless of whether the subject is correct.
   - Even under loose scoring, accuracy is virtually 0%:
     - Parent Baseline & Candidate A: matched `Udaipur` exactly once by chance on probe `CR15_DRDIST_0964` (`You is currently in Udaipur.`), yielding $1/1050 = 0.095\%$.
     - Candidate B, C, D, Arm B, Arm C: matched $0/1050$ target values.
2. **Why Subject Match Rate Remains at 0% / 0.1%:**
   - Candidate B output `Tarun is currently in Chine.` on `CR15_STAT_0370` where the subject happened to be `Tarun` ($1/1050 = 0.095\%$).
   - All other models matched $0/1050$ subjects.
3. **No Hidden Pass Rates Obscured by String Matching:**
   - Word-boundary regex matching confirms that models did not generate the correct names or cities in alternative punctuation or phrasing.
   - The zero score is an accurate, incontrovertible reflection of the models' actual text outputs.

