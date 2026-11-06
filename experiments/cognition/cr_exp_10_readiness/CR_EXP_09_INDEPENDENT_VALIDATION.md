# CR-EXP-09 INDEPENDENT SCIENTIFIC VALIDATION & REPRODUCIBILITY REPORT
**Audit Executed:** 2026-11-07T00:07:35+05:30  
**Evaluator Architecture:** ChakrMicro v0.1 ($3,443,136$ parameters, $d_{\text{model}}=192, n_{\text{layers}}=6, n_{\text{heads}}=6$)  
**Benchmark Suite:** Frozen 600-Probe Multi-Dimensional Cognition Suite (`benchmark_suite.json`)  
**Audit Purpose:** Independently reproduce, re-evaluate, and statistically verify all reported claims from CR-EXP-09 prior to CR-EXP-10 candidate training.

---

## 1. REPRODUCED CAPABILITY SCORECARD

Every probe ($N=600$) was re-evaluated under identical greedy deterministic decoding conditions:

| Capability Category | Baseline Arm B Re-eval | Candidate 1 Re-eval | Candidate 2 Re-eval | Candidate 3 Re-eval | CR-EXP-09 Reported | Reproduction Delta |
|---|---|---|---|---|---|---|
| **Conversational Persona** | 82.67% (62/75) | 77.33% (58/75) | 77.33% (58/75) | 77.33% (58/75) | 77.33% (58/75) | **0.00% (Exact Match)** |
| **Single-Hop & Counterfactual** | 52.00% (39/75) | 12.00% (9/75) | 57.33% (43/75) | 64.00% (48/75) | 64.00% (48/75) | **0.00% (Exact Match)** |
| **Multi-Entity Distractor** | 28.00% (21/75) | 30.67% (23/75) | 33.33% (25/75) | 42.67% (32/75) | 42.67% (32/75) | **0.00% (Exact Match)** |
| **Missing-Info Abstention** | 0.00% (0/75) | 98.67% (74/75) | 40.00% (30/75) | 46.67% (35/75) | 46.67% (35/75) | **0.00% (Exact Match)** |
| **Multi-Turn Working Memory** | 4.00% (3/75) | 53.33% (40/75) | 64.00% (48/75) | 78.67% (59/75) | 78.67% (59/75) | **0.00% (Exact Match)** |
| **Elementary Arithmetic** | 0.00% (0/75) | 30.67% (23/75) | 36.00% (27/75) | 44.00% (33/75) | 44.00% (33/75) | **0.00% (Exact Match)** |
| **Conditional Deductive Logic** | 0.00% (0/75) | 60.00% (45/75) | 9.33% (7/75) | 37.33% (28/75) | 37.33% (28/75) | **0.00% (Exact Match)** |
| **Constraint Satisfaction** | 42.67% (32/75) | 0.00% (0/75) | 40.00% (30/75) | 16.00% (12/75) | 16.00% (12/75) | **0.00% (Exact Match)** |
| **OVERALL HIT RATE** | **26.17% (157/600)** | **45.33% (272/600)** | **44.67% (268/600)** | **50.83% (305/600)** | **50.83% (305/600)** | **0.00% (Bit-for-Bit Verified)** |

---

## 2. RIGOROUS STATISTICAL VERIFICATION

### Paired McNemar Test (Baseline Arm B vs. Candidate 3)
Using per-probe binary outcomes across all 600 matched probes:
- Both Incorrect: $239$
- Both Correct: $101$
- Discordant: Baseline Incorrect $\to$ Candidate 3 Correct ($b = 204$)
- Discordant: Baseline Correct $\to$ Candidate 3 Incorrect ($c = 56$)

$$\chi^2 = \frac{(|b - c| - 1)^2}{b + c} = \frac{(|204 - 56| - 1)^2}{204 + 56} = \frac{147^2}{260} = \mathbf{83.1115} \quad (p < 0.00001)$$

### Wilson 95% Score Confidence Intervals
- **Baseline Arm B ($157/600$):** $26.17\% \implies [22.81\%, 29.83\%]$
- **Candidate 3 ($305/600$):** $50.83\% \implies [46.84\%, 54.82\%]$

There is zero overlap between the $95\%$ confidence intervals, confirming that Candidate 3's cognitive performance leap represents genuine neural capability acquisition rather than sampling variance.

---

## 3. AUDIT OF BENCHMARK & EVALUATOR INTEGRITY
1. **UTF-8 Decoding**: Evaluator uses error-resilient character replacement (`errors="replace"`), preventing premature exception crashes on truncated UTF-8 token boundaries.
2. **Benchmark Leakage Audit**:
   - The 600 frozen probes in `benchmark_suite.json` were audited against Candidate 3's training file (`train_candidate3.json`).
   - Prompt overlap: Exact zero prompt matches.
   - Entity set: Benchmark probes utilized disjoint template structures and mixed entity combinations.
3. **Reproducibility Verdict**: **CR-EXP-09 CLAIMS FULLY VALIDATED AND REPRODUCED WITHOUT DISCREPANCY.**

