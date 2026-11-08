# GENERALIZED CONSTRAINT SATISFACTION AUDIT
**Investigation:** CR-EXP-13 Section 8  
**Auditor:** Autonomous Cognitive Benchmark Engineer  
**Date:** 2026-11-08T10:07:35+05:30  

---

## 1. EMPIRICAL VALIDATION ACROSS MULTIPLE CLASSES

In CR-EXP-11, Candidate A claimed $100.0\%$ (50/50) on fruit/color constraints.
In CR-EXP-12's 1,500-probe benchmark, constraints were expanded across 5 semantic domains (fruits, colors, animals, vehicles, furniture).

### 75-Probe Benchmark Results:
- **Parent (CR-11 Candidate A)**: **36.00% (27/75)**
- **CR-12 Candidate B**: **34.67% (26/75)**
- **CR-12 Candidate A**: **10.67% (8/75)**
- **CR-10 Arm B**: **0.00% (0/75)**

### Analysis:
1. When presented with the familiar classes (fruits and colors), Candidate B and Parent scored $>80\%$.
2. When presented with vehicles (`truck`, `airplane`) or furniture (`desk`, `sofa`), accuracy dropped to $<15\%$.
3. The model exhibits category-specific semantic attraction rather than an abstract rule-following engine that can parse any arbitrary semantic class constraint.

**Conclusion**: The reported 100% in CR-11 was an artifact of narrow template vocabulary. Real generalized constraint satisfaction currently sits at $\sim 35\%$.
