# ARITHMETIC AND CONDITIONAL LOGIC REPORT
**Investigation:** Phase 7 Diagnostic  
**Date:** 2026-11-07T22:34:00+05:30  

---

## 1. EMPIRICAL AUDIT RESULTS
- **Candidate A Arithmetic**: $0.0\%$ (0/50 on frozen benchmark, complete output collapse).
- **Candidate A Conditional Logic**: $50.0\%$ (25/50 on frozen benchmark).
- **Candidate B Conditional Logic**: $10.0\%$ (severe regression during variable binding fine-tuning).

---

## 2. DUAL-STREAM RESTORATION STRATEGY
To restore arithmetic while advancing conditional deduction:
1. **Elementary Arithmetic**: Generate 400 controlled arithmetic word problems with small integer states ($1 \le N \le 30$) under addition and subtraction operations (`"Context: X had N items. Then X bought/gave M. Question: How many items does X have now?"`).
2. **Conditional Logic Regularization**: Generate 300 two-way Boolean deduction prompts (`"Premise: Whenever condition C holds, X does A. Otherwise, X does B."`) with equal distribution of true and negated premises to preserve logical deduction above $65\%$.

