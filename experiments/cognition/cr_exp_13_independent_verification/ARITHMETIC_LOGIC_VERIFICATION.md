# ARITHMETIC AND CONDITIONAL LOGIC VERIFICATION REPORT
**Investigation:** CR-EXP-13 Section 7  
**Auditor:** Autonomous Scientific Auditor  
**Date:** 2026-11-08T10:07:15+05:30  

---

## 1. ELEMENTARY ARITHMETIC AUDIT

In CR-EXP-11, Candidate A collapsed to **0.00%** on arithmetic.
In CR-EXP-12, an arithmetic stream was reintroduced.
- **CR-12 Candidate A**: **6.67% (5/75)**
- **CR-12 Candidate B**: **5.33% (4/75)**
- **Parent (CR-11 Candidate A)**: **1.33% (1/75)**

### Inspection of Arithmetic Predictions:
- Prompt: `"Context: Damini had 13 coins in the wallet. Later Damini spent 4 coins. Question: How many coins does Damini have now? Answer: "` (Target: `9`)
  - Candidate B generated: `"9."` (EXACT HIT).
- Prompt: `"Context: Indira had 18 coins in the wallet. Later Indira received 5 additional coins. Question: How many coins does Indira have now? Answer: "` (Target: `23`)
  - Candidate B generated: `"23."` (EXACT HIT).
- In the remaining 71 probes, outputs were either off-by-a-few numbers (`"14."` instead of `"16."`) or collapsed to training numbers (`"200."`).

### Conclusion
Arithmetic has technically rebounded from zero output collapse, but at $\sim 5\%-6\%$, it remains completely below the $60.0\%$ release gate.

---

## 2. CONDITIONAL LOGIC AUDIT
- **Parent (CR-11 Candidate A)**: **50.00% (50/100)**
- **CR-12 Candidate B**: **40.00% (40/100)**
- **CR-12 Candidate A**: **10.00% (10/100)**

Candidate A suffered severe regression in logic ($50\% \to 10\%$) because its curriculum over-indexed on multi-turn dialogue updates to the exclusion of premise deduction. Candidate B preserved $40.00\%$, but both models remain at or below chance level ($50\%$ on a binary Yes/No task).
