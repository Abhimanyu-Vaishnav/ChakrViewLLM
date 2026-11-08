# BENCHMARK LEAKAGE AND EVALUATOR AUDIT REPORT
**Investigation:** CR-EXP-13 Section 9  
**Auditor:** Scientific Integrity & Evaluation Auditor  
**Date:** 2026-11-08T10:07:55+05:30  

---

## 1. DATASET OVERLAP & SPLIT AUDIT

We performed an exhaustive string and entity intersection check between the training datasets (`train_candidate_a.json`, `train_candidate_b.json`) and the frozen 1,500-probe benchmark (`benchmark_1500_suite.json`):

### A. Entity Names: **ZERO LEAKAGE CONFIRMED**
- Training sets used 40 names: `Aarav`, `Meera`, `Kabir`, `Neha`, etc.
- Benchmark used 40 strictly held-out names: `Aalok`, `Brijesh`, `Charu`, `Damini`, etc.
- Overlap: **$\emptyset$ (0 names)**. The entity split is $100\%$ clean.

### B. Locations and Professions: **CLEAN HELD-OUT SPLIT**
- Benchmark evaluated on distinct regional cities (`Shimla`, `Kozhikode`, `Guwahati`) and professions (`biologist`, `economist`, `dentist`).

### C. Constraint Prompt Overlap: **12 EXACT PROMPT MATCHES FOUND**
- Audit identified **12 exact prompt overlaps** between `train_candidate_b.json` and the benchmark's constraint section (e.g., `"Constraint: Pick the color only... Options: red, tiger. Answer: "`).
- **Severity**: Low (accounts for only $0.8\%$ of the 1,500 benchmark). Even if all 12 are deducted, Candidate B's score remains $500/1500 = 33.33\%$, which is still statistically significantly higher than the parent ($440/1500 = 29.33\%$).
- **Classification**: **Minor Template Leakage in Constraint Category**.

---

## 2. EVALUATOR AUDIT & NORMALIZATION VULNERABILITIES

1. **Substring Matching Vulnerability**:
   - In `multiturn_working_memory` and `single_hop_retrieval`, the evaluator checked if `target.lower() in gen.lower()`.
   - As discovered in Section 5, when Candidate A generated `"Ananya is currently in Amritsar.. Patna..."` on a probe where the subject was `Aalok`, the evaluator awarded a full hit because `"amritsar"` was present in the output.
   - This inflated the reported multi-turn working memory score ($43.0\%$ loose vs $0.0\%$ strict subject-value binding).
2. **Abstention Cue Matching**:
   - In `answerability_calibration`, the model generated `"The context does not state where that person works"` on both answerable and unanswerable queries. Because the evaluator checked `is_abstained` for unanswerable cases, it correctly awarded hits on unanswerable ($100\%$ recall), but correctly penalized answerable cases ($0\%$ accuracy).
