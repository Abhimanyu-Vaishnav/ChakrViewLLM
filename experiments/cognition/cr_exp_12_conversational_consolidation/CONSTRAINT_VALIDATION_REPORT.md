# CONSTRAINT SATISFACTION VALIDATION REPORT
**Investigation:** Phase 8 Diagnostic  
**Date:** 2026-11-07T22:34:15+05:30  

---

## 1. EMPIRICAL DECONSTRUCTION OF THE CR-11 "100%" CLAIM
- In CR-EXP-11, Candidate A achieved $50/50$ on constraint satisfaction.
- In our independent Phase 8 audit, we created a rigorous 150-probe multi-category validation test evaluating 5 distinct semantic domains:
  1. Fruits vs. non-fruits
  2. Animals vs. non-animals
  3. Colors vs. non-colors
  4. Vehicles vs. non-vehicles
  5. Furniture vs. non-furniture

### Results on 150 Unseen Probes:
- **Candidate A Accuracy**: **10.0% (15/150)**
- **Candidate B Accuracy**: **0.0% (0/150)**

### Conclusion
Candidate A only memorized a single specific vocabulary pairing (`"fruit only and strictly ignore other items"` with apple/banana/orange/mango/grape). When asked to `"Pick the animal only"`, `"Pick the vehicle only"`, or `"Pick the furniture only"`, the attention heads failed completely to attend to the constraint class.
**Finding**: The reported 100% was an artifact of narrow template vocabulary in CR-11, **NOT** generalizable semantic constraint satisfaction.

---

## 2. ACTIONS FOR CR-EXP-12
Candidate training in Phase 9 will incorporate a diversified constraint training stream (500 samples) spanning animals, colors, fruits, vehicles, and household objects to build true semantic category selection.

