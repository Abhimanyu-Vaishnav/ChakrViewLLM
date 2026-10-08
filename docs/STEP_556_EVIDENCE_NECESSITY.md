# STEP 556: EVIDENCE NECESSITY
**Wave Range**: 556  
**Focus**: Causal Necessity Ablation, Distinguishing Necessary vs Optional vs Irrelevant Evidence  
**Date**: 2026-10-08  

---

## 1. Executive Summary
Wave 556 establishes the causal evaluation protocol determining whether retrieved evidence is actually necessary for a given conclusion ([`EvidenceEvaluator.evaluate_necessity`](file:///d:/Project/ChakrView/chakrview/cognition/grounding/evidence_evaluation.py)).

### Distinguishable Roles
1. **`EVIDENCE_NECESSARY`**: Conclusion or confidence drops significantly upon removal ($\Delta \text{conf} \ge 0.40$).
2. **`EVIDENCE_OPTIONAL`**: Corroborating evidence where the conclusion remains reachable via redundant paths.
3. **`EVIDENCE_IRRELEVANT`**: Belongs to another entity or unrelated predicate; zero causal impact on decision.
4. **`EVIDENCE_CONTRADICTORY`**: Inverts polarity or negates conclusion; mandates qualification or rejection.

---

## 2. Empirical Verification
- **Test Case**: Query on Kepler-22b orbital period.
- **Ablation Protocol**:
  - Full Evidence: Answered "Kepler-22b is 290 days." (Confidence 0.95).
  - Ablated Evidence: Answered "I do not have enough evidence to answer this factually." (Confidence 0.0, Abstention True).
- **Result**: `conclusion_changed_on_removal = True`, $\Delta \text{conf} = 0.95$.
- **Classification**: **`EVIDENCE_NECESSARY`** (Verified genuine causal dependence).
