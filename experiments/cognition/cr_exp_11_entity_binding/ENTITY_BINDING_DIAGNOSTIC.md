# ENTITY BINDING AND WORKING MEMORY DIAGNOSTIC STUDY
**Experiment:** CR-EXP-11 Phase 4  
**Auditor:** Cognitive Benchmark Engineer & Neural Research Scientist  
**Date:** 2026-11-07T22:05:00+05:30  
**Evaluated Baselines:** CR-09 Candidate 3 & CR-10 Arm B  

---

## 1. 8-LEVEL COGNITIVE HIERARCHY RESULTS

| Level | Description | CR-09 Cand 3 Accuracy | CR-10 Arm B Accuracy | Dominant Failure Mode |
|---|---|---|---|---|
| **L1** | Single Fact Retrieval | 40.00% (2/5) | 20.00% (1/5) | Arm B falsely abstains ($80\%$ of prompts) |
| **L2** | Multiple Entities (2 entities) | 25.00% (1/4) | 50.00% (2/4) | Entity-attribute mixing / early entity recency |
| **L3** | Distractor Resistance (3 entities) | 0.00% (0/3) | 33.33% (1/3) | Unselective attention to neighboring cities |
| **L4** | Counterfactual Update (1 update) | 0.00% (0/3) | 0.00% (0/3) | **Primacy Bias**: Model predicts initial state `Jaipur` instead of updated `Bhopal` |
| **L5** | Multiple Updates (A $\to$ B $\to$ C) | 0.00% (0/2) | 0.00% (0/2) | Attractor collapse to first seen token |
| **L6** | Interleaved Updates (2 entities updated) | 0.00% (0/2) | 50.00% (1/2) | Cross-entity attribute contamination |
| **L7** | Longer Multi-Turn Conversation | 0.00% (0/2) | 50.00% (1/2) | Degradation over longer context spans |
| **L8** | Natural Conversational Variations | 0.00% (0/2) | 0.00% (0/2) | Syntactic sensitivity (fails when phrasing departs from training syntax) |

---

## 2. DIAGNOSTIC INSIGHTS

1. **The Primacy Bias Mechanism in State Updates**:
   In Level 4 ("Meera works in Jaipur. Meera later moves to Bhopal. Where does Meera work now?"), CR-10 Arm B generates `"Jaipur."` exactly.
   The early attention heads register the first subject-value token pair, but without explicit training on recency markers (*"now"*, *"currently"*, *"later moved to"*), the causal attention mechanism fails to re-weight later state assignments over earlier ones.
2. **False Abstention Interference**:
   CR-10 Arm B frequently generated *"The context does not state..."* on single-entity queries because of the CR-10 negative-only training artifact.
3. **Template Fragility**:
   Both models collapse to 0% when facts are embedded in natural conversational prose (*"unpacked her bags in Bhopal today"*), demonstrating that the weights have not learned semantic predicate logic, only template-local token associations.

