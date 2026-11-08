# ENTITY BINDING AND DISTRACTOR AUDIT REPORT
**Investigation:** CR-EXP-13 Section 6  
**Auditor:** Autonomous Tokenization & Representation Researcher  
**Date:** 2026-11-08T10:06:50+05:30  

---

## 1. EMPIRICAL BREAKDOWN OF DISTRACTOR HANDLING

In the 1,500-probe benchmark, Multi-Entity Distractor Disambiguation queries test whether the model can extract the location of one specific person among three simultaneous entities:
`Context: N1 is in C1. N2 is in C2. N3 is in C3. Question: Where is N2 currently located? Answer:`

- **Parent (CR-11 Candidate A)**: $13.50\%$ (27/200)
- **CR-12 Candidate A**: $13.00\%$ (26/200)
- **CR-12 Candidate B**: **24.50% (49/200)**

### Inspection of Candidate B's Gains:
Candidate B almost doubled distractor performance because the training curriculum included randomized 3-entity distractor prompts where the completion was solely the target city (`"Patna."`) rather than a full sentence. This prevented the model from getting confused by generating a subject name that wasn't in the query.

---

## 2. THE DISTINCTION BETWEEN VALUE SELECTION AND DYNAMIC BINDING
1. **Value Selection (Partial Success)**: When asked directly in context (`Context: ... Question: ... Answer: `), the attention heads in Layer 4/5 are capable of finding the adjacent token following the target name about $24.5\%$ of the time.
2. **Dynamic Generation Binding (Failure)**: When forced to generate the subject name in the response (`Assistant: [Subject] is currently in [City]`), the causal generation unbinds and defaults to high-frequency training subjects ($0\%$ strict subject-value accuracy).
