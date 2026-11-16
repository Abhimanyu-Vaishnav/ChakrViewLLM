# CR-EXP-21: Evaluator Forensic Report

**Experiment ID:** CR-EXP-21  
**Target:** Evaluator Forensic Verification & Capability Integrity  
**Date:** 2026-11-15  
**Auditor:** Autonomous Independent Verification Agent  

---

## 1. Executive Summary

During prior missions (CR-20 and CR-20R), automated evaluators reported notable conversational progress, but several ambiguities remained regarding whether metrics reflected genuine capability vs artifactual surface matching:
1. **Loss vs Generation Quality:** Training loss dropped sharply ($2.7878$), but model output was prone to short syntax templates.
2. **Evaluator Leakage / Data Overlap:** In previous audit sets, benchmark prompts shared overlapping syntactic patterns with training sets, masking out-of-distribution failure.
3. **Metric Generosity on Entity Retrieval:** String-matching evaluators previously marked conversational framing (`"Your name is..."`) as success even when the predicted entity token was completely wrong or hallucinated.

CR-EXP-21 instituted strict evaluator forensic standards:
- **Zero Prompt Leakage:** 1,400 sealed benchmark probes constructed with verified 0-token prompt overlap against all training files.
- **Exact-Match & Semantic Disentanglement:** Entity binding is evaluated strictly on whether the exact multi-subword token sequence for the entity is recovered, distinct from the syntactic frame.
- **Cross-Lingual Script Checks:** Devanagari script checks explicitly verify unicode block ranges ($[\text{U}+0900..\text{U}+097\text{F}]$) rather than ASCII transliterations.

---

## 2. Forensic Analysis of Evaluators

### 2.1 Entity Binding Evaluator
- **Prior Flaw:** In CR-20, an evaluator counting "answerable" responses credited any response starting with `"Your name is"` as a successful multi-turn recall.
- **CR-21 Correction:** The `sealed_entity_binding_200` benchmark requires exact substring matching of the target multi-subword entity (e.g. `Ananya`, `Bhopal`, `Docker`) inside the model output.
- **Finding:** Under this rigorous check, both v0.10.1 and v0.11 achieve $0.0\%$ (0/200). While the model faithfully reproduces the syntactic prompt frame, it outputs phonetically frequent or generic tokens rather than the specific in-context entity.

### 2.2 Script Appropriateness Evaluator
- **Devanagari Probe:** Tests whether output text consists of valid Devanagari Unicode code points.
- **Result:** v0.10.1 and v0.11 both achieve $95.33\%$ (143/150). The remaining $\approx 4.7\%$ represent mixed code-switching or punctuation.
- **Finding:** The model does not suffer from Devanagari vocabulary collapse or catastrophic script confusion.

### 2.3 Persona & English Fluency Evaluator
- **English Persona Probe:** Tests self-identification and greeting completion.
- **Result:** Both v0.10.1 and v0.11 achieve $100.0\%$ (150/150). The model consistently outputs its canonical identity (`"I am ChakrView..."`) when asked `"Who are you?"`.

### 2.4 Instruction Following Evaluator
- **Probe Format:** Evaluates compliance with negative constraints (e.g. "Answer in one word", "Do not mention X").
- **Result:** v0.10.1 scored $81.0\%$; v0.11 scored $82.0\%$. Modest improvement due to training on diverse multi-turn instruction dialogs.

---

## 3. Conclusion

The evaluators used in CR-21 provide unvarnished, reproducible ground truth. The metrics do not inflate performance; they clearly demarcate what a 3.44M parameter model can do (structure, script, persona, syntax) from what it fundamentally cannot do without external memory or scaling (exact in-context multi-subword pointer retrieval).

