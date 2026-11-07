# ENTITY BINDING AND SUBJECT-VALUE ASSOCIATION REPORT
**Investigation:** Phase 4 Diagnostic  
**Date:** 2026-11-07T22:33:30+05:30  

---

## 1. EMPIRICAL MECHANISM OF ENTITY BINDING IN CHAKRMICRO

In the 6-layer ChakrMicro architecture ($d_{\text{model}}=192, 6\text{ heads}$), attention heads are shared across:
1. Positional syntax alignment (via RoPE).
2. Entity token recognition.
3. Attribute-value binding.

When training data presents varied subjects with uniform prompt templates (e.g., `"Where is [Name] now?"`), the attention heads in Layer 4 and Layer 5 learn to project the position of the question mark backwards. However, when the context contains multiple turns, the query tokens attend to the most recent city token but lose their binding to the subject token in the first turn.

---

## 2. THE THREE-STEP INTERVENTION FOR CR-EXP-12
1. **Explicit Subject Conditioning in Completions**:
   Instead of generating open unstructured text, completions must strictly adhere to:
   `[Subject] is currently in [Updated City].`
   This forces the autoregressive loss to penalize predictions where the subject is substituted.
2. **Variable Name Perturbation Across Context**:
   Train with 50 distinct subject names in rotation to break the attractor basins of `Aarav` and `Kabir`.
3. **Paired Multi-Attribute Anchors**:
   Include prompts where two entities exchange cities (`A moves to B's city, B moves to A's city`) to penalize indiscriminate recency matching.

