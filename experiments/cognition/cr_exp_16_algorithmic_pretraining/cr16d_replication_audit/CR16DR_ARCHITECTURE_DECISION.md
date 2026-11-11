# CR-16DR Architecture & Research Decision

### 1. Decision Question
Should the canonical ChakrMicro v0.1 architecture ($3.44\text{M}$ parameters, $d=192, L=6, H=6, d_{\text{mlp}}=512, V=4096$) be modified with pointer-generator heads, external memory, or expanded width/depth to resolve remaining limitations?

---

### 2. Evidence-Based Assessment

1. **Self-Attention Capacity is Sufficient for Routing:**
   - The model achieved **100.0% accuracy on dynamic value reassignment** across 5 independent seeds.
   - It achieved **$96.9\%$ distractor filtering**, accurately ignoring irrelevant stored facts.
   - On G2 unseen-value probes, logit diagnostics confirmed that the attention mechanism **successfully routes to the value and predicts its first subword token at Rank 1 with $>86\%$ confidence**.
2. **The Failure Modes are Algorithmic & Curricular, Not Architectural:**
   - **G2 multi-token drift:** Caused by lack of multi-token span supervision.
   - **Position permutation failure:** Caused by training on fixed-offset templates rather than shuffled multi-fact contexts.
   - **Abstention drop:** Caused by absence of negative copying examples in the training mixture.
3. **Risks of Premature Architecture Redesign:**
   - Adding pointer-generator heads or external memory increases parameter complexity, disrupts standard autoregressive inference pipelines, and masks underlying training curriculum deficiencies.

---

### 3. Formal Decision
**DECISION:** **PRESERVE CANONICAL CHAKRMICRO ARCHITECTURE ($3.44\text{M}$ PARAMETERS).**

**Next Action:** Proceed directly to curricular refinements in CR-EXP-16E:
1. Multi-token span extraction pretraining.
2. Position-shuffled multi-fact training contexts to enforce content-based associative key matching.
3. Negative-sample regularization to restore calibrated abstention.

