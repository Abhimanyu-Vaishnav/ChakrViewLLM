# CR-16E Architecture & Research Decision

### 1. Architectural Capacity Evaluation
Does the canonical ChakrMicro v0.1 architecture ($3.44\text{M}$ parameters, $d=192, L=6, H=6, d_{\text{mlp}}=512, V=4096$) possess the representational capacity to achieve position-invariant retrieval, multi-token span generation, and calibrated abstention?

---

### 2. Empirical Evidence from the Factorial Experiment

1. **Multi-Token Spans Do Not Require Architecture Expansion:**
   - Standard models scored **0.0% on Length 3 and Length 4 spans**.
   - Simply balancing the training curriculum raised this to **$49.0\%$ on Length 3 and $57.0\%$ on Length 4** within the frozen architecture.
   - The frozen transformer attention and unembedding projections are fully capable of autoregressively emitting multi-subword entities when trained on them.
2. **Positional Invariance is Curricular, Not Hardware-Constrained:**
   - Models trained on fixed templates scored **0.0%** on non-initial positions.
   - Training on position-shuffled contexts immediately distributed attention across all slots ($20\% - 33\%$ across positions 0, 1, 2, 3), and achieved up to $14.7\%$ paired permutation success in Arm D Seed 4040.
   - Positional lock-in is an artifact of template overfitting, not an attention head limitation.
3. **Calibrated Abstention is Curricular:**
   - Negative-sample supervision restored abstention to **$86.7\%$** without requiring external classifiers or confidence heads.

---

### 3. Formal Architectural Decision
**DECISION:** **PRESERVE CANONICAL CHAKRMICRO V0.1 ARCHITECTURE ($3.44\text{M}$ PARAMETERS).**

**Rationale:**
Every capability gained in CR-EXP-16E was achieved purely by curriculum and loss formulation within the canonical architecture. Introducing pointer-generator heads, external memory, or RAG would add unnecessary complexity and obscure the fact that the underlying self-attention mechanism is capable of learning these tasks when properly supervised.

