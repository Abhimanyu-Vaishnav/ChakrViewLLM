# CR-EXP-16B Architecture Decision

## 1. Architectural Boundaries Preserved
During CR-EXP-16B, all architectural constraints were strictly respected:
- Model: Canonical ChakrMicro ($3.44\text{M}$ parameters)
- Width: $d_{\text{model}} = 192$ (Unchanged)
- Layers: $n_{\text{layers}} = 6$ (Unchanged)
- Attention Heads: $n_{\text{heads}} = 6, d_{\text{head}} = 32$ (Unchanged)
- Hidden Dimension: $d_{\text{mlp}} = 512$ (Unchanged)
- Vocabulary: $4,096$ tokens (Unchanged)
- No external memory, no auxiliary pointer heads, no recurrent scratchpads added.

## 2. Capacity Assessment: Is ChakrMicro Sufficient?
1. **Sufficient for Bounded Algorithmic Copying:**
   ChakrMicro clearly possesses sufficient capacity for short sequence copying ($L=1\dots 3$: 79.1%), ordinary word extraction (91.8%), and distractor filtering (29.0%). The previous 0% failure in CR-16 was an objective defect, not an architectural limitation.
2. **Deficiencies at Horizon $\ge 7$ and Subword Re-composition:**
   The failure on long sequences ($L \ge 7$: 0%) and 5-piece subword names reflects the physical limits of 6 attention layers attempting to perform simultaneous in-context pointer routing and subword character composition.

## 3. Formal Decision
- **MAINTAIN CANONICAL ARCHITECTURE FOR CR-16 STAGE B/C:** Proceed with Stage B (Key-Value Retrieval) using the existing ChakrMicro model, incorporating the validated target-masking and multi-template design.
- **DEFER ARCHITECTURAL EXPANSION:** Do not widen or deepen the model until multi-step retrieval experiments establish whether retrieval is bottlenecked by attention routing or parameter capacity.

