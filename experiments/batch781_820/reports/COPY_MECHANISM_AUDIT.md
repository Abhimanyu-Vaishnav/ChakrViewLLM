# Exact Argument Copying & Pointer Mechanism Audit: Batches 781–820
## Empirical Findings, Architecture Constraints & Copy Bottleneck Analysis

**Date:** 2026-10-15  
**Milestone:** Batches 781–820  
**Repository:** `ChakrView`  
**Location:** `experiments/batch781_820/reports/COPY_MECHANISM_AUDIT.md`

---

### 1. Executive Summary & Core Finding

In Batches 781–820, we evaluated whether pure autoregressive causal language modeling with grammar constraints can accurately copy numerical operands and textual strings from user prompts into the generated Plan DSL.

**Key Result:**  
- **Math Operand Copy Accuracy on Held-Out Validation Items:** **0.0% (0/34)**.
- **Text Phrase Copy Accuracy on Held-Out Validation Items:** **0.0% (0/8)**.
- **Overall Exact Argument Copy Accuracy:** **0.0% (0/42)**.

While Grammar-Constrained Decoding achieved **52.0% to 56.0% syntactic validity** (forcing the generation to adhere strictly to valid `STEP`, `tool`, `args={...}`, `deps=[...]`, and `<END_PLAN>` syntax), the actual emitted argument values were memorized training-set constants (e.g., emitting `args={expr=43+3}` when the prompt requested `Calculate 300 * 3`).

---

### 2. Forensic Breakdown: Why Pure Generative Softmax Fails on Disjoint Operands

1. **Disjoint Numeric Representation:**
   The training split contained arithmetic operations with numbers in range $[10, 259]$. The validation set evaluated numbers in ranges $[300, 509]$, and the frozen test set evaluates numbers in $[600, 811]$.
2. **Generative Softmax Bias:**
   In a causal autoregressive Transformer (`ChakrMicro`), the probability of emitting token $y_t$ given context $x$ is:
   $$P(y_t \mid y_{<t}, x) = \text{Softmax}(W_{\text{vocab}} h_t)$$
   Because $W_{\text{vocab}}$ has high dot-product alignment with high-frequency training digit sequences (such as `13`, `43`, `2`), the model samples these frequent tokens whenever predicting the value inside `expr=...`.
3. **No Direct Gradient Signal for Position-to-Position Copying:**
   Without a dedicated attention pointer or copy gate, the causal decoder must reconstruct the sequence from latent memory rather than directly pointing to the source span in $x$.

---

### 3. Pointer-Generator & Copy Architecture Feasibility Review

To achieve nonzero exact argument fidelity on completely held-out operands without violating ChakrView's from-scratch principles:
1. **Pointer-Generator Attention (See-et-al. Style):**
   A cross-attention mechanism computing:
   $$p_{\text{gen}} = \sigma(w_h^T h_t + w_s^T s_t + b_{\text{ptr}})$$
   $$P(w) = p_{\text{gen}} P_{\text{vocab}}(w) + (1 - p_{\text{gen}}) \sum_{i: w_i = w} a_i^t$$
   Where $a_i^t$ represents the attention weight over prompt token $i$.
2. **Integration Constraint:**
   `ChakrMicro` is currently a causal decoder-only architecture. Introducing a pointer-generator layer requires modifying the base attention blocks to expose cross-attention weights over prompt positions or adding a dual-head pointer projection (analogous to `ChakrSpanPointerModel` developed in Batch 581–620 for single-turn intent extraction).
3. **Engineering Verdict:**
   In accordance with the Work Package 803–807 instructions, because introducing architectural layer modifications to `ChakrMicro` during evaluation would alter the shared model core and invalidate the canonical checkpoint weights, we report this negative result honestly. A pointer-generator multi-task head should be formally designed as the central objective for Batches 821–860.
