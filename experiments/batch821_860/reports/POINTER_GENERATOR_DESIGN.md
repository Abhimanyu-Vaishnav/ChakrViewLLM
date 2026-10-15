# Neural Pointer-Generator Architecture Design: Batches 821–860
## Dual-Distribution Emission, Attention Pointing & Copy Gate

**Date:** 2026-10-15  
**Milestone:** Batches 821–860  
**Repository:** `ChakrView`  
**Location:** `experiments/batch821_860/reports/POINTER_GENERATOR_DESIGN.md`

---

### 1. Architectural Motivation

In Batches 781–820, unconstrained and grammar-constrained autoregressive generation produced 0.0% argument copy accuracy on disjoint held-out operands. Pure generative softmax over a 4,096-token vocabulary was forced to hallucinate memorized training constants (such as `expr=43+3`) whenever encountering novel numbers in user prompts.

To solve argument grounding without injecting external models or cheating with hardcoded regex rules, we designed and implemented `ChakrPointerMicro` (`chakrview/brain/pointer_generator.py`).

---

### 2. Dual-Distribution Formulation

For each generation step $t$, the model produces two distinct probability distributions:

1. **Vocabulary Distribution ($P_{\text{vocab}}$):**
   Standard language model projection over the token vocabulary:
   $$P_{\text{vocab}}(w) = \text{Softmax}(W_{\text{vocab}} h_t)$$
   where $h_t \in \mathbb{R}^{d_{\text{model}}}$ is the output of the final RMSNorm layer.

2. **Pointer Distribution ($P_{\text{ptr}}$):**
   Scaled dot-product attention between decoder state $h_t$ and prompt token representations $h_1 \dots h_{L_{\text{prompt}}}$:
   $$\text{score}(t, i) = \frac{(W_q h_t)^T (W_k h_i)}{\sqrt{d_{\text{model}}}}, \quad \forall i \in [0, L_{\text{prompt}})$$
   $$P_{\text{ptr}}(i) = \frac{\exp(\text{score}(t, i))}{\sum_{j=0}^{L_{\text{prompt}}-1} \exp(\text{score}(t, j))}$$

3. **Dynamic Copy Gate ($p_{\text{gen}}$):**
   A learned scalar gate in $(0, 1)$ balancing generative token emission (for Plan DSL grammar keywords like `STEP`, `tool_call`, `args=`) versus direct prompt copying (for numeric operands and quoted text):
   $$c_t = \sum_{i=0}^{L_{\text{prompt}}-1} P_{\text{ptr}}(i) h_i$$
   $$p_{\text{gen}} = \sigma(W_{\text{gate}} [h_t ; c_t])$$

4. **Combined Probability Distribution ($P(w)$):**
   $$P(w) = p_{\text{gen}} P_{\text{vocab}}(w) + (1 - p_{\text{gen}}) \sum_{i: x_i = w} P_{\text{ptr}}(i)$$

---

### 3. Boundary & Non-Contamination Constraints
- **Strict Prompt Boundary:** The pointer head is strictly masked to prompt tokens $i \in [0, L_{\text{prompt}})$. It cannot point to target tokens generated in prior output steps or future tokens.
- **Tied Backbone Compatibility:** The core Transformer weights $W_{\text{embed}}$, self-attention blocks, and LMHead remain fully compatible with canonical checkpoint step 10,000 weights.
