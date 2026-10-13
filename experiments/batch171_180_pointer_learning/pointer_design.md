# Batch 172 — Pointer-Generator Copy Mechanism Design
## Auxiliary Copy Attention, Dynamic Position Gating & Normalized Mixture Distribution

### 1. Mathematical Formulation
To give ChakrMicro the capability to copy directly from the context window without altering canonical parameters:
1. **Base Model Hidden States**: Let $H = [h_1, h_2, \dots, h_T] \in \mathbb{R}^{T \times d_{\text{model}}}$ be the final pre-norm hidden states from ChakrMicro.
2. **Context Window Demarcation**: Let $C = \{p_{\text{start}}, \dots, p_{\text{end}}\}$ be the sequence indices corresponding to the active context tokens (excluding system prompt, user query framing, and future answer tokens).
3. **Pointer Attention Distribution**:
   For the current generation step $t$, the query vector is $q_t = W_q h_t$.
   The keys for context positions $j \in C$ are $k_j = W_k h_j$.
   The pointer attention score over context positions is:
   $$e_{t, j} = \frac{q_t^\top k_j}{\sqrt{d_k}}$$
   $$P_{\text{copy}}(j) = \frac{\exp(e_{t, j})}{\sum_{m \in C} \exp(e_{t, m})}$$
4. **Copy-to-Vocabulary Mapping**:
   If context token $x_j$ appears at multiple positions in $C$, its copy probability is accumulated:
   $$P_{\text{copy\_vocab}}(w) = \sum_{j \in C: x_j = w} P_{\text{copy}}(j)$$
5. **Generation Gate**:
   A learned sigmoid gate $g_t \in [0, 1]$ balances vocabulary generation vs. context pointer copying:
   $$g_t = \sigma(W_g [h_t; c_t] + b_g)$$
   where $c_t = \sum_{j \in C} P_{\text{copy}}(j) h_j$ is the context summary vector.
6. **Final Mixture Distribution**:
   $$P_{\text{final}}(w) = (1 - g_t) P_{\text{vocab}}(w) + g_t P_{\text{copy\_vocab}}(w)$$

### 2. Architectural Guarantees
- **Non-Invasive**: Leaves canonical `ChakrMicro` parameters and `step_10000.pt` checkpoint completely immutable.
- **Strict Position Masking**: Positions outside the designated context span $C$ receive attention weight $-\infty$, preventing leakage from user queries or future answer positions.
- **Normalization Invariant**: $\sum_{w \in V} P_{\text{final}}(w) = 1.0$.
