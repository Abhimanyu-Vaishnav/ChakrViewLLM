# STEP 187: NEURAL COPY / POINTER MECHANISM

## Objective
Design and implement a minimal, fully differentiable neural copy/pointer mechanism that allows ChakrMicro to attend directly to contextual tokens and scatter source probabilities into the output vocabulary distribution without relying on static embedding geometry.

## Architecture
- **Query & Key Projections**:
  $$q_i = W_q h_i \in \mathbb{R}^{d_{\text{model}}}$$
  $$k_j = W_k h_j \in \mathbb{R}^{d_{\text{model}}}$$
- **Causal Copy Attention**:
  $$S_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_{\text{model}}}}, \quad \alpha_{ij} = \text{softmax}(S_{i,:})_j$$
- **Copy vs Generation Gating**:
  $$g_i = \sigma(W_g h_i + b_g) \in [0, 1] \implies P_{\text{gen}} = g_i, \quad P_{\text{copy}} = 1 - g_i$$
- **Vocabulary Scattering & Mixture**:
  $$P(w \mid x_{\le i}) = g_i P_{\text{gen}}(w \mid x_{\le i}) + (1 - g_i) \sum_{j: x_j = w} \alpha_{ij}$$
- **Parameter Overhead**:
  - $W_q$: $192 \times 192 = 36,864$
  - $W_k$: $192 \times 192 = 36,864$
  - $W_g, b_g$: $192 + 1 = 193$
  - Total head parameters: **$73,921$** ($+2.15\%$ overhead over baseline $3,443,136$).

## Invariants Adherence
- Strictly 100% neural and differentiable.
- Zero Python dictionary lookups, string pattern extractors, or external answer lookups.
- Exposes complete diagnostic tensors: $P(\text{copy})$, selected position $\arg\max(\alpha)$, and source token identity.
