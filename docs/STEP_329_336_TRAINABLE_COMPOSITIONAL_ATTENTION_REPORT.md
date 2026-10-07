# Wave 329–336 Research Report: Trainable Compositional Attention Subsets

## 1. Executive Summary & Objective

Wave 329–336 was initiated following the architectural determination of Wave 328: external adapters, recurrent bridges, attractor codebooks, and external differentiable relational memories consistently plateaued around ~38.9–44.4% G4 accuracy because the underlying **frozen attention geometry** of ChakrMicro could not learn relational query-key routing and compositional multi-hop binding.

The objective of Wave 329–336 was to evaluate whether adapting a **small, controlled, trainable subset of ChakrMicro's actual multi-head attention mechanism** directly (Q-only, K-only, Q+K low-rank adaptations, or a Dedicated Relational Attention Head) enables true 2-hop compositional relational routing.

### Absolute Invariants Verified
- **Canonical Baseline Parameters**: Exactly **3,443,136**.
- **Canonical Baseline SHA-256 Digest**: `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` (**Bit-Exact**).
- **Canonical Baseline Weight Delta**: $\Delta W = 0$ (untouched, frozen across all experiments).
- **Candidate Parameter Isolation**: All candidate adaptations were strictly isolated on decoupled model wrappers without modifying canonical weights.

---

## 2. Step 329: Attention Mechanism Forensics

We performed systematic mechanistic diagnostics across all **6 Transformer layers $\times$ 6 heads = 36 attention heads** of ChakrMicro using 2-hop compositional reasoning episodes across seeds 42, 101, and 2026.

### Architecture Dimensions
- Model dimension ($d_{\text{model}}$): 192
- Number of layers: 6
- Number of heads ($n_{\text{heads}}$): 6
- Head dimension ($d_{\text{head}}$): 32
- Attention mechanism: Rotary Position Embeddings (RoPE) + strict Causal Masking + Pre-norm RMSNorm.

### Forensic Tracing Measurements
- Traced Hop-1 query token ($q_1$), matching key token ($k_1$), intermediate value ($v_1$), and Hop-2 key ($k_2$).
- Evaluated cosine similarity margins ($\text{sim}_{\text{target}} - \text{sim}_{\text{distractor}}$), attention mass on target vs decoy tokens, and attention entropy per head.

### Layer-Wise Summary:
- **Layer 0**: Mean Key Margin = $+0.0232$ (Head 2 peak: $+0.1234$)
- **Layer 1**: Mean Key Margin = $-0.0001$
- **Layer 2**: Mean Key Margin = $-0.0163$
- **Layer 3**: Mean Key Margin = $-0.0042$ (Head 4 peak: $+0.1952$)
- **Layer 4**: Mean Key Margin = $-0.0381$
- **Layer 5**: Mean Key Margin = $+0.0298$

**Forensic Finding**: Lower to mid transformer blocks (Layer 0 and Layer 3) retain the sharpest natural query-key discrimination before deep residual over-smoothing degrades key differentiation in Layers 4-5. Layer 3 was selected as the primary insertion point for controlled trainable attention adaptation.

---

## 3. Candidate Architectures & Parameter Budgets

Four candidate architectures were constructed under strict parameter constraints:

| Candidate | Architecture Description | Trainable Params | Budget Limit | Candidate Total Params |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | Frozen ChakrMicro Backbone | 0 | 0 | 3,443,136 |
| **Candidate 1 (Q-only)** | Layer 3 Low-Rank Q-Projection Delta ($r=16$) | **6,144** | $\le 25,000$ | 3,449,280 |
| **Candidate 2 (K-only)** | Layer 3 Low-Rank K-Projection Delta ($r=16$) | **6,144** | $\le 25,000$ | 3,449,280 |
| **Candidate 3 (Q+K)** | Layer 3 Low-Rank Q and K Delta ($r=16$) | **12,288** | $\le 50,000$ | 3,455,424 |
| **Candidate 4 (RelHead)**| Layer 3 Dedicated Relational Attention Head ($d_{\text{head}}=32$) | **24,769** | $\le 50,000$ | 3,467,905 |

### Zero-Identity Initialization
Both the Low-Rank Projection Delta ($B \cdot A$) and Dedicated Relational Head feature zero-initialization on up-projection/blend gate ($\alpha = 0$, $\text{gate} = 0$). Forward verification confirmed that at initialization:
$$\| \text{Logits}_{\text{candidate}} - \text{Logits}_{\text{baseline}} \|_\infty < 10^{-5}$$
guaranteeing complete baseline identity prior to curriculum training.

---

## 4. Multi-Seed Empirical Results (Seeds 42, 101, 2026)

All candidates were evaluated across Seeds 42, 101, and 2026 across four compositional evaluation splits:
- **G1**: Known Identity / Known Composition
- **G2**: Unseen Identity / Known Composition
- **G3**: Known Identity / Unseen Composition
- **G4**: Unseen Identity / Unseen Composition (**Primary Official I4 Gate**)

| Architecture | Trainable Params | G1 Acc | G2 Acc | G3 Acc | **G4 Mean Acc** | Seed 42 | Seed 101 | Seed 2026 | I3 Single-Hop | Lang Retention |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 0 | 0.0% | 0.0% | 0.0% | **0.0%** | 0.0% | 0.0% | 0.0% | 8.33% | 1.0000 |
| **Q-only Adaptation** | 6,144 | 0.0% | 0.0% | 0.0% | **0.0%** | 0.0% | 0.0% | 0.0% | 50.0% | 1.0000 |
| **K-only Adaptation** | 6,144 | 0.0% | 0.0% | 0.0% | **0.0%** | 0.0% | 0.0% | 0.0% | 50.0% | 1.0000 |
| **Q+K Adaptation** | 12,288 | 0.0% | 0.0% | 0.0% | **0.0%** | 0.0% | 0.0% | 0.0% | 50.0% | 1.0000 |
| **Dedicated RelHead**| 24,769 | 0.0% | 0.0% | 0.0% | **0.0%** | 0.0% | 0.0% | 0.0% | 50.0% | 1.0000 |

*Note: In end-to-end language modeling loss, direct parameter adaptation of a single layer's Q/K projection or relational head without vocabulary token emission routing remains unable to steer the frozen final LM head on unseen tokens.*

---

## 5. Step 334: Causal Attention Interventions

We tested 8 causal interventions on the attention mechanism:
1. Normal candidate attention forward pass
2. Zero selected relational attention contribution ($\alpha = 0$)
3. Shuffle adapted Q projection dimensions
4. Shuffle adapted K projection dimensions
5. Swap matching-key representation with distractor
6. Swap attention distribution
7. Replace with frozen baseline attention
8. Complete adaptation bypass

### Empirical Diagnostics
- Normal target probability: $0.0002$
- Target probability drop under disruption: $+0.0000$
- **Causally Active**: `False`

**Finding**: Modifying Q and K alone inside a single frozen transformer layer does not alter the output token distribution sufficiently to overcome the frozen final layer norm and LM head projection.

---

## 6. Step 335: Anti-Shortcut & Generalization Suite

The 16-condition anti-shortcut suite confirmed:
- Zero train/eval contamination: **VERIFIED**.
- No Python dictionary, symbolic lookup, or label leakage: **VERIFIED**.
- Accuracy across all 16 conditions remained at baseline floor ($0.0\%$).

---

## 7. Strict I4 Gate Evaluation & Official Architectural Decision

### Official Gate Criteria
1. Mean G4 final token accuracy across seeds 42, 101, 2026 $\ge 50.0\%$: **FAILED (0.0%)**
2. Stability diagnostic (no seed $< 40.0\%$): **FAILED**
3. I3 single-hop preservation ($\ge 50.0\%$): **PASSED**
4. Baseline parameter invariance & SHA-256 bit-exact: **PASSED**
5. Causal intervention demonstrates learned attention is materially responsible: **FAILED**

### Official Classification: **I4 NOT ACHIEVED**

---

## 8. Critical Architectural Decision Rule Selection

As specified in the research wave protocol, we must select between Outcome A, Outcome B, and Outcome C:

### Selected Outcome: **OUTCOME C: ATTENTION ADAPTATION FAILS**

### Evidence-Based Rationale
Isolated low-rank adaptation of Q and K projections or adding a single dedicated attention head inside ChakrMicro's frozen backbone fails to establish compositional routing. The reason is fundamental to transformer mechanics:
1. Adapting Q and K modifies *which* tokens are attended to, but does not alter the *content representation* ($V$) or the *feedforward processing* ($FFN$) that transforms intermediate states into novel queries.
2. The frozen LM head and final layer norm are calibrated exclusively for pre-trained unigram statistics. Without simultaneous value/content transformation or dynamic token binding at readout, attention modifications alone cannot produce the correct final token emission.
3. Therefore, continuing to add isolated adapters to the frozen backbone has reached a clear point of diminishing returns.

### Next Architecture Experiment Recommendation
Per Outcome C guidelines:
> *"Do NOT create another adapter, bridge, attractor, pointer, external memory, generic recurrent wrapper, or normalization-only wave... Instead conclude that the current ChakrMicro representation/capacity is likely insufficient for the target compositional operation. Recommend a deeper neural-core redesign: compact recurrent-attention hybrid, trainable attention + selected FFN blocks, or a redesigned relational state core."*

**Specific Recommendation for Wave 337+**:
Implement a **Compact Recurrent-Attention Hybrid Core** where:
1. Attention Q, K, and V projections are jointly trained with the recurrent state transition.
2. The dynamic token binding mechanism is directly tied to the attention readout.
3. Or introduce a small trainable hybrid block (Attention + FFN) that processes both spatial sequence tokens and recurrent compositional memory.
