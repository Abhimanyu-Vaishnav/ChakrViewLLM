# CR-EXP-14 Phase 7: Architectural Diagnostic & Mechanistic Analysis

## 1. Architectural Configuration Under Inspection
- **Model:** ChakrMicro
- **Parameters:** $3,443,136$ (~$3.44\text{M}$)
- **Hidden Dimension ($d_{\text{model}}$):** $192$
- **Layers:** 6 Transformer Blocks
- **Attention Heads:** 6 heads per block ($d_{\text{head}} = 32$)
- **Feed-Forward Dimension:** $768$ ($4 \times d_{\text{model}}$)
- **Positional Encoding:** Rotary Positional Embeddings (RoPE)
- **Vocabulary:** 4,096 ByteBPE

---

## 2. Mechanistic Analysis: Why Dynamic Binding Fails at $3.44\text{M}$ Parameters

### The Induction Head Bottleneck
In transformer circuit theory (e.g., Elhage et al., 2021), in-context copy and binding operations require a coordinated **two-head Induction Circuit**:
1. **Previous-Token Head ($L_{k}$):** Attends to token $A$ and writes representation of $A$'s follower token $B$ into the residual stream.
2. **Induction Head ($L_{k+m}$):** When reading query token $A^*$ at position $t$, attends back to previous occurrences of $A$ and copies corresponding token $B$ to predict the next token.

In ChakrMicro:
- With only 6 layers and $d_{\text{head}}=32$, each attention head has minimal subspace capacity.
- Indian names and locations in ByteBPE vocabulary are frequently tokenized into 3 to 5 arbitrary subword fragments (`'A'`, `'bha'`, `'y '`).
- An induction head must not merely bind single-token ids, but bind **multi-token entity spans** across variable distances.
- Because $d_{\text{head}}=32$ is too narrow to simultaneously encode positional offsets, subword boundary markers, and semantic role bindings, the network resorts to an easier empirical loss minimum: memorizing high-frequency unigram/bigram token continuations (`Ananya is currently in...`).

### Capacity Satiation vs. Memorization
- When fine-tuned on 600 dialogues, training loss rapidly plummeted to $0.90$ (Candidate A/B) and $0.57$ (Candidate C).
- In-distribution tests showed that Candidate A could successfully track state updates for names present during training (`Aaditya` $\to$ `Bhopal`).
- However, when presented with novel name spans (`Abhay`, `Bimla`, `Chetan`), the weights could not generalize the routing logic. This demonstrates that the capacity was consumed by **subword token memorization** rather than learning an abstract, symbol-independent pointer mechanism.

---

## 3. Necessary Architectural Conditions for v0.9.1+ Readiness

To enable genuine symbol-independent entity binding in low-resource neural brains:
1. **Head Dimension Expansion:** $d_{\text{head}}$ should be expanded from $32$ to at least $64$ (e.g., $d_{\text{model}}=256, 4\text{ heads}$ or $d_{\text{model}}=384, 6\text{ heads}$), providing sufficient vector subspace for key-query matching across multi-token spans.
2. **Span-Level Position Indication or Delimiter Pretraining:** Structural delimiters (`<ent>`, `</ent>`) are syntactically useful, but must be introduced during pretraining rather than short fine-tuning to avoid catastrophic forgetting and overfitting.
3. **Explicit Synthetic Pointer Loss / Copy Mechanisms:** Training objectives should incorporate pointer auxiliary losses or copy-enforcing masking schemes that explicitly penalize unconditioned name substitution.
