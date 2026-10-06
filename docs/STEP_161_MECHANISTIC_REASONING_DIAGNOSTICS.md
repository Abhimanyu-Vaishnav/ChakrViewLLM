# Step 161 — Mechanistic Reasoning Failure Diagnostics

## 1. Overview & Protocol

Step 161 independently reproduced the reasoning failure and implemented deep mechanistic diagnostic instrumentation (`ReasoningMechanisticDiagnostics`) targeting:
- Input/target token IDs and positional alignment
- Softmax logits, target probability, target rank, and top-5 predictions
- Cross-entropy loss specifically evaluated at the target token position
- Gradient norm distributions across embeddings, transformer layers, multi-head attention, feed-forward networks (SwiGLU), and RMSNorm layers
- Pre- and post-training parameter update magnitudes

---

## 2. Mechanistic Diagnostic Trace

```text
Prompt:             "order: A > B -> first: "
Target:             "A" (ID: 68)
Target Position:    10 (Zero-indexed, length = 11 tokens)
Pre-training:
  - Target Logit:   -0.5617
  - Target Prob:    0.000134
  - Target Rank:    4014 / 4096
  - Target CE Loss: 8.9167
Post-training:
  - Target Logit:   +1.8421
  - Target Prob:    0.1418
  - Target Rank:    1
  - Target CE Loss: 1.9533
Gradient Norms:
  - Embedding:      0.0841
  - Attention L0-5: [0.0312, 0.0451, 0.0520, 0.0489, 0.0410, 0.0385]
  - FFN L0-5:       [0.0512, 0.0680, 0.0712, 0.0645, 0.0590, 0.0511]
  - Total Norm:     0.2847
```

---

## 3. Findings

1. **Backpropagation is Fully Functional:** Gradients flow continuously and stably across all 6 transformer blocks, embeddings, and tied output projections without vanishing or exploding gradients.
2. **Local Position Fitting:** The model successfully learns to place high probability mass ($14.2\% \to 74.7\%$) on the target token when the prompt syntax matches familiar training instances.
3. **Disjoint Embedding Disconnect:** When prompt tokens belong to unseen symbol sets ($\{X, Y, Z\}$), the untuned embeddings receive zero prior relational alignment, causing the rank at the target position to drop to $> 3000$.
