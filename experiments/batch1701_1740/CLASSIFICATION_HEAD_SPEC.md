# CLASSIFICATION HEAD SPECIFICATION
## Batches 1701–1740 Architecture & Tensor Pooling Contract for Direct Neural Routing

### 1. Architectural Design & Pooling Formulation

The primary bottleneck identified in Batches 1661–1700 was autoregressive token generation: generating a 30-token JSON string on CPU required ~260 ms and suffered an 85.8% syntax invalidity rate.

The **Direct Neural Classification Head** replaces autoregressive generation with a single-pass sequence classifier:

```
Input Tokens: [BOS, t_1, t_2, ..., t_L]
       │
       ▼
ChakrMicro Transformer Backbone (6 Layers, d_model=192, RoPE, SwiGLU)
       │
       ▼
Sequence Hidden Representations: H in R^{B x L x 192}
       │
       ▼
Representation Pooling: Last Non-Padding Token Representation
       h_pooled = H[b, last_valid_idx, :] in R^{192}
       │
       ▼
Layer Normalization: RMSNorm(h_pooled) in R^{192}
       │
       ▼
Linear Classification Projection: W_cls in R^{192 x 6} + b_cls in R^{6}
       │
       ▼
Logits in R^{6} ──(Softmax)──► Class Probabilities: P(Intent | Query)
```

---

### 2. Specification of Controlled Model Variants

We implement and evaluate three distinct neural variants on the identical frozen dataset:

#### Variant A: Frozen Backbone + Linear Head
- **Backbone Weights:** Frozen completely (`requires_grad = False` on all 3,443,136 backbone parameters).
- **Trainable Parameters:** Only `W_cls` ($192 \times 6 = 1,152$) + `b_cls` (6) + `norm` (192) = **1,350 trainable parameters**.
- **Research Question:** Does the frozen canonical checkpoint already encode linearly separable semantic representations for tool intents?

#### Variant B: Fine-Tuned Backbone + Linear Head
- **Backbone Weights:** Full fine-tuning of all 6 Transformer layers (`requires_grad = True`).
- **Trainable Parameters:** $3,443,136 + 1,350 = \mathbf{3,444,486}$ trainable parameters.
- **Learning Rate:** $3 \times 10^{-5}$ on backbone, $1 \times 10^{-4}$ on classification head.
- **Research Question:** Can gradient updates adapt the attention heads to cluster semantic intent representations without catastrophic collapse?

#### Variant C: Autoregressive Baseline (Replication)
- Causal LM generation predicting JSON text (`"Router: {\"intent\": ...}"`).
- Serves as the control to directly measure the latency and validity difference between generation and classification.

---

### 3. Representation Selection & Pooling Rules
1. **No `[CLS]` Token in Causal Decoder:** Because ChakrMicro is a causal decoder-only architecture trained with causal attention masks, earlier tokens cannot attend to subsequent tokens. Therefore, the representation of the **final non-padding token** ($t_L$) contains full causal attention over the entire utterance and serves as the pooled representation.
2. **Padding Invariance:** Batched inputs are masked such that padding tokens (`pad_token_id=2`) do not contribute to loss or pooling indices.
3. **Guaranteed Output Validity:** The classification head outputs an argmax over the 6 allowed intents. Syntax invalidity rate is mathematically **0.0%**.
