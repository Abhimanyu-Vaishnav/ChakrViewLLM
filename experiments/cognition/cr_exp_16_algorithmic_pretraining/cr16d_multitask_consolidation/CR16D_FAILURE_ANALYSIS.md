# CR-16D Systematic Failure Mode Analysis

### 1. Classification of Observed Failure Modes
Analysis of predictions across the 2,600 evaluation probes identifies three distinct failure classes:

```
Failure Distribution in CR-16D:
1. OOD Value Unembedding Projection Failure (~68% of errors)
2. Abstention Degradation under Strong Copying Bias (~22% of errors)
3. Attention Routing Drift on Highly Distracting Contexts (~10% of errors)
```

---

### 2. Deep Dive into Failure Modes

#### Failure Mode 1: OOD Value Unembedding Projection Bottleneck
- **Observed Behavior:**
  - On Group G2 (Seen Key, Unseen Value) and G4 (Unseen Key, Unseen Value), performance drops precipitously (4% - 8% accuracy).
  - When the target is `Cotton`, the model outputs `[UNKNOWN]`.
  - When the target is `Mercury`, the model outputs `Mango`.
  - When the target is `Shield`, the model outputs `Ship`.
  - When the target is `Wool`, the model outputs `Wolf`.
- **Root Cause:**
  - Attention routing to the value token is preserved, but the hidden state vector at the query position must be mapped to token probabilities via the linear unembedding layer $W_U \in \mathbb{R}^{d \times V}$.
  - In a standard autoregressive transformer without a pointer-generator network, generating a token requires that its corresponding row in $W_U$ has high inner product with the contextualized hidden state.
  - In ChakrMicro ($d=192$), tokens never observed as generation targets in retrieval tasks suffer from representational misalignment, collapsing to phonetically or distributionally nearby familiar tokens.

#### Failure Mode 2: Over-Eager Copying Bias (Abstention Breakdown)
- **Observed Behavior:**
  - On CAT9 and CAT10, when the queried key was never defined in context, consolidated models often extract a random value present in context rather than outputting `[UNKNOWN]`.
- **Root Cause:**
  - Multi-task training strongly rewards extracting values from context. In the absence of an exact key match, the attention distribution defaults to an uncalibrated diffuse state over the context tokens, and the copying circuit emits a visible entity value rather than abstaining.

#### Failure Mode 3: Distractor Context Length Sensitivity
- **Observed Behavior:**
  - In contexts with more than 3 distractor key-value pairs, retrieval accuracy degrades slightly (~8% drop compared to single-pair contexts).
- **Root Cause:**
  - With $d_{head} = 32$ across 6 attention heads, the attention capacity is constrained. When multiple keys share subword prefix tokens, query-key attention scores exhibit minor cross-talk.

