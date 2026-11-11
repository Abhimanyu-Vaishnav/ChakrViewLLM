# CR-EXP-16F: Mechanistic Investigation & Interference Analysis

## 1. Mathematical Mechanics of Representational Interference

In ChakrMicro v0.1 ($d = 192, L = 6, H = 6, V = 4096$), all knowledge representations, attention routing circuits, and vocabulary projection weights are compressed into $3.44\text{M}$ parameters.

### A. Gradient Conflict Between Verbatim Copying and Key-Value Routing
1. **Algorithmic Copying Mechanism:** Requires induction heads or token-shift circuits across consecutive positions:
   $$A_{i, j} \approx \delta(i, j+k)$$
   The residual stream must maintain token identities invariant to their semantic meaning, directly transferring input embeddings $E[x_t]$ to the output logits.
2. **Key-Value Retrieval Mechanism:** Requires content-addressable attention routing where query token $Q(k)$ matches key token $K(k)$, and value representation $V(v)$ is extracted across arbitrary position offsets $\Delta p \in \{0, 1, 2, 3\}$.
3. **The Clash:** When fine-tuning on position-shuffled key-value pairs with negative examples, gradients update the self-attention projection matrices ($W_Q, W_K, W_V, W_O$) and MLP layers to suppress literal sequence propagation and prioritize key-value association lookups.
   Even with 50% replay in every training batch, the loss on copy replay examples drops during training because the network overfits to the small replay pool, but the general copying induction circuit in the early layers is destroyed by retrieval updates.

### B. Mechanics of Abstention Loss Interference ($\lambda_{\text{neg}}$)
The negative abstention examples require the model to predict `[UNKNOWN]` whenever a query key does not match stored facts:
$$\mathcal{L}_{\text{neg}} = -\log P(\text{"[UNKNOWN]"} \mid \mathbf{x}_{\text{unans}})$$
Because `[UNKNOWN]` is a single token appearing as the target across many structurally diverse negative sequences, the gradient for `[UNKNOWN]` acts as a global attractor in the output projection layer:
$$\frac{\partial \mathcal{L}_{\text{neg}}}{\partial W_{\text{unembed}}} \propto (\mathbf{p} - \mathbf{y}_{\text{unknown}}) \cdot \mathbf{h}_T^\top$$
- Under $\lambda = 1.00$, the unembedding vector for `[UNKNOWN]` receives massive positive reinforcement on every unanswerable example. Since answerable examples also contain multiple distractor keys that do not match the target query, the intermediate representations partially activate the `[UNKNOWN]` detector, triggering false abstentions on over 70% of answerable prompts ($>1000/1400$).
- When $\lambda$ is reduced to $0.25$, the norm of the `[UNKNOWN]` gradient is attenuated by $75\%$. This prevents `[UNKNOWN]` from overwhelming weak positive retrieval signals, but makes the model unable to reliably detect genuinely missing keys, dropping unanswerable recall to below $50\%$.

### C. Mechanism of the Unseen-Value Wall (CAT7)
In all 12 models, zero out of 150 CAT7 unseen-value probes succeeded.
- Why? In a 6-layer 192-dim transformer without explicit copy mechanisms or pointer networks, generating a value requires decoding its specific subword tokens through the unembedding matrix $W_{\text{unembed}} \in \mathbb{R}^{192 \times 4096}$.
- If the model has never trained on routing that specific token from the context to the output in a retrieval prompt template, it relies purely on static token frequencies or falls back to distractor keys. The model never formed an abstract "copy whatever token follows the matched key" circuit; it formed low-rank associative memories between specific token pairs.

