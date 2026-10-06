# STEP 185: UNTIED READOUT BASELINE INVESTIGATION

## Objective
Test whether the tied LM head ($W_{\text{out}} = E^T$) is the primary bottleneck preventing zero-shot disjoint token retrieval.

## Architecture Modification
- **Baseline**: Tied projection $W_{\text{out}} \in \mathbb{R}^{d_{\text{model}} \times V}$ sharing the exact parameters of $E \in \mathbb{R}^{V \times d_{\text{model}}}$ ($3,443,136$ total parameters).
- **Candidate**: Independent learned linear projection $W_{\text{out}} \in \mathbb{R}^{d_{\text{model}} \times V}$ initialized with $E^T$ ($4,229,568$ parameters, $+786,432$ parameter delta, $+22.84\%$ capacity).
- **Backbone**: Frozen Transformer backbone (6 layers, $d_{\text{model}}=192$, 6 heads, SwiGLU FFN, RoPE embeddings).

## Experimental Design
- Train on familiar mappings: `map |A| -> |1| and |B| -> |2| and |C| -> |3| query |A| -> |1|`
- Evaluate on:
  1. Familiar train pairs (`A->1`, `B->2`, `C->3`)
  2. Held-out value permutations (`A->3`, `B->1`, `C->2`)
  3. Disjoint key/value pairs (`X->7`, `Y->8`, `Z->9`)

## Empirical Results
| Candidate | Total Parameters | Trainable Parameters | Familiar Acc | Held-out Perm Acc | Disjoint Acc | Disjoint Target Prob | Median Rank | Cosine Sim to $E^T$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tied Baseline** | $3,443,136$ | $3,443,136$ | $1.0000$ | $0.3333$ | **$0.0000$** | $1.97 \times 10^{-6}$ | $1,197$ | $1.0000$ |
| **Untied Readout** | $4,229,568$ | $4,229,568$ | $1.0000$ | $0.3333$ | **$0.0000$** | $3.18 \times 10^{-6}$ | $1,094$ | $0.5303$ |

## Scientific Conclusion
Untied readout **does not** solve disjoint-token retrieval. Although the independent LM head has $786,432$ additional parameters, the columns corresponding to unseen tokens (e.g. `X`, `Y`, `7`, `8`) receive zero gradient updates during associative training. The tied readout was **not** the primary bottleneck causing zero-shot disjoint retrieval failure; rather, standard vocabulary-space readout cannot emit unseen token IDs without contextual routing mechanisms.
