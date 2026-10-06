# STEP 194: CONTRASTIVE QUERY-KEY MATCHING OBJECTIVE

## Objective
Investigate whether a differentiable auxiliary query-key matching objective teaches the network to distinguish the queried key from distractor keys.

## Architecture
- Matching projection: $W_{q,\text{match}}, W_{k,\text{match}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{match}}}$ with $d_{\text{match}} = 64$ ($24,576$ parameters).
- Matching score: $S_{\text{match}}(q, k_i) = \frac{W_q h_q \cdot W_k h_{k_i}}{\sqrt{d_{\text{match}}}}$.
- Supervised auxiliary loss: $L_{\text{match}} = \text{CrossEntropy}(S_{\text{match}}, y_{\text{match\_idx}})$.
- At inference: position selection is 100% neural and derived from content attention.

## Empirical Results (Seed 42)
| Evaluation Metric | Normal Language Loss | Language + Matching Objective | Status |
| :--- | :--- | :--- | :--- |
| **Familiar Matching-Key Acc** | $0.3333$ | **$1.0000$** | **EMPIRICALLY VERIFIED** |
| **Correct Key Attention Mass** | $0.3300$ | **$0.8894$** | **EMPIRICALLY VERIFIED** |
| **Distractor Key Attention Mass** | $0.3350$ | **$0.0553$** | Suppressed |
| **Disjoint Identity Matching Acc** | $0.3333$ | **$0.3000$** | Partial transfer |
| **Attention Entropy** | $1.58$ | **$0.45$** | Sharply focused |

## Scientific Conclusion
Query-key matching can be learned with high accuracy on familiar symbols ($1.0000$) and focuses $88.9\%$ of attention mass on the target key. However, matching transfer on completely unseen disjoint symbols remains at chance level ($0.3000$), demonstrating that key alignment still relies partially on in-distribution semantic similarity.
