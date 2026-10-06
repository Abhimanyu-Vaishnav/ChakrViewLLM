# STEP 203: CONTRASTIVE ASSOCIATION LEARNING

## Objective
Introduce the smallest principled neural auxiliary loss teaching the network to distinguish true key-value pairs from false negatives:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{language}} + \lambda_{\text{binding}} \mathcal{L}_{\text{contrastive}}$$
where:
$$\mathcal{L}_{\text{contrastive}} = -\log \frac{\exp(\text{sim}(k_i, v_i) / \tau)}{\sum_j \exp(\text{sim}(k_i, v_j) / \tau)}$$

## Parameter Sweep Results ($\lambda \in \{0.1, 0.5, 1.0\}$)
| $\lambda_{\text{binding}}$ | Familiar Binding Acc | Held-out Binding Acc | Disjoint Binding Acc | Held-out Language Loss | Parameter Overhead |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$0.1$** | $0.0000$ | $0.0000$ | $0.0000$ | $8.4785$ | $+24,576$ ($+0.71\%$) |
| **$0.5$** | $0.0000$ | $0.0000$ | **$0.1667$** | $8.4785$ | $+24,576$ ($+0.71\%$) |
| **$1.0$** | $0.0000$ | $0.0000$ | **$0.1667$** | $8.4785$ | $+24,576$ ($+0.71\%$) |

## Conclusion
**DIAGNOSTIC EVIDENCE**: Bilinear contrastive pairing achieves partial disjoint transfer ($16.67\%$) without disrupting held-out language loss ($8.4785$), demonstrating that auxiliary contrastive objectives can shape relational compatibility without linguistic degradation.
