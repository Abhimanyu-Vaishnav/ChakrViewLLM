# STEP 201: KEY–VALUE BINDING REPRESENTATION

## Objective
Investigate whether contextual representations in ChakrMicro encode the relationship between a key and its paired value ($B \to Y$) as distinct from individual key/value presence ($B$ exists, $Y$ exists) or negative non-associated pairings ($B \to X$, $B \to Z$, $A \to Y$).

## Methodology
- Extracted contextual slot vectors for key tokens and value tokens.
- Evaluated cosine similarities for true associative pairs $(K_i, V_i)$ vs negative pairs $(K_i, V_j)$ for $j \neq i$.
- Trained linear probes on concatenated binding features to evaluate separability of positive bindings.

## Empirical Results (Seed 42)
| Evaluation Metric | Measured Value | Interpretation |
| :--- | :---: | :--- |
| **Positive Pair Cosine Sim** | **$0.6428$** | Intrinsic contextual alignment between true key and associated value |
| **Negative Pair Cosine Sim** | **$0.5716$** | Baseline semantic/syntactic similarity across non-paired slots |
| **Contrastive Margin** | **$+0.0711$** | Positive contrastive separation indicates emergent relational alignment |
| **Binding Probe Accuracy** | **$0.6667$** | Linear separability of positive vs negative pairings above chance ($0.50$) |
| **Key-Only Probe Accuracy** | **$1.0000$** | Key identity linearly decodable |
| **Value-Only Probe Accuracy** | **$1.0000$** | Value identity linearly decodable |

## Scientific Classification
**DIAGNOSTIC EVIDENCE**: The network exhibits a positive contrastive margin ($+0.0711$) and linear separability ($0.6667$) for paired relationships. However, the modest margin indicates that associative binding is partially overshadowed by token identity representations.
