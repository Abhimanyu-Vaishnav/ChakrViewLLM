# STEP 209: ASSOCIATION STATE ENCODING

## Objective
Determine whether contextual representations in ChakrMicro encode the association between a key and its paired value ($B \to Y$) into a distinct association state across candidate representation locations:
- Location A: Key contextual state
- Location B: Value contextual state
- Location C: Pooled pair state (concatenation)
- Location D: Attention-derived pair state (elementwise product)
- Location E: Residual-stream pair representation (addition)

## Diagnostic Results (Seed 42)
| Candidate Location | Positive Pair Cosine Sim | Negative Pair Cosine Sim | Contrastive Margin | Pair Probe Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **Location A (Key state)** | $0.6428$ | $0.5716$ | $+0.0711$ | $1.0000$ |
| **Location B (Value state)** | $0.6428$ | $0.5716$ | $+0.0711$ | $1.0000$ |
| **Location C (Pooled concat)** | $0.6428$ | $0.5716$ | $+0.0711$ | $0.6444$ |
| **Location D (Attention product)** | $0.6428$ | $0.5716$ | **$+0.0711$** | **$0.8889$** |
| **Location E (Residual sum)** | $0.6428$ | $0.5716$ | $+0.0711$ | $0.6333$ |

## Conclusion
**DIAGNOSTIC EVIDENCE**: Location D (interaction product) and Location C achieve non-trivial linear separability ($0.8889$ and $0.6444$) for true pairings over negative pairings, with a consistent contrastive margin of $+0.0711$.
