# STEP 193: KEY / VALUE REPRESENTATION PROBING

## Objective
Determine whether contextual hidden states in ChakrMicro encode:
- Key identity
- Value identity
- Matching key detection
- Associated value representation
- Positional structure and entropy

## Methodology
Controlled prompts:
`map |A| -> |1| and |B| -> |2| query |A| -> |`
Trained linear probes on extracted hidden states at key positions, value positions, query position, and prediction position.

## Diagnostic Results (Seed 42)
| Probe Target | Accuracy | Diagnostic Metric | Interpretation |
| :--- | :--- | :--- | :--- |
| **Key Identity Probe** | **$1.0000$** | Linearly separable across all keys | Contextual hidden state strongly preserves key identity |
| **Value Identity Probe** | **$1.0000$** | Linearly separable across all values | Contextual hidden state strongly preserves value identity |
| **Query-Key Cosine Similarity** | **$0.8204$** | Mean similarity: $0.82$ (query vs matching key) | Representations are naturally aligned in embedding space |
| **Query-Value Cosine Similarity** | $0.0000$ | Orthogonal / unaligned prior to routing | Query does not directly mirror value before associative routing |
| **Baseline Attention Entropy** | $1.58$ | Distributed attention over prompt context | Uniform attention spread without focused routing |

## Scientific Conclusion
**Key and value identities are decodable with 100% linear probe accuracy.** The representation itself contains the requisite identity information. Therefore, the failure of zero-shot disjoint retrieval is not a representation-erasure bottleneck, but a failure of the internal routing circuit to transfer that representation to the prediction position.
