# CR-16D Variable Abstraction Follow-Up & Synthetic Curriculum

### 1. The Core Variable Abstraction Mystery
Why did ChakrMicro master Group G3 (Unseen Keys paired with familiar values, 94.4%) while failing Group G2 (Familiar Keys paired with unseen values, 8.0%)?

```
Variable Binding Anatomy:
Context: [Store: KEY = VALUE . Query: KEY . Answer: ]
                  │       ▲             │
                  └───────┴─────────────┘
              Attention Matching (Q-K)  ---> Routes correctly to VALUE position
                                               │
                                               ▼
                                      Output Generation
                                      Can token be emitted?
                                      - If VALUE is familiar: YES (94.4% G3)
                                      - If VALUE is unseen:   NO  ( 8.0% G2)
```

### 2. Theoretical Implications
1. **The Query-Key Mechanism is Abstract:**
   The model does not rely on an explicit list of known keys. It has learned an abstract attention operator:
   $$\text{Attn}(q_{\text{query\_key}}, k_{\text{context\_key}}) \approx 1$$
   This attention pattern fires correctly for unseen keys because the positional and syntactic framing (`Store:`, `=`, `Query:`) isolates the key tokens.
2. **The Bottleneck is Generative, Not Attentional:**
   The failure on G2 is not an inability to identify the value token in context; rather, it is the inability of the final layer representation to project onto the unembedding vector of an unpracticed token.

---

### 3. Recommended Follow-Up Curriculum: CR-EXP-16E
To resolve the OOD value generation bottleneck within the canonical architecture:
1. **Dynamic Target Vocabulary Scaling:**
   Instead of training on a restricted set of 30 value nouns, expose the retrieval target position to a randomly sampled distribution spanning the full 4,096-token vocabulary during pretraining.
2. **Subword & Span Tokenization Exposure:**
   Train key-value extraction across single-token, multi-token, and alphanumeric value spans to encourage the residual stream to align arbitrary token embeddings with output logits.
3. **Dedicated Abstention Calibration Loss:**
   Introduce calibrated confidence loss or explicit negative key samples to prevent the strong copying bias from suppressing `[UNKNOWN]` predictions on unbound queries.

