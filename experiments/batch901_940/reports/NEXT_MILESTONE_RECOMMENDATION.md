# WP 938–940: Next Milestone Recommendation — Batches 941–980

## 1. Recommendation: Joint Autoregressive Plan-and-Slot Tokenization

### Context
Batches 901–940 demonstrated that:
1. **Separating plan topology prediction from autoregressive token generation** introduces an unavoidable topological mismatch: heuristic or classification-based topology selection cannot easily scale to arbitrary, variable-length dependency DAGs (scoring only 2/70 on exact plan match).
2. **Oracle Structure + Oracle Spans scores 70/70 (100.0%)**, proving the Plan DSL itself is fully expressive and capable of representing every benchmark task.
3. **The core missing link is joint autoregressive token emission of abstract slots**: the transformer language model must directly emit slot tokens (e.g. `args={expr=<SLOT_0>*<SLOT_1>}`) as part of its autoregressive vocabulary, with immediate cross-attention conditioning onto the prompt span representations.

### Proposed Architecture for Batches 941–980:
1. **Extend Tokenizer with Special Slot Tokens**:
   Add atomic tokens `<SLOT_0>`, `<SLOT_1>`, `<SLOT_2>`, `<SLOT_3>` to the model vocabulary.
2. **Cross-Conditioned Step Decoding**:
   When the language model emits `<SLOT_k>`, an attention pooling gate immediately queries the prompt embeddings to verify span grounding before emitting closing DSL syntax.
3. **Curriculum Topology Training**:
   Train on a multi-hop curriculum containing variable-length 1-step, 2-step, 3-step, and 4-step plans with explicit `<SLOT_k>` markers.
