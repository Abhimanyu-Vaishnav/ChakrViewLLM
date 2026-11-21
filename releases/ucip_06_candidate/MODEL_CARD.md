# ChakrView Model Card — UCIP-06 Candidates

## Model Details
- **Architecture:** `ChakrTransformerLM` (Decoder-only Causal Transformer)
- **Token Vocabulary:** 4,096 byte-level BPE tokens (`data/experiments/vocab_4096`)
- **Execution Target:** CPU-first, local-first execution.
- **Candidates Released:**
  - `chakrview_ucip06_medium_model.pt` (12.93M parameters, 7 layers, $d=384, n_h=6, d_{ff}=896$)
  - `chakrview_ucip06_micro_model.pt` (3.44M parameters, 6 layers, $d=192, n_h=6, d_{ff}=512$)

## Verified Hashes
- **Medium SHA-256:** `be843bc2cbc08f3f8459b85d6eec2cdce534441ac5ccc797f0bc0d2be8aa1b72`
- **Micro SHA-256:** `7706e19ba568946c527cc15d90dc7b50180b7c754eb2b5b6dd7a5cdb3c55fd71`
- **Protected Runtime SHA-256 (`chakrview/runtime/session.py`):** `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` (Preserved intact)

## Intended Use
Local research and evaluation of small-parameter autoregressive models for Hindi, Roman Hindi, Hinglish, and English without external pretraining or API dependencies.

## Known Limitations
1. **Arithmetic:** Complex multi-step reasoning or word problems often experience numerical hallucination due to lack of an explicit scratchpad mechanism.
2. **Long Generation Drift:** Sequences beyond 30 tokens may experience token repetition or grammatical degradation.
3. **Capacity Constraints:** 3.44M to 12.93M parameter budgets require careful trade-offs between broad factual coverage and narrow task accuracy.
