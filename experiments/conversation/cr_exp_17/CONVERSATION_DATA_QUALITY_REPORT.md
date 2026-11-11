# CR-EXP-17: Conversational Dataset Quality & Audit Report

## 1. Corpus Composition and Lineage

The conversational dataset for CR-EXP-17 was engineered to resolve the three primary failure modes identified during the Phase A forensic audit:
1. **Elimination of the "Refusal Attractor Sink"**: Filtered out all instances containing rigid template phrases such as `"not provided in the context"` or `"the provided context does not contain"`.
2. **Elimination of Target-Length Collapse**: Removed artificial single-word slot fillings (`" SQLite"`, `" 8000"`, `" DONE"`) that trained the model to emit EOS prematurely.
3. **Structured Multi-Turn Representation**: Enriched the corpus with balanced multi-turn context retention (name, city, food, and coding language tracking across turns), natural uncertainty clarification, and bilingual Hindi/Hinglish dialogue.

### Split Summary & Artifact Hashes

- **Corpus Name**: `CR17-Conversational-Corpus-v1`
- **Total Clean Records**: 1,576
- **Training Set (`conversation_train_1080.jsonl`)**: 1,419 records
  - SHA-256: `91ca4c4ca4dbe4d59fca76865d4ebf07297e6e58dfda5f69c5e2ef975ea8e24c`
- **Validation Set (`conversation_val_120.jsonl`)**: 157 records
  - SHA-256: `c4960ce0f9f3bfb5883ea57cb5a909b6d8591f24d770c675fb621375b48db9d0`
- **Manifest (`conversation_corpus_manifest.json`)**: Formally recorded with deterministic splitting seed `1337`.

---

## 2. Leakage and Quality Audit

1. **Zero Overlap against `CONVERSATION_BENCHMARK_V1.json`**:
   - Every candidate prompt in the training and validation sets was audited against all 200 held-out evaluation probes.
   - Result: **0 prompt overlaps detected ($0.0\%$ leakage)**.
2. **Vocabulary and Tokenization Integrity**:
   - 100% of tokens in the training and validation splits map cleanly into the frozen 4,096-token vocabulary (`vocab_4096`). Zero `<unk>` tokens are generated.
3. **Sequence Length Distribution**:
   - Mean sequence length: ~65 tokens.
   - Maximum sequence length: 340 tokens (comfortably below the 512-token context ceiling of ChakrMicro v0.1).
