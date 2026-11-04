# CBR-01 — WORKSTREAM A & C: TOKENIZER AND PROMPT CONTRACT AUDIT

---

## 1. Tokenizer Specifications & Special Tokens

- **Tokenizer Type:** Custom Byte-Pair Encoding (BPE) with UTF-8 byte fallback.
- **Vocabulary Size:** 4,096 tokens.
- **Artifacts:**
  - `vocab.json` (SHA-256: `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a`)
  - `merges.json` (SHA-256: `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7`)
- **Special Token Table:**

| Token Name | Token String | Token ID | Role |
| :--- | :--- | :---: | :--- |
| `<BOS>` | `<|bos|>` | `0` | Sequence start marker |
| `<EOS>` | `<|eos|>` | `1` | Sequence end / turn termination |
| `<PAD>` | `<|pad|>` | `2` | Batch padding token |
| `<UNK>` | `<|unk|>` | `3` | Unknown fallback |
| `<IM_START>` | `<|im_start|>` | Merged sequence | Turn delimiter start |
| `<IM_END>` | `<|im_end|>` | Merged sequence | Turn delimiter end |

---

## 2. Key Diagnostic Finding: Tokenizer Fragmentation of Dynamic Entities

From the M5.1 forensic audit, a primary bottleneck in small models (3.44M parameters) is **subword fragmentation of dynamic entity names**:
- In a 4,096 vocabulary, common English/Indian entities and numbers fragment across 3 to 6 subword tokens (e.g. `Meera` $\to$ `[Me, er, a]`).
- For a small 6-layer model, copying an entity requires a sequential 3-step autoregressive induction copy rather than a single direct token emission.
- **Mitigation Strategy:**
  1. Train on explicit algorithmic induction tasks to strengthen copy attention circuits across token spans.
  2. Normalize conversational prompts to provide clear delimiters surrounding dynamic variables.
  3. Ensure whitespace parity so tokens are not fractured into unnatural single-character pieces.
