# CV-GENESIS-NR-010 Tokenizer Compatibility Specification

## 1. Executive Summary

Mission **CV-GENESIS-NR-010 Phase 2** definitively resolved the special token configuration discrepancies observed between NR-005 and NR-008, establishing the immutable canonical contract for all 17M foundation models.

---

## 2. Canonical Tokenizer Specification

- **Vocabulary Size:** `4,096` BPE tokens.
- **Vocabulary File:** `data/experiments/vocab_4096/vocab.json`.
- **Vocabulary Cryptographic SHA-256:** `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a`.
- **Merges File:** `data/experiments/vocab_4096/merges.json`.
- **Merges SHA-256:** `7498d92adeef7c9e64d4d1bbd5b4b23f8774692d868742e3b98e30c584c4403f`.

---

## 3. Special Token Contract

| Special Token | Token ID | Semantics / Role |
| :--- | :--- | :--- |
| `<BOS>` | `0` | Sequence initialization control token. |
| `<EOS>` | `1` | End-of-sequence delimiter (supervised termination). |
| `<PAD>` | `2` | Batch padding token (masked with `-100` during loss computation). |

*Resolution Note:* All future training scripts, data preparation pipelines, and decoding loops strictly bind `eos_token_id = 1` and `pad_token_id = 2`.
