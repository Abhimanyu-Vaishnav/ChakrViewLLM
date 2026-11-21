# CHAKRVIEW RESEARCH REPORT: CV-GENESIS-NR-004
## TOKENIZER ALIGNMENT VERIFICATION REPORT

**Mission ID:** CV-GENESIS-NR-004  
**Date:** 21 November 2026  
**Status:** COMPLETE & VERIFIED  
**Lead Engineer:** Principal Neural Network Research Engineer, ML Systems Engineer  

---

### 1. Executive Summary

During investigation mission CV-GENESIS-NR-003, a catastrophic alignment fault was discovered in the ChakrGenesis foundation training lineage:
- Pre-tokenized binary dataset shards (`data/tokenized/stage_c/train/*.bin`) were generated using a canonical vocabulary (`data/experiments/vocab_4096/vocab.json`).
- However, pretraining and inference scripts in Project Genesis were configured to point to a candidate vocabulary (`experiments/tokenizer/candidates/v4096/vocab.json`).
- Out of 4,096 tokens, **3,677 token IDs were completely permuted** between the two files. The embedding table learned in previous pretraining was indexed according to candidate token IDs, while binary shards fed canonical token IDs. This caused complete semantic dislocation where the model produced token sequences that decoded into scrambled, nonsensical fragments.

In Stage 1 of CV-GENESIS-NR-004, strict cryptographic verification and token alignment audits were executed before launching training.

---

### 2. Cryptographic Tokenizer Hash Audit

| Tokenizer Path | File Size | SHA-256 Hash | Status |
| :--- | :--- | :--- | :--- |
| `data/experiments/vocab_4096/vocab.json` | 112,627 bytes | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | **CANONICAL GROUND TRUTH** |
| `experiments/tokenizer/candidates/v4096/vocab.json` | 112,969 bytes | `fbde6847bb4e678edd2590908f472c0883d74c6abcb92c9b626269ea18a28adb` | **DEFECTIVE / MISMATCHED** |

#### Dataset Manifest Provenance Verification
Audit of `data/manifests/stage_c_manifest.json` confirms:
- **`tokenizer_path`**: `"data/experiments/vocab_4096"`
- **`vocab_size`**: `4096`
- **`total_shards`**: 27 train shards (`6,651,093` tokens) + 2 validation shards (`500,000` tokens) + 3 test shards (`750,000` tokens).

---

### 3. Shard Token Decoding Audit

To conclusively verify that `data/experiments/vocab_4096` matches the binary token data, `shard_00000.bin` (first 1,000 tokens) was decoded using both tokenizers:

1. **Decoded with Canonical Tokenizer (`db4d6119...`):**
   - Clean, grammatical English sentences (Wikipedia, Science, Python documentation).
   - Coherent Hindi Devanagari text.
   - Clean Python code constructs (`def`, `import`, indentation, variables).
2. **Decoded with Candidate Tokenizer (`fbde6847...`):**
   - Scrambled byte strings and inverted glyphs.
   - Misaligned punctuation and disjointed character fragments.

---

### 4. Hard Verification Gate in Pretraining Pipeline

To guarantee that mismatched tokenizers can never again corrupt training or inference, a programmatic verification assertion was baked directly into `experiments/genesis_canonical_pretraining_004/train_canonical_pretraining.py`:

```python
EXPECTED_VOCAB_HASH = "db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a"

vocab_file = Path(tokenizer_dir) / "vocab.json"
actual_hash = hashlib.sha256(vocab_file.read_bytes()).hexdigest()
if actual_hash != EXPECTED_VOCAB_HASH:
    raise ValueError(
        f"FATAL: Tokenizer vocabulary hash mismatch!\n"
        f"Expected: {EXPECTED_VOCAB_HASH}\n"
        f"Actual:   {actual_hash}"
    )
```

Furthermore, both saved checkpoints (`canonical_pretrain_best.pt` and `canonical_pretrain_final.pt`) explicitly embed `tokenizer_vocab_hash` in their metadata dictionary.
