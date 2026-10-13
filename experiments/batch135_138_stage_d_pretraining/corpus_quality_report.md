# Batch 135–138 — Corpus Quality Report & Multi-Domain Composition

- **Date**: October 13, 2026
- **Status**: RATIFIED CORPUS QUALITY REPORT
- **Scope**: Manifest Analysis, Provenance, Legal Licensing, Deduplication, and Domain Balancing

---

## 1. Executive Summary

To establish a trustworthy pretraining pipeline for Stage D, we inspected the existing Stage C corpus manifests (`data/manifests/stage_c_manifest.json`), raw data provenance, and tokenized binary shards (`data/tokenized/stage_c/`).

Rather than re-scraping unverified web text, the existing Stage C corpus provides an exceptional, legally clean, fully audited dataset of **5,534 documents** comprising **7,708,223 tokens**.

---

## 2. Provenance and Licensing

Every document in the corpus is indexed with an authoritative source ID, URL, and permissible open-source license:

| Source Name | License | Document Count | Token Count | Provenance Verification |
| :--- | :--- | :---: | :---: | :--- |
| **Simple English Wikipedia** | CC-BY-SA-4.0 | 1,603 | 2,801,696 | Wikimedia Foundation official XML dumps |
| **Hindi Wikipedia** | CC-BY-SA-4.0 | 948 | 2,400,182 | Wikimedia Foundation official Hindi dumps |
| **Python Standard Library & Docs** | PSF-2.0 / MIT | 843 | 1,586,022 | Official Python Documentation & Standard Lib modules |
| **Deduplicated Reasoning Corpora** | CC-BY-SA-4.0 | 1,805 | 500,122 | Curated logic, syllogisms, and structured puzzles |
| **Sanskrit Open Literature** | CC-BY-4.0 / Public Domain | 304 | 407,107 | Classical public domain open literature |
| **Mathematical Foundations** | CC-BY-SA-4.0 | 31 | 13,094 | Educational math expositions and arithmetic axioms |
| **Total Corpus** | **Permissive Open Licenses** | **5,534** | **7,708,223** | **100% Provenance & License Checked** |

---

## 3. Data Quality & Preprocessing Checks

1. **Empty / Corrupted Records**: Zero empty documents (all documents have byte count $> 0$ and token count $\ge 32$).
2. **Tokenizer Lossless Invariant**: Byte-level BPE tokenizer (vocab 4,096) guarantees 100% round-trip lossless encoding on all UTF-8 characters without out-of-vocabulary crashes.
3. **Cross-Split Hash Deduplication**: Exact document SHA-256 hashes were compared pairwise across all three splits:
   - Train vs Validation overlap: **0 documents**
   - Train vs Test overlap: **0 documents**
   - Validation vs Test overlap: **0 documents**
4. **Binary Shard Integrity**:
   - Stored in memory-mapped contiguous `uint16` binary format.
   - All token values are within bounds $[0, 4095]$.
   - Shard byte sizes are exact multiples of 2 bytes (no truncated tokens).
5. **Domain Imbalance & Mitigation**:
   - The natural distribution is ~36.9% English, ~31.2% Hindi, ~20.1% Code, ~6.4% Reasoning, ~5.3% Sanskrit, ~0.1% Mathematics.
   - To prevent catastrophic forgetting of minority domains during training, `MultiShardDataset` supports balanced sampling across shard partitions.
