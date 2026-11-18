# UCIP-12: Corpus Engineering & Pipeline Specification

**Mission:** UCIP-12 — Large-Scale Corpus Foundation & Learning Readiness  
**Pipeline Root:** `scripts/stage_c/acquire_and_ingest_stage_c.py`  
**Streaming Shard Reader/Writer:** `chakrview/training/sharding.py`  

---

## 1. Pipeline Architecture

The Stage C corpus ingestion pipeline implements a deterministic, multi-stage transformation from upstream raw archives to production binary tokens:

```
[ Upstream Dumps ] (Wikimedia bz2, GitHub repos, GSM8k JSONL)
        │
        ▼
[ Streaming Decompression & Extraction ]
        │
        ▼
[ Unicode Normalization & Source-Specific Cleaning ]
   - Wikipedia markup, templates, categories stripping
   - Python docstrings/comments preserved; synthetic line numbers removed
   - Devanagari ligatures, viramas, and danda punctuation preserved
        │
        ▼
[ Strict Security & Quality Filters ]
   - UTF-8 validation (reject null bytes, invalid control characters)
   - Secret/PII scanning (regex filters for private keys, API tokens)
   - Bounded length checks (min 50 chars, max 200,000 chars)
        │
        ▼
[ Cryptographic Document Deduplication ]
   - Exact SHA-256 fingerprinting across documents
        │
        ▼
[ Split Assignment ]
   - Deterministic 85% Train, 7.5% Validation, 7.5% Test
        │
        ▼
[ Frozen Tokenizer BPE Encoding (V=4096) ]
   - Byte-level fallback, explicit EOS delimiter appended
        │
        ▼
[ Atomic Binary Sharding ]
   - Little-endian uint16 binary files (~250,000 tokens per shard)
   - Shard integrity validation
```

---

## 2. Invariants & Guarantees

1. **Lossless Tokenization:** Guaranteed by `decode(encode(text)) == text` across all accepted documents.
2. **Deterministic Partitioning:** Random seed fixed ($42$) ensuring zero train-validation leakage across runs.
3. **Atomic Writes:** Shards written to temporary files before atomic rename, preventing truncated shard artifacts.
4. **Memory Efficiency:** Streaming readers process uint16 tokens via memory mapping (`np.memmap` or buffered uint16 byte chunks), allowing multi-million token ingestion on standard CPU laptops without RAM exhaustion.
