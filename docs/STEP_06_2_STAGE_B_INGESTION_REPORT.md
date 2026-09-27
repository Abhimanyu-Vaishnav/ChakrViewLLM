# STEP 6.2 — STAGE B PRE-TRAINING DATA CURATION, CLEANING & INGESTION REPORT

**Author**: ChakrView Engineering & Data Governance  
**Timestamp**: 2026-09-27T15:10:00Z  
**Phase**: Step 6.2 — Stage B Learning-Validation Corpus Ingestion  
**Status**: COMPLETE, AUDITED & READY FOR TRAINING VALIDATION  

---

## 1. Baseline State

Prior to initiating Step 6.2, repository state was verified against git commit `dd295a7` (`Step 6.1: Ratify corpus engineering specification, data governance, and state audit`):
- **Operating Environment**: Windows 11, Python 3.14.7, PyTorch 2.14.0+cpu, Intel Core i9-13900H CPU, 64GB RAM.
- **Baseline Test Suite**: 235 passed, 0 failures, 0 errors.
- **Neural Core (FROZEN)**: ChakrMicro v0.1 ($3,443,136$ parameters, 6 layers, $d_{\text{model}}=192$, 6 heads, $d_{\text{ff}}=512$, context 512, weight tying, Pre-RMSNorm, SwiGLU, RoPE, bias-free).
- **Tokenizer (FROZEN)**: Byte-Level BPE, $V=4096$, BOS=0, EOS=1, PAD=2, learned merges=3837, checksum `7498d92adeef7c9e64d4d1bbd5b4b23f8774692d868742e3b98e30c584c4403f`. Lossless invariant $\text{Decode}(\text{Encode}(x)) \equiv x$ strictly verified.
- **Corpus Infrastructure**: Step 6.1 modules (`loader.py`, `validators.py`, `cleaner.py`, `normalizer.py`, `splitter.py`, `statistics.py`, `manifest.py`, `sharding.py`, `dataset.py`).

---

## 2. Sources & Categories Included

The Stage B corpus combines existing baseline control documents with curated open material across 9 categories and 27 files in `data/raw/`:

| Category | Source Files in `data/raw/<cat>/` | Domain Description |
| :--- | :--- | :--- |
| **English** | `stage_b_systems.txt`, `ai.txt`, `general.txt`, `science.txt` | Computer architecture, OS kernels, distributed systems, compilers, networking, robotics, design patterns, physics, biology. |
| **Code** | `stage_b_algorithms.txt`, `c_lang.txt`, `python.txt`, `sql.txt` | Classical algorithms (Quicksort, KMP, BFS, Dijkstra, DSU, Trie), PyTorch neural layers (SwiGLU, RMSNorm, RoPE), C kernels, SQL. |
| **Hindi** | `stage_b_literature.txt`, `culture.txt`, `general.txt`, `literature.txt` | Computer science in Hindi, Indian history (Indus Valley, Mauryas, Cholas, Marathas), Premchand literature, षड्दर्शन, scientists, freedom struggle, linguistics. |
| **Mathematics** | `stage_b_derivations.txt`, `formulas.txt`, `proofs.txt`, `theorems.txt` | Linear algebra, spectral theorem, SVD, Central Limit Theorem, Bayes' theorem, complex analysis, calculus proofs, matrix analysis. |
| **Hinglish** | `stage_b_dialogues.txt`, `chat.txt`, `tech.txt` | Natural engineering dialogues on model architecture, profiling, cache miss analysis, Docker, CI/CD, Kafka, database indexing. |
| **Reasoning** | `stage_b_procedures.txt`, `formal.txt`, `procedural.txt` | Procedural incident playbooks (memory fragmentation, distributed locks, cache thrashing, SYN flood) and formal syllogisms. |
| **Sanskrit** | `stage_b_classics.txt`, `grammar.txt`, `shlokas.txt` | Bhagavad Gita verses with anvaya, Panini Ashtadhyayi sutras with vyakhya, Vidura Neeti, Bhartrihari Neeti Shatakam subhashitas. |
| **Numbers** | `stage_b_tables.txt`, `data.txt`, `tables.txt` | Structured ledgers (INR ₹, USD $, EUR €), city demographics, hardware metrics, coordinates. |
| **Mixed** | `general.txt` | Step 3 multi-script baseline control samples. |

---

## 3. License & Provenance Summary

- **License Classes**: Strictly open source and permissive (`CC0-1.0`, `MIT`, `Public Domain`).
- **Provenance**: Project-authored open technical documentation, formal mathematical theorem formulations, public domain classical literature excerpts, and textbook algorithmic implementations.
- **Ethical & Safety Guardrails**: Zero private personal data, zero phone numbers, zero personal IDs, zero credentials/API keys, zero automated web-scraping boilerplate.

---

## 4. Cleaning & Unicode Semantics

Text cleaning was executed via `clean_training_text()` adhering strictly to the lossless contract:
- Replaced non-semantic `\r\n` and `\r` with standard `\n`.
- Stripped trailing whitespace per line.
- Preserved internal spaces, tabs, and exact Unicode semantics.
- Preserved Devanagari combining marks (matras, anusvara, visarga, nukta, halant) and formatting controls (ZWJ `\u200d`, ZWNJ `\u200c`).
- Zero destructive Unicode normalization (e.g. no lossy NFKD folding).

---

## 5. Exact Deduplication & Data Quality Results

Exact SHA-256 deduplication and data quality checks were performed via `validate_corpus_collection()`:

```
Checked Documents:    9,029
Valid Documents:      9,002 (99.70%)
Exact Duplicates:        26 ( 0.28%)
Discarded Documents:      1 ( 0.01%)
Flagged Documents:       13 ( 0.14%)
```

### Breakdown of Violations & Actions
1. **EXTREMELY_SHORT (1 document, DISCARDED)**: Intentional single-character test document from Step 3 test corpus (`length < min_chars=2`).
2. **EXACT_DUPLICATE (26 documents, DISCARDED)**: Identical repeated lines across files (e.g., standard table headers and common syntax delimiters). Discarded deterministically with zero loss of distinct content.
3. **EXCESSIVE_REPEATED_CHARS (13 documents, FLAGGED & PRESERVED)**: Markdown tabular divider lines (`| :--- | :---: |`) exceeding the 15-character threshold. Preserved safely.
4. **BINARY_NULL_BYTE / UNEXPECTED_CONTROL_CHARS / SURROGATE_CODEPOINT**: **0 detected**. 100% valid UTF-8.

---

## 6. Character, Byte & Script Statistics

Computed via `compute_corpus_statistics()` and saved to `data/statistics/stage_b_corpus_statistics.json`:
- **Total Valid Documents**: 9,002
- **Total Characters**: 1,182,829
- **Total UTF-8 Bytes**: 1,555,115
- **Average Bytes per Character**: 1.3147
- **Unique Unicode Codepoints**: 230
- **Codepoint Range**: `0x20` (Space) to `0x1f680` (Rocket Emoji)

### Script Breakdown
- **Latin / ASCII**: 79.2% (English prose, code tokens, LaTeX math syntax, numbers)
- **Devanagari**: 19.8% (Hindi literature, Sanskrit verses, Devanagari grammar)
- **Mathematical Symbols & Greek**: 0.8%
- **Combining Marks & Punctuation**: 0.2%

---

## 7. Tokenization & Actual Domain Distribution

All tokens were counted using the **actual frozen Byte-Level BPE tokenizer** ($V=4096$). Zero estimation or word heuristics were used.

### Actual Achieved Distribution vs Experimental Target

| Domain / Category | Measured Valid Tokens | Measured % | Step 6.1 Policy Target % | Variance |
| :--- | :---: | :---: | :---: | :---: |
| **English** | 123,789 | 25.13% | 30.0% | -4.87% |
| **Code** | 94,628 | 19.21% | 15.0% | +4.21% |
| **Hindi** | 90,731 | 18.42% | 25.0% | -6.58% |
| **Mathematics** | 67,174 | 13.64% | 10.0% | +3.64% |
| **Hinglish** | 48,921 | 9.93% | 10.0% | -0.07% |
| **Reasoning** | 30,436 | 6.18% | 4.0% | +2.18% |
| **Sanskrit** | 22,336 | 4.53% | 4.0% | +0.53% |
| **Numbers / Data** | 13,539 | 2.75% | 2.0% | +0.75% |
| **Mixed (Baseline)**| 1,097 | 0.22% | 0.0% | +0.22% |
| **Total Content Tokens** | **492,651** | **100.00%** | **100.0%** | — |

**Total Target**: ~500,000 tokens.  
**Total Measured Content Tokens**: **492,651 tokens** (-1.47% from target).  
**Total Sharded Tokens (Content + `<EOS>` delimiters)**: **501,653 tokens** (+0.33% from target).

---

## 8. Deterministic Splits & Disjointness

Corpus partitioned using `partition_corpus(seed=42)` across categories:
- **Train (80%)**: 7,258 documents | 397,986 content tokens | 405,244 sharded tokens
- **Validation (10%)**: 891 documents | 47,636 content tokens | 48,527 sharded tokens
- **Test (10%)**: 853 documents | 47,029 content tokens | 47,882 sharded tokens

### Disjointness Verification
- `train_texts.isdisjoint(val_texts)` $\to$ **True**
- `train_texts.isdisjoint(test_texts)` $\to$ **True**
- `val_texts.isdisjoint(test_texts)` $\to$ **True**
- Zero document overlap across splits.

---

## 9. Production Binary Shards & Integrity

Shards written via `ShardWriter` using atomic two-phase write (`.tmp` $\to$ `os.replace`):
- **Format**: Contiguous `uint16` little-endian (2 bytes per token, $V=4096$)
- **Max tokens per shard**: 250,000
- **Document Boundary**: `<EOS>` (ID 1) appended after every document

### Shard Accounting & Integrity

| Split | Filename | Token Count | Byte Size | SHA-256 Checksum | Integrity Status |
| :--- | :--- | :---: | :---: | :--- | :---: |
| **train** | `shard_00000.bin` | 250,000 | 500,000 B | `4c8d5c43d2c8845bc2ca718b5b5b1adbe5eebebb45ec46f183dc87593c200dd0` | **PASS** |
| **train** | `shard_00001.bin` | 155,244 | 310,488 B | `cb7efef59d1ca542d992f808cc97eb1df0a7fc972d37923769c8989a318269e3` | **PASS** |
| **validation** | `shard_00000.bin` | 48,527 | 97,054 B | `325fe73be906ba58d3434ea9c072b25b68cecb7e3240f9bb6702c2e0bb1ca830` | **PASS** |
| **test** | `shard_00000.bin` | 47,882 | 95,764 B | `c484f9a06ca27d6d3936998634e2c943df43f309a06f3630f91ce072120e7d58` | **PASS** |

Total Sharded Tokens: **501,653 tokens** across 4 binary files ($1,003,306$ bytes).  
`verify_shard_integrity()` returned **True** across all three split directories.

---

## 10. Verification of Frozen Contracts

1. **Neural Core Architecture**:
   - Zero changes to `chakrview/brain/`
   - Parameters: bit-exact $3,443,136$
   - Context window: 512 tokens
2. **Tokenizer Contract**:
   - Zero changes to `chakrview/tokenizer/`
   - Vocabulary: bit-exact $V=4096$
   - Special tokens: BOS=0, EOS=1, PAD=2
   - Lossless invariant verified: $\text{Decode}(\text{Encode}(x)) \equiv x$
   - Every token ID in shard stream strictly satisfies $0 \le \text{ID} < 4096$
3. **Atomic Sharding Contract**:
   - `ShardWriter` verified atomic writing with temporary `.tmp` files and `os.replace`
   - Zero partially written or corrupt files can survive write interruption

---

## 11. Testing & Validation Results

```
============================= test session starts =============================
platform win32 -- Python 3.14.7, pytest-9.1.1, pluggy-1.6.0
collected 241 items

tests/test_bpe_trainer.py .................................. [PASSED]
tests/test_brain_causality.py .............................. [PASSED]
tests/test_brain_causality_strict.py ....................... [PASSED]
tests/test_brain_config.py ................................. [PASSED]
tests/test_brain_gradient.py ............................... [PASSED]
tests/test_brain_gradient_flow_strict.py ................... [PASSED]
tests/test_brain_initialization.py .......................... [PASSED]
tests/test_brain_mathematical_validation.py ................ [PASSED]
tests/test_brain_model.py .................................. [PASSED]
tests/test_brain_primitives.py ............................. [PASSED]
tests/test_brain_shapes_contract.py ........................ [PASSED]
tests/test_brain_synthetic_learnability.py ................. [PASSED]
tests/test_checkpoint.py ................................... [PASSED]
tests/test_collator.py ..................................... [PASSED]
tests/test_corpus_engine.py ................................ [PASSED]
tests/test_corpus_pipeline.py .............................. [PASSED]
tests/test_dataset.py ...................................... [PASSED]
tests/test_evaluation.py ................................... [PASSED]
tests/test_loss.py ......................................... [PASSED]
tests/test_neural_core_spec.py ............................. [PASSED]
tests/test_optimizer.py .................................... [PASSED]
tests/test_resume.py ....................................... [PASSED]
tests/test_seed.py ......................................... [PASSED]
tests/test_stage_b_ingestion.py ............................ [PASSED]
tests/test_step3_tokenizer_benchmark.py .................... [PASSED]
tests/test_tokenizer_adversarial.py ........................ [PASSED]
tests/test_tokenizer_baseline.py ........................... [PASSED]
tests/test_tokenizer_benchmark.py .......................... [PASSED]
tests/test_tokenizer_bpe_engine.py ......................... [PASSED]
tests/test_tokenizer_bytes.py .............................. [PASSED]
tests/test_tokenizer_determinism.py ........................ [PASSED]
tests/test_tokenizer_interface.py .......................... [PASSED]
tests/test_tokenizer_serialization.py ...................... [PASSED]
tests/test_tokenizer_special_tokens.py ..................... [PASSED]
tests/test_tokenizer_utf8_lossless.py ...................... [PASSED]
tests/test_trainer.py ...................................... [PASSED]
tests/test_training_config.py .............................. [PASSED]

============================ 241 passed in 14.72s =============================
```

- **Baseline Tests**: 235 passed
- **New Tests Added**: 6 passed
  - `tests/test_dataset.py::test_shard_writer_atomic_write`
  - `tests/test_stage_b_ingestion.py::test_stage_b_manifest_contract`
  - `tests/test_stage_b_ingestion.py::test_stage_b_shard_integrity`
  - `tests/test_stage_b_ingestion.py::test_stage_b_shard_binary_contract`
  - `tests/test_stage_b_ingestion.py::test_stage_b_streaming_dataset_compatibility`
  - `tests/test_stage_b_ingestion.py::test_stage_b_processed_split_disjointness`
- **Total Passing Tests**: **241 / 241 (100% GREEN, 0 FAILURES, 0 ERRORS)**

---

## 12. Reproducibility Instructions

To reproduce the complete Step 6.2 ingestion, validation, and sharding process:
```powershell
# 1. Populate/verify raw stage B source documents
.venv\Scripts\python.exe scripts/populate_stage_b_corpus.py

# 2. Run complete end-to-end ingestion, validation, and sharding pipeline
.venv\Scripts\python.exe scripts/run_stage_b_ingestion.py

# 3. Execute the full test battery
.venv\Scripts\pytest.exe -v
```

---

## 13. Scope Boundaries & Limitations

1. **Learning-Validation Scale Only**: This corpus contains **501,653 sharded tokens** and is intended specifically for **Stage B learning validation** (verifying that the ChakrMicro neural core can learn real multi-domain text distributions on CPU). It is NOT full-scale pre-training (10M–50M tokens).
2. **CPU-First Low-Resource Execution**: All loaders and dataset readers stream sequentially, keeping peak process memory usage below 250MB RAM.
3. **No Weights or Tokenizer Modified**: Architectural freezing was 100% maintained throughout this substep.
