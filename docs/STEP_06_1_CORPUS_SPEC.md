# ChakrView — Step 6.1: Pre-Training Corpus Engineering & Data Governance Specification

**Document Version**: 1.0.0  
**Phase**: Step 6.1 — Corpus Engineering Specification & Data Governance  
**Status**: RATIFIED & ARCHITECTURE-FROZEN  
**Target Architecture**: ChakrMicro v0.1 (3,443,136 parameters, $N=6, d_{\text{model}}=192, H=6, d_{\text{ff}}=512, T_{\text{max}}=512, V=4096$)  
**Tokenizer Contract**: Frozen Byte-Level BPE ($V=4096, \langle\text{BOS}\rangle=0, \langle\text{EOS}\rangle=1, \langle\text{PAD}\rangle=2$, 256 byte primitives)  

---

## 1. Executive Summary & Core Purpose

ChakrView is an indigenous, modular, low-resource artificial intelligence brain designed from first principles. Its architectural goal is to create a reusable neural foundation that operates strictly on local CPU hardware without reliance on external pretrained weights, opaque web-scraped dumps, or third-party proprietary frameworks.

Step 6 marks the transition from architectural verification (Steps 0–5) to empirical pre-training. Step 6.1 formally establishes the **Corpus Engineering Specification, Data Governance Framework, Quality Gates, Provenance Manifests, Deduplication Policy, and Shard Contracts**.

### Mandatory Data Invariant
> **CORPUS PURITY INVARIANT**:  
> 1. No copyrighted books, paywalled articles, or private personal communications may enter the training pipeline.  
> 2. No scraped web dumps containing unverified credentials, secrets, or tracking telemetry.  
> 3. Zero reliance on synthetic outputs from proprietary commercial LLMs.  
> 4. All data sources must possess verifiable provenance and permissive licensing (CC0, MIT, Apache 2.0, Public Domain).  
> 5. Tokenizer correctness and byte lossless reconstruction strictly supersede compression ratios.

---

## 2. Status of Existing Corpus Infrastructure

A forensic audit of the active repository confirms that the core corpus engineering pipeline is already implemented, tested, and passing all verification suites:

| Subsystem Component | Module Location | Status | Verified Functionality |
| :--- | :--- | :---: | :--- |
| **Document Loader** | [`chakrview/corpus/loader.py`](file:///d:/Project/ChakrView/chakrview/corpus/loader.py) | **Production** | Streaming and tree-based document reader yielding structured `CorpusDocument` records with line indices and SHA-256 hashes. |
| **Quality Validator** | [`chakrview/corpus/validators.py`](file:///d:/Project/ChakrView/chakrview/corpus/validators.py) | **Production** | Strict detection of empty documents, null bytes, illegal control characters, lone surrogate codepoints, length violations, repeated characters, and whitespace anomalies. |
| **Document Cleaner** | [`chakrview/corpus/cleaner.py`](file:///d:/Project/ChakrView/chakrview/corpus/cleaner.py) | **Production** | Stage-aware regularizer preserving exact Devanagari ligatures, ZWJ/ZWNJ characters, and Unicode byte values while standardizing linebreaks (CRLF $\to$ LF). |
| **Unicode Normalizer** | [`chakrview/corpus/normalizer.py`](file:///d:/Project/ChakrView/chakrview/corpus/normalizer.py) | **Production** | Enforces the foundational invariant: *Zero destructive Unicode normalization at runtime inference* (strictly lossless identity). |
| **Deterministic Splitter** | [`chakrview/corpus/splitter.py`](file:///d:/Project/ChakrView/chakrview/corpus/splitter.py) | **Production** | SHA-256 category-stratified deterministic partitioner (80% train, 10% validation, 10% test) with mathematical disjointness verification. |
| **Statistical Profiler** | [`chakrview/corpus/statistics.py`](file:///d:/Project/ChakrView/chakrview/corpus/statistics.py) | **Production** | Computes Unicode codepoint bounds, script distribution (Devanagari, Latin, Digits, Math, Emoji, Whitespace), whitespace frequencies, and duplicate rates. |
| **Manifest Generator** | [`chakrview/corpus/manifest.py`](file:///d:/Project/ChakrView/chakrview/corpus/manifest.py) | **Production** | Generates machine-readable JSON manifests containing file sizes, line counts, SHA-256 hashes, source attribution, and licensing tags. |
| **Binary Shard Writer** | [`chakrview/training/sharding.py`](file:///d:/Project/ChakrView/chakrview/training/sharding.py) | **Production** | Compact `uint16` little-endian binary packer generating `shard_*.bin` files with `metadata.json` checksum tracking. |
| **Streaming Dataset** | [`chakrview/training/dataset.py`](file:///d:/Project/ChakrView/chakrview/training/dataset.py) | **Production** | Memory-bounded causal sequence reader yielding $[T]$ input/target pairs on demand without loading whole corpora into RAM. |
| **Causal Batch Collator**| [`chakrview/training/collator.py`](file:///d:/Project/ChakrView/chakrview/training/collator.py) | **Production** | Stacks token tensors into $[B, T]$ batches with strict $T \le 512$ boundary enforcement and PAD masking. |

---

## 3. Audit of Existing Corpus Data

Audit of `data/raw/` and active manifests:

### 3.1 Measured Data Summary
- **Physical Files in `data/raw/`**: 19 text files across 8 domain subdirectories.
- **Total Physical Byte Size**: $61,592\text{ bytes}$ ($60.15\text{ KB}$).
- **Total Character Count**: $43,023\text{ characters}$.
- **Total Document Lines**: $609\text{ documents}$.
- **Quality Audit Findings**:
  - `valid_documents`: $600$ ($98.5\%$)
  - `duplicate_documents`: $8$ exact duplicate lines (flagged and safely discardable)
  - `discarded_documents`: $1$ document ($< 2$ characters: `EXTREMELY_SHORT`)
  - `flagged_documents`: $5$ documents ($> 15$ repeated characters)
  - `empty_files`: $0$
  - `invalid_utf8_files`: $0$
  - `corrupted_files`: $0$
- **Measured Script Distribution**:
  - Latin (ASCII): $52.16\%$
  - Devanagari (Hindi / Sanskrit): $21.74\%$
  - Whitespace (semantic): $15.58\%$
  - Punctuation: $6.06\%$
  - ASCII Digits: $3.43\%$
  - Mathematical Symbols: $0.95\%$
  - Zero-Width / Combining Marks / Other: $0.08\%$

### 3.2 Generated Artifacts Isolation Notice
The following directories in `data/` are compiled intermediate artifacts and **MUST NEVER** be ingested back into the raw training corpus:
- `data/processed/` (derived train/test splits)
- `data/validation/` (held-out evaluation fixtures)
- `data/tokenized/` (compiled binary shards)
- `data/statistics/` (JSON profiling summaries)
- `data/tokenizer_experiments/` and `data/experiments/` (candidate BPE merge matrices)

---

## 4. ChakrView Multi-Domain Corpus Mix Specification

ChakrView is architected as a foundational cognitive engine. To support diverse reasoning, bilingual literacy, algorithmic understanding, and mathematical stability, the corpus mix is structured across eight primary domains.

### Explicit Data Classification Legend
- **[MEASURED DATA]**: Empirically measured properties from the existing 609-document research corpus.
- **[PLANNED DATA]**: Targets established for Stage B (learning validation) and Stage C (initial pre-training).
- **[EXPERIMENTAL]**: Domain ratios subject to empirical perplexity and loss balancing during training.
- **[HYPOTHESIS]**: Theoretical downstream capabilities expected from specific domain inclusions.
- **[FUTURE DATA]**: Extended domains reserved for Stage D scaling.

### Proposed Initial Corpus Domain Mix

| Domain Category | Sub-categories / Focus Areas | Target Share (%) | Status | Justification / Role in Brain |
| :--- | :--- | :---: | :---: | :--- |
| **1. Hindi (Devanagari)** | Literature, science, grammar, dialogic exchange, foundational culture, encyclopedic facts | **$25.0\%$** | *[PLANNED / EXPERIMENTAL]* | Core indigenous identity; ensures native high-fertility Devanagari representation without translation bottleneck. |
| **2. English (Academic/Systems)** | Computer science, operating systems, hardware architecture, science, encyclopedic knowledge | **$30.0\%$** | *[PLANNED / EXPERIMENTAL]* | Foundational global technical vocabulary, grammar structure, and high-resource knowledge grounding. |
| **3. Hinglish / Code-Switching** | Natural colloquial bilingual conversational text, technical discussions, transliteration | **$10.0\%$** | *[PLANNED / EXPERIMENTAL]* | High-frequency colloquial Indian conversational medium; bridges Latin script and Devanagari concepts smoothly. |
| **4. Programming Code** | Python, C/C++, Shell/Bash, configuration schemas (JSON, YAML), algorithmic logic | **$15.0\%$** | *[PLANNED / EXPERIMENTAL]* | Instills strict syntax discipline, scoping, indentation semantics, and deterministic state transitions. |
| **5. Mathematics & Logic** | Arithmetic steps, proofs, boolean algebra, LaTeX formulas, numerical equations | **$10.0\%$** | *[PLANNED / EXPERIMENTAL]* | Mitigates token-boundary hallucinations; grounds arithmetic token sequences in causal chains. |
| **6. Sanskrit / Classical Texts** | Subhashitani, grammatical structures (Paninian rules), etymology, classical literature | **$4.0\%$** | *[PLANNED / EXPERIMENTAL]* | Highly regular morphological and syntactic structures; enriches root-word representations and compound logic. |
| **7. Structured Reasoning** | Deductive logic puzzles, procedural explanations, step-by-step troubleshooting | **$4.0\%$** | *[PLANNED / EXPERIMENTAL]* | Enhances multi-step token prediction coherence within a compact 512-token context window. |
| **8. Structured Data & Numbers** | Tabular records, dates, financial notations, scientific measurements, currencies | **$2.0\%$** | *[PLANNED / EXPERIMENTAL]* | Hardens digit stability and prevents catastrophic failure on dates, currencies ($\text{₹}, \$, €$), and units. |

---

## 5. Data Quality Policy

All incoming data must pass through deterministic automated quality filters before admission to the training split.

### 5.1 ACCEPT Criteria (All must be TRUE)
1. **Valid UTF-8**: Byte sequence decodes cleanly without replacement characters (`\uFFFD`) or lone surrogates (`\uD800..\uDFFF`).
2. **Character Density**: Printable characters constitute $\ge 80\%$ of total document characters.
3. **Length Bounding**: Document character length satisfies $10 \le \text{len} \le 8,000$ characters (preventing empty stubs and context-overflowing monolithic lines).
4. **Verifiable Provenance**: Source repository, author, or publication origin is documented in `manifest.json`.
5. **Permissive Licensing**: Source text is released under CC0, Public Domain, MIT, Apache 2.0, or project-authored open documentation.
6. **Syntactic Validity (for Code)**: Python code parses via standard AST or token scanner; JSON/YAML configs parse cleanly.

### 5.2 REJECT Criteria (Any triggers immediate DISCARD)
1. **Accidental Binary / Null Bytes**: Presence of `\x00` or non-text control characters (`0x00-0x08`, `0x0B`, `0x0C`, `0x0E-0x1F`, `0x7F`).
2. **Personal Identifiable Information (PII)**:
   - Direct personal phone numbers, Aadhaar/SSN IDs, home addresses, private email lists.
   - Any document matching PII regex heuristics is discarded immediately.
3. **Secrets, Credentials & Private Keys**:
   - RSA/OpenSSH private key blocks (`-----BEGIN RSA PRIVATE KEY-----`).
   - High-entropy API tokens (AWS keys, GitHub personal access tokens, OpenAI/Anthropic/Google API keys, password strings).
4. **Excessive Repetition**: Any single character repeated consecutively $> 15$ times (e.g., `!!!!!!!!!!!!!!!!`), or identical lines repeated $> 3$ times within a document.
5. **Automated Web Spam & Scraper Cruft**: Cookie notices, navigation footers, SEO link farms, boilerplate legal disclaimers.
6. **Corrupted Devanagari**: Incomplete UTF-8 sequences that split multi-byte Indic codepoints, leading to corrupted glyph fragments.

---

## 6. Provenance & Manifest Specification

Every raw text file admitted into `data/raw/` must possess a cryptographically verifiable entry in `data/raw/manifest.json`.

### Authoritative Document Metadata Schema
```json
{
  "document_id": "string (UUIDv4 or deterministic category:index)",
  "source_id": "string (unique identifier of upstream corpus repository)",
  "source_type": "string (e.g., 'indigenous_authored', 'public_domain_classics', 'permissive_open_source')",
  "path": "string (relative POSIX path inside data/raw/)",
  "language": "string (ISO 639-1 code: 'hi', 'en', 'sa', 'hi-en', 'code', 'math')",
  "script": "string ('Devanagari', 'Latin', 'ASCII_Code', 'Mixed')",
  "category": "string ('hindi', 'english', 'hinglish', 'sanskrit', 'code', 'mathematics', 'numbers', 'mixed')",
  "license": "string (SPDX identifier: 'CC0-1.0', 'MIT', 'Apache-2.0', 'Public-Domain', or 'UNKNOWN')",
  "provenance": "string (detailed textual origin; must be 'UNKNOWN' if undocumented)",
  "sha256": "string (64-character lowercase hex SHA-256 hash of raw byte content)",
  "byte_count": "integer (exact byte size on disk)",
  "character_count": "integer (Unicode codepoint count)",
  "token_count": "integer (computed with frozen V=4096 tokenizer)",
  "quality_status": "string ('VALIDATED_CLEAN', 'FLAGGED_REVIEW', 'DISCARDED')",
  "duplicate_group": "string (hash of normalized text for exact duplicate grouping)",
  "ingestion_timestamp": "string (ISO 8601 UTC timestamp)",
  "preprocessing_version": "string ('0.1.0')",
  "tokenizer_version": "string ('ChakrMicro-BPE-V4096-v0.1')"
}
```

> **GOVERNANCE RULE**: Never fabricate or assume licensing or provenance. If a document's legal origin is unverified, its `license` and `provenance` fields must be marked strictly as `"UNKNOWN"`, and it will be barred from production pre-training shards until reviewed.

---

## 7. Low-Resource Deduplication Strategy

To maintain high learning efficiency and avoid memorization pathologies, deduplication is executed at two distinct levels optimized for low-RAM CPU hardware.

### 7.1 Level 1: Exact Deduplication (Zero Memory Footprint)
- **Algorithm**: SHA-256 hash computed over normalized line content (`text.strip()`).
- **Implementation**: Streaming set of 64-bit truncated hash keys or disk-backed SQLite/Bloom filter.
- **Complexity**: $O(N)$ time, $\sim 8\text{ bytes}$ per document in RAM.
- **Action**: All exact duplicates are eliminated from the training split.

### 7.2 Level 2: Near-Deduplication (Low-RAM CPU SimHash)
Traditional MinHash LSH across massive corpora can demand tens of gigabytes of RAM. For ChakrView's edge-conscious design:
- **Algorithm**: **64-bit Char-NGram SimHash**:
  1. Extract character 5-grams (or token 3-grams) across document text.
  2. Hash each n-gram to a 64-bit integer using MurmurHash3 / FNV-1a.
  3. Aggregate positive and negative bit contributions into a 64-dimensional accumulator vector.
  4. Form a 64-bit fingerprint where bit $i = 1$ if $\text{acc}[i] > 0$, else $0$.
- **Storage Requirement**: Exactly $8\text{ bytes}$ per document. For a 100,000 document corpus, the entire fingerprint table occupies only **$800\text{ KB}$ of RAM**, fitting entirely inside CPU L2 cache!
- **Matching Criterion**: Documents with Hamming distance $\le 3$ (out of 64 bits) are classified as near-duplicates.
- **Resolution**: The shorter or lower-quality document is pruned, or flagged for inspection.

---

## 8. Data Leakage & Evaluation Isolation Policy

To guarantee empirical integrity, the pre-training corpus must never contaminate downstream evaluation benchmarks:
1. **Physical Isolation**:
   - `data/raw/` contains only training-eligible source materials.
   - `data/validation/` and benchmark fixtures (`experiments/tokenizer/fixtures/`) are physically segregated.
2. **String Contamination Scanner**:
   - Before compiling training shards, an automated scanner checks that no 64-character n-gram from evaluation fixtures (`tests/`, `experiments/`) appears in the training text.
3. **Synthetic Output Prohibition**:
   - Model-generated completions must never be recycled into training text without human verification, preventing self-reinforcing model collapse.

---

## 9. Staged Pre-Training Data Budgets

To prevent premature over-allocation of compute, training data is staged into progressive milestones:

| Stage Milestone | Target Token Count | Shard Count ($250\text{k}$ tok/shard) | Binary Disk Footprint | Expected CPU Training Duration | Primary Engineering Objective |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Stage A: Smoke Corpus** | **$10,000$ tokens** | 1 partial shard | $\sim 20\text{ KB}$ | $< 5\text{ seconds}$ | Validate end-to-end shard streaming, loss backprop, and checkpointing. *[MEASURED / VERIFIED]* |
| **Stage B: Learning Validation** | **$500,000$ tokens** | 2 binary shards | $\sim 1.0\text{ MB}$ | $\sim 2.0\text{ minutes}$ | Verify sustained loss decay ($8.3 \to 5.5$) and associative recall on real bilingual text. *[PLANNED]* |
| **Stage C: Initial Pre-Training** | **$10,000,000$ tokens** | 40 binary shards | $\sim 20.0\text{ MB}$ | $\sim 40\text{ minutes}$ | First true pre-training run; establish foundational n-gram and syntactic representations. *[TARGET]* |
| **Stage D: Scaled Pre-Training** | **$50,000,000$ tokens** | 200 binary shards | $\sim 100.0\text{ MB}$ | $\sim 3.5\text{ hours}$ | Deep representation learning across all 8 domains on local Intel i9 CPU. *[FUTURE / HYPOTHESIS]* |

---

## 10. Binary Shard Format Contract

The shard format implemented in `chakrview/training/sharding.py` is ratified as the production standard:

### Contract Specifications
1. **Data Type**: `numpy.uint16` little-endian binary array (`shard_00000.bin`).
2. **Storage Density**: Exactly 2 bytes per token for $V=4096$ ($0 \le \text{ID} < 4096$).
3. **Shard Size Limit**: Configurable, default $250,000\text{ tokens}$ per file ($500\text{ KB}$ per shard). Keeps individual I/O read operations within single OS disk block buffers.
4. **Document Boundary Handling**: Documents are appended sequentially. Callers MUST append `<EOS>` ($\text{ID}=1$) to delineate document boundaries.
5. **Checksum Tracking**: Every shard is hashed with SHA-256 upon write and recorded in `metadata.json`.
6. **Integrity Verification**: `verify_shard_integrity(shard_dir)` validates every binary file against its recorded checksum before training commences.
7. **Zero RAM Pressure**: `StreamingTokenDataset` reads binary shards sequentially, maintaining an in-memory buffer of only 1 shard at a time.

---

## 11. Resource Budget for Low-Resource Hardware

Corpus engineering and shard compilation must execute comfortably on resource-constrained consumer hardware:

- **Host RAM Ceiling**: Process RSS $\le 1.0\text{ GB}$ (Total system RAM $\le 2.0\text{ GB}$).
- **CPU Threading**: Configurable worker pools; streaming I/O prevents GIL deadlocks.
- **Disk Space Requirement**:
  - Raw text for 50M tokens: $\sim 150\text{ MB}$.
  - Tokenized `uint16` binary shards for 50M tokens: exactly **$100\text{ MB}$**.
  - Total working disk space: $< 500\text{ MB}$ (including temporary buffers).
- **Execution Portability**: 100% native Python and NumPy; zero external C-extension or CUDA runtime dependencies.

---

## 12. Corpus Governance Configuration Schema

Corpus governance settings are formalized into the following configuration schema:

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class CorpusGovernanceConfig:
    """Authoritative configuration for corpus curation and governance."""
    corpus_version: str = "0.1.0"
    random_seed: int = 42
    
    # Target domain weights (percentages summing to 100.0)
    domain_weights: Dict[str, float] = field(default_factory=lambda: {
        "hindi": 25.0,
        "english": 30.0,
        "hinglish": 10.0,
        "code": 15.0,
        "mathematics": 10.0,
        "sanskrit": 4.0,
        "reasoning": 4.0,
        "numbers": 2.0,
    })
    
    # Partition ratios
    train_ratio: float = 0.80
    validation_ratio: float = 0.10
    test_ratio: float = 0.10
    
    # Quality filter thresholds
    min_doc_chars: int = 10
    max_doc_chars: int = 8000
    max_repeated_chars: int = 15
    allow_unknown_license: bool = False
    
    # Deduplication
    exact_dedup: bool = True
    simhash_near_dedup: bool = True
    simhash_max_hamming_distance: int = 3
    
    # Sharding
    tokens_per_shard: int = 250_000
    token_dtype: str = "uint16"
```

---

## 13. Ratification & Next Steps

This specification formally governs all corpus curation, ingestion, and preprocessing for Step 6.  
- **Ratification Date**: 2026-09-27  
- **Approved Substep**: Step 6.1 complete.  
- **Pending Action**: Step 6.2 (Curate, clean, and validate Stage B training text data in `data/raw/`).
