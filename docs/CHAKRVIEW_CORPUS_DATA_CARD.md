# ChakrView Corpus Data Card

**Dataset Title**: ChakrView Indigenous Multi-Domain Pre-Training Corpus  
**Version**: 0.1.0  
**Date**: 2026-09-27  
**Curator**: ChakrView Core Research Initiative  
**Target Architecture**: ChakrMicro v0.1 ($3.44\text{M}$ parameters, $V=4096$)  

---

## 1. Dataset Description & Purpose

### 1.1 Intended Use
- **Foundational Pre-Training**: Pre-training the indigenous, decoder-only causal language model (`ChakrMicro v0.1`) from scratch on local CPU architectures.
- **Bilingual & Polyglot Competence**: Equipping the neural brain with strong, balanced natural language understanding across Hindi (Devanagari script), English (Latin script), and Hinglish (colloquial transliteration and code-switching).
- **Logical & Algorithmic Grounding**: Teaching procedural syntax, programmatic logic (Python, C, Shell), and mathematical notation to prevent purely associative linguistic hallucinations.
- **Edge Deployment Research**: Providing a compact, high-signal, clean training stream tailored for small parameter budgets ($3.44\text{M}$ parameters) operating within tight RAM constraints ($< 1\text{ GB}$).

### 1.2 Non-Intended Use
- **Not for Unchecked Web Scraping**: The dataset must not be populated with uncontrolled, bulk web scrapes.
- **Not for Commercial Chatbot Impersonation**: Not designed for mimicking specific commercial chatbot conversational personas.
- **Not for Exploiting PII or Credentials**: Any presence of personal emails, phone numbers, private keys, or passwords constitutes a critical quality defect.
- **Not for Copyright Infringement**: No copyrighted literary works, paywalled journalism, or proprietary closed-source codebases may be ingested.

---

## 2. Languages & Domain Distribution

### 2.1 Language Coverage
- **Hindi (`hi`)**: Devanagari script; standard grammatical prose, encyclopedic definitions, cultural concepts, literature, and educational texts.
- **English (`en`)**: Latin script; computer science fundamentals, systems engineering, scientific principles, and procedural documentation.
- **Hinglish (`hi-en`)**: Latin script with phonetic Devanagari transliteration; natural dialogue, code-switching technical instructions, and everyday communication.
- **Sanskrit (`sa`)**: Devanagari script; classical Subhashitani, grammatical aphorisms (Paninian patterns), and philosophical etymology.
- **Code (`code`)**: Algorithmic Python, C/C++, Shell scripts, configuration files (JSON, YAML).
- **Mathematics (`math`)**: Arithmetic proofs, algebraic expressions, LaTeX equation formulations.

### 2.2 Domain Balance Target
- English (Technical & Systems): $30.0\%$
- Hindi (Devanagari Cultural & General): $25.0\%$
- Programming Code (Python, C, Systems): $15.0\%$
- Mathematics & Formal Logic: $10.0\%$
- Hinglish (Conversational Code-Switching): $10.0\%$
- Sanskrit & Classical Indic: $4.0\%$
- Structured Multi-Step Reasoning: $4.0\%$
- Numbers, Currencies & Dates: $2.0\%$

---

## 3. Provenance & Licensing Governance

### 3.1 Provenance Requirements
Every individual file admitted into the corpus must trace to one of the following verifiable sources:
1. **Indigenous Project-Authored**: Synthesized directly by the ChakrView development team for specific algorithmic or linguistic training gates.
2. **Public Domain Classics**: Out-of-copyright classical literature, foundational historical texts, and open scientific definitions.
3. **Permissive Open Source Documentation**: Technical documentation released under permissive open-source licenses (MIT, Apache 2.0, BSD-3-Clause).
4. **CC0 Knowledge Repositories**: Data released under Creative Commons Zero 1.0 Universal public domain dedication.

> **STRICT POLICY**: If the provenance or license of a document cannot be verified with certainty, it must be flagged with `provenance: "UNKNOWN"` and excluded from training shards until legal review is completed.

---

## 4. Preprocessing & Quality Filtering

### 4.1 Cleaning & Normalization Pipeline
1. **Line Ending Regularization**: `\r\n` (CRLF) and isolated `\r` are converted to standard `\n` (LF) to guarantee identical cross-platform byte semantics.
2. **Trailing Whitespace Removal**: Superfluous trailing spaces and tabs are stripped.
3. **Lossless Runtime Invariant**: *Zero destructive Unicode normalization* (no NFKC/NFD stripping of Devanagari half-forms, nuktas, or viramas). All Devanagari conjuncts, Zero-Width Joiners (ZWJ: `\u200D`), and Zero-Width Non-Joiners (ZWNJ: `\u200C`) are strictly preserved.
4. **Length Bounding**: Documents must have length $10 \le \text{length} \le 8,000$ characters.

### 4.2 Automated Safety & PII Filtering
All text is subjected to regex-based gatekeeping in `chakrview/corpus/validators.py`:
- **Discards**:
  - Documents containing binary null bytes (`\x00`).
  - Documents containing non-printable ASCII control characters (`0x00-0x08`, `0x0B`, `0x0C`, `0x0E-0x1F`, `0x7F`).
  - Lone surrogate codepoints (`\uD800..\uDFFF`).
  - Private keys (`-----BEGIN PRIVATE KEY-----`), API access tokens, or password strings.
  - Personal identification numbers (Aadhaar, PAN, SSN) and telephone numbers.
- **Flags**:
  - Documents with consecutive character repetition $> 15$ times.
  - Documents exceeding 8,000 characters (flagged for chunking).

---

## 5. Deduplication Strategy

To prevent overfitting, rote memorization, and wasted gradient updates on low-resource hardware:
1. **Exact Deduplication**: SHA-256 hash sets calculated over normalized line content prune exact duplicated lines.
2. **Near-Deduplication**: Lightweight 64-bit SimHash over character 5-grams. Documents sharing Hamming distance $\le 3$ are flagged as near-duplicates and pruned. Memory footprint: 8 bytes per document ($< 1\text{ MB}$ for 100,000 documents).

---

## 6. Evaluation Benchmark Separation & Anti-Leakage

- **Zero Overlap Guarantee**: Training data is partitioned strictly using category-stratified SHA-256 deterministic hashing (`chakrview/corpus/splitter.py`).
- **Disjointness Invariant**:
  $$\text{Train} \cap \text{Validation} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Validation} \cap \text{Test} = \emptyset$$
- **Benchmark Contamination Protection**: Evaluation fixtures under `tests/` and `experiments/` are physically isolated in separate directories and scanned against training shards to prevent data leakage.

---

## 7. Known Limitations & Biases

1. **Vocabulary Density Trade-Off**: With $V=4096$, complex technical jargon or obscure Sanskrit conjuncts decompose into 3–4 byte/subword tokens. Sequence lengths for specialized domains are longer than in models with $V=32000$.
2. **Synthetic Data Prohibition**: The absence of bulk synthetic data limits rapid volume expansion, requiring diligent manual curation of high-quality authentic text.
3. **Domain Representation**: High concentration of technical computing prose (English and Code) requires deliberate counter-weighting with rich Devanagari Hindi literature to preserve bilingual parity.

---

## 8. Versioning & Lineage Tracking

- **Dataset Identifier**: `chakrview-corpus-v0.1.0`
- **Manifest Location**: `data/raw/manifest.json` (SHA-256 tracked)
- **Token Shards Metadata**: `data/tokenized/train/metadata.json` (SHA-256 per shard)
- **Changelog Tracking**: Any addition or modification of raw files triggers automated manifest regeneration and updates the dataset patch version.
