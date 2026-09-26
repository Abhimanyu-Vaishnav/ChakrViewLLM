# ChakrView Tokenizer Research Corpus

**Version**: 0.1.0  
**Phase**: Step 2.3 — Corpus Engineering & Vocabulary Experimentation  
**License**: Project Internal Research & Synthetic Creative Work (Permissively Generated / MIT)  

---

## 1. Corpus Purpose

This corpus is designed **exclusively for tokenizer engineering and BPE vocabulary candidate evaluation**.
Its purpose is to:
1. Provide a controlled, reproducible text distribution to train candidate BPE vocabularies ($V \in \{2048, 4096, 8192, 16384\}$).
2. Empirically measure compression ratios, token fertility, and encode/decode latencies across linguistic and technical categories.
3. Validate the lossless reconstruction invariant:
   $$\text{Decode}(\text{Encode}(\text{text})) \equiv \text{text}$$
   on the entire dataset.

---

## 2. Origin & Provenance of Categories

| File | Category | Origin / Provenance | Synthetic Status |
| :--- | :--- | :--- | :--- |
| `hindi.txt` | Devanagari Hindi | Manually curated synthetic sentences covering everyday conversation, technical explanations, classical literature excerpts, philosophy, and geography. | 100% Synthetic / Manually authored |
| `english.txt` | Standard English | Technical computer architecture descriptions, hardware manuals, programming guides, and conversational text. | 100% Synthetic / Manually authored |
| `hinglish.txt` | Romanized Hindi | Phonetically variable colloquial Hindi in Latin script, covering everyday technical conversation, chat, and system questions. | 100% Synthetic / Manually authored |
| `mixed.txt` | Code-Switched / Mixed | Intra-sentence switching between Devanagari Hindi and English technical terms, model specifications, and function calls. | 100% Synthetic / Manually authored |
| `code.txt` | Source Code | Clean Python, C, Bash, and SQL code snippets with indentation (2-space, 4-space, tabs), variable naming conventions, and logic. | 100% Synthetic / Manually authored |
| `numbers.txt` | Numeric & Arithmetic | Floating point numbers, timestamps, dates, Indian currency (₹), phone numbers, serial IDs, and arithmetic equations. | 100% Synthetic / Algorithmically generated |
| `math.txt` | Mathematical Symbols | First-order logic, linear algebra, calculus, set theory, Greek letters, and LaTeX-style ASCII/Unicode expressions. | 100% Synthetic / Manually authored |
| `unicode.txt` | Extended Unicode & Emojis | Complex emojis, ZWJ sequences, Sanskrit conjuncts, Vedic accents, runes, non-Latin scripts (Tamil, Bengali, Telugu, Gujarati). | Curated standard Unicode test strings |

---

## 3. Explicit Limitations & Boundaries

> ### CRITICAL SCIENTIFIC DISCLAIMER
> 1. **Not a General Language Representation**: This corpus does **NOT** represent the global statistical distribution of natural Hindi, English, Hinglish, or source code.
> 2. **Controlled Benchmark Rig**: It is an isolated engineering fixture designed to expose tokenizer edge cases, verify byte-level fallback, and compare relative compression behavior between candidate vocabulary budgets.
> 3. **Prohibition of Downstream Claims**: Performance on this corpus must **NOT** be used to claim that a vocabulary size is "universally optimal" for any language or task. All conclusions are qualified as *"best on this benchmark corpus"*.
