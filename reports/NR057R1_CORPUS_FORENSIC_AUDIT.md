# NR-057-R1 CORPUS V5 FORENSIC AUDIT REPORT

**Date:** November 28, 2026  
**Mission:** NR-057-R1 Corpus Expansion Completion & Training Gate Revalidation  
**Artifact ID:** `NR057R1_CORPUS_FORENSIC_AUDIT`  
**Audit Output JSON:** [corpus_v5_audit.json](file:///d:/Project/ChakrView/experiments/corpus_nr057_r1/corpus_v5_audit.json)

---

## 1. Executive Summary

A comprehensive, zero-assumption forensic audit of `Corpus V5` was executed directly from the stored JSONL files (`corpus_v5_train.jsonl`, `corpus_v5_val.jsonl`, and `corpus_v5_test.jsonl`). Each document was tokenized line-by-line using the canonical Byte-Level BPE 4096 tokenizer (`data/experiments/vocab_4096`), hashed with SHA-256, and verified for disjointness across splits.

### Key Forensic Audit Findings:
1. **Mathematical Disjointness Confirmed:**
   - Train vs Validation Hash Overlap: **0 documents**
   - Train vs Held-out Test Hash Overlap: **0 documents**
   - Validation vs Held-out Test Hash Overlap: **0 documents**
   - **Verdict:** Strict document-level disjointness is 100% verified.
2. **Corpus Size & Tokens Reconciled:**
   - **Total Usable Documents:** **9,511 unique documents**
   - **Total Usable Tokens:** **3,565,296 tokens**
   - **Train Split:** 8,561 documents, 3,187,865 tokens (89.41%)
   - **Validation Split:** 475 documents, 221,282 tokens (6.21%)
   - **Test Split:** 475 documents, 156,149 tokens (4.38%)
   - **Malformed / Empty Documents:** **0 documents** (100% valid JSON, non-empty text fields).
   - **Exact Duplicate Rate:** **0.0000%** (all 9,511 documents possess distinct SHA-256 hashes).

---

## 2. In-Depth Origin Analysis of the 1.51M Python Tokens

A specific mandate of NR-057-R1 was to determine the exact origin of the 1,510,114 tokens tagged under `code_python_stdlib`.

### Provenance Audit:
- **Source Origin:** Physical Python 3.11 Standard Library source distribution located at `sys.base_prefix\Lib` (e.g. `C:\Program Files\Python314\Lib` or virtual environment base standard library).
- **Collection Mechanism:** Filtered out test directories (`test/`, `tkinter/`, `idlelib/`, `distutils/`) and segmented Python files by structural definition boundaries (`def `, `class `, `async def `) with length bounds (150 to 4,000 characters).
- **Constituent Modules:**
  - `typing.py` (105 docs): Type hinting system, generic aliases, type variables.
  - `inspect.py` (84 docs): Python AST introspection, stack frames, signature binding.
  - `statistics.py` (44 docs): Statistical calculations, distributions, variance, mean implementations.
  - `dataclasses.py` (38 docs): Decorator architectures, field definitions, code generation.
  - `functools.py` (36 docs): Partial application, lru_cache implementations, functional utilities.
  - Core utilities: `shutil.py`, `platform.py`, `_pydecimal.py`, `doctest.py`, `urllib/parse.py`.
- **Finding:** Every single token stems directly from the official Python Software Foundation Standard Library distribution (PSF License 2.0). None are synthetic LLM generations or web forum scrapes.

---

## 3. Domain & Linguistic Imbalance Reality

The audit reveals significant domain skew that must be addressed:
- **Code Dominance:** Python stdlib code represents **42.36%** (1.51M tokens). When combined with `local_code` (2.71%), code totals **45.07%** of the dataset.
- **English Expository Prose:** Wikipedia STEM/Humanities accounts for **23.45%** (836K tokens), and local English foundations account for **3.51%** (125K tokens), totaling **26.96%**.
- **Devanagari Hindi:** Hindi Wikipedia prose represents only **4.81%** (171K tokens), and local Hindi accounts for **2.57%** (91K tokens), totaling **7.38%**.
- **Hinglish:** Only **1.41%** (50K tokens).
- **Formal Mathematics:** Only **1.92%** (68K tokens).

This imbalance violates the target linguistic and cognitive proportions, which will be rebalanced in **Corpus V6**.
