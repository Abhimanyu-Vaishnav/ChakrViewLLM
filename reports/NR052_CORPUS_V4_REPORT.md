# NR-052: Corpus V4 Construction & Specification Report

**Mission:** CHAKRVIEW NR-052  
**Date:** November 27, 2026  
**Corpus Directory:** `data/corpus_v4/`  
**Manifest Path:** [corpus_v4_manifest.json](file:///D:/Project/ChakrView/data/manifests/corpus_v4_manifest.json)

---

## 1. Executive Summary

**ChakrView Corpus V4** is an isolated, deduplicated pretraining corpus built from clean, verified text sources. It resolves the template repetition and boilerplate noise discovered during the Phase 1 & 2 audits.

### Forensic Findings on Corpus Size Targets
- The Phase 3 engineering milestone specified:
  - First target: *10 million verified unique tokenizer tokens.*
  - Stretch target: *25 million verified unique tokenizer tokens.*
  - **Explicit Non-Fabrication Constraint:** *"If the repository cannot support 10 million clean unique tokens, report the actual quantity and the remaining gap. Do not manufacture data to satisfy the target."*
- **Actual Quantity:** Corpus V4 contains **624,068 verified unique tokenizer tokens** across **6,134 clean documents**.
- **Remaining Gap to 10M Target:** **9,375,932 tokens (93.8% gap)**.
- **Root Cause of Gap:** The repository's raw continuous prose text consists of 1.57 MB of text (~476k tokens). All claims of 15,000+ item corpora in legacy files (`ucip14r`, `ucip17`) were produced by repeating a tiny set of 3 to 15 response templates across artificial prompt variations. Removing duplicates, synthetic boilerplate, and benchmark contamination leaves exactly ~624k tokens of genuine linguistic content.

---

## 2. Corpus V4 Architecture & Splits

| Split | Documents | Tokens (Canonical 4096 BPE) | File Path | SHA-256 Checksum |
| :--- | :--- | :--- | :--- | :--- |
| **Train Split** (95%) | 5,828 docs | **592,824 tokens** | `data/corpus_v4/corpus_v4_train.jsonl` | `33036e4f358f2da6739e8020a1db8935ce3ef8dbb7df3fc1e231b26fce7c0c16` |
| **Validation Split** (5%) | 306 docs | **31,244 tokens** | `data/corpus_v4/corpus_v4_val.jsonl` | `577b21dc862fe4369e06e9ecfe39fa763dfd71c893080eb2743810f279f66ae9` |
| **Total Corpus V4** | **6,134 docs** | **624,068 tokens** | — | — |

---

## 3. Language & Domain Distributions

- **Languages:** English (2,856 docs, ~46.6%), Hindi (1,319 docs, ~21.5%), Hinglish (662 docs, ~10.8%), Code/Python (586 docs, ~9.6%), Mathematics (421 docs, ~6.9%), Reasoning (195 docs, ~3.2%), Sanskrit (95 docs, ~1.5%).
- **Domains:** Systems architecture, literature, algorithms, formal derivations, natural dialogues, instructional QA, procedural reasoning, and classical grammar.

---

## 4. Contamination Gating

- Evaluated against both `sealed_120_benchmark.json` and `fresh_matched_benchmark_1000.json`.
- Exact string matches: **0 (0.0% contamination)**.
- 10-gram overlap: **0 (0.0% contamination)**.
