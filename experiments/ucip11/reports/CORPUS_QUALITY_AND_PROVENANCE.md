# UCIP-11 Phase B: Training Corpus Investigation & Provenance

**Mission ID:** UCIP-11  
**Audit Scope:** Full repository corpus inventory, language balance, duplicate analysis, and provenance.  
**Result Artifact:** `experiments/ucip11/data/manifests/corpus_audit_summary.json`  

---

## 1. Executive Summary

A core scientific question in UCIP-11 is whether ChakrView's poor generalization in earlier iterations stemmed from superficial template repetition and inadequate natural-language exposure.

Our forensic inventory audited 8 distinct dataset tiers across the repository, spanning pretraining corpora (Stage B), instruction collections (Exp34, M3), and previous experimental tuning sets (UCIP-09, UCIP-09S).

**Key Findings:**
1. **The Memorization Trap in UCIP-09:** While `ucip09_train.jsonl` contains 3,240 samples, it has only **593 unique targets**. Probes were synthetic variations with superficial tags like `(domain query 36)`. This trained the model to memorize fixed phrase outputs rather than learn flexible autoregressive generation.
2. **Untapped Stage B Natural Language Corpus:** The Stage B pretraining corpus (`data/processed/stage_b/train`) contains **7,258 natural-language paragraphs** across English, Hindi, Hinglish, Code, and Mathematics with **0% duplicates**, representing ~66,200 tokens of high-quality un-templated text.
3. **High-Quality Instruction Tuning (Exp34 v3):** The `exp34_candidate_v3` dataset contains **900 verified, diverse instruction-response pairs** covering algorithmic reasoning, debugging, bilingual Hinglish, Hindi comprehension, and conversational clarification with CC0-1.0 provenance and zero template duplication.

---

## 2. Quantitative Corpus Inventory Table

| Corpus Tier | Path | Total Entries | Unique Entries | Duplicate Rate (%) | Estimated Tokens (vocab_4096) | Avg Tokens / Entry | Provenance & Role |
|---|---|---|---|---|---|---|---|
| **Stage B Train** | `data/processed/stage_b/train` | 7,258 | 7,258 | **0.0%** | 66,229 | 9.1 | Curated natural language foundation (CC0/Local) |
| **Stage B Val** | `data/processed/stage_b/validation` | 891 | 891 | **0.0%** | 23,125 | 26.0 | Held-out pretraining validation |
| **Stage B Test** | `data/processed/stage_b/test` | 853 | 853 | **0.0%** | 23,504 | 27.6 | Held-out pretraining test |
| **Exp34 v3 Train**| `data/sft/exp34_candidate_v3/...` | 900 | 900 | **0.0%** | 102,406 | 113.8 | Balanced multi-domain SFT (CC0-1.0) |
| **Exp34 v3 Val** | `data/sft/exp34_candidate_v3/...` | 100 | 100 | **0.0%** | 10,558 | 105.6 | Held-out SFT validation |
| **M3 Conversational**| `data/sft/m3_conversational/...` | 540 | 540 | **0.0%** | 37,122 | 68.7 | Identity, greeting, clarification |
| **UCIP-09 Train** | `experiments/cognition/ucip_09/...` | 3,240 | 3,229 | 0.3% (593 targets) | 226,994 | 70.1 | Synthetic template collection |
| **UCIP-09 Val** | `experiments/cognition/ucip_09/...` | 360 | 360 | 0.0% | 24,845 | 69.0 | Template validation split |

---

## 3. Linguistic Composition of Stage B Pretraining Text

| Domain File | Non-Empty Paragraphs | Total Characters | Primary Language | Description |
|---|---|---|---|---|
| `english.txt` | 1,005 | 281,721 | English | Computer science, architecture, operating systems, data structures. |
| `hindi.txt` | 608 | 165,182 | Devanagari Hindi | Technical definitions, computing terminology, cultural & general literature. |
| `hinglish.txt` | 997 | 115,169 | Hinglish (Code-switched) | Conversational dialogue, programming assistance, natural code-mixing. |
| `code.txt` | 2,692 | 169,271 | Python / Config | Algorithms, data structures, implementation snippets. |
| `mathematics.txt` | 1,028 | 118,204 | Math / LaTeX | Equations, definitions, arithmetic derivations, proofs. |
| `reasoning.txt` | 465 | 55,611 | English | Formal logic, incident response, troubleshooting protocols. |
| `sanskrit.txt` | 209 | 30,480 | Devanagari Sanskrit | Shlokas, classical grammatical texts (linguistic regularizer). |
| `numbers.txt` | 213 | 18,656 | Numeric / Tables | Sequences, numeric tables, mathematical foundations. |

---

## 4. Corpus Strategy for UCIP-11

To avoid the catastrophic template memorization of UCIP-09 while preserving language coherence:
1. **Condition B (CPT):** Continuous Pretraining on the clean multi-domain Stage B text using pure causal language modeling (predicting the full sequence without artificial prompt tags).
2. **Condition C (SFT):** Supervised Instruction Tuning on the 900-sample Exp34 v3 mixture, which features genuine diversity across 10 distinct task categories.
3. **Condition D (CPT + SFT):** Two-stage training (Stage B pretraining followed by Exp34 v3 instruction tuning).
4. **Condition E (Scratch):** Training a fresh ChakrMicro architecture from random initialization on the Stage B corpus to measure baseline language acquisition dynamics.
