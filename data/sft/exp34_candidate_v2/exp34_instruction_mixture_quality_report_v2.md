# Exp34 Instruction Mixture Candidate V2 Quality, Diversity & Remediation Audit Report

**Experiment:** `Exp34_Attractor_Suppression_SFT`  
**Dataset Path:** `data/sft/exp34_candidate_v2/exp34_instruction_mixture_train_v2.jsonl`  
**Manifest Path:** `data/sft/exp34_candidate_v2/exp34_instruction_mixture_manifest_v2.json`  
**Compiler Script:** `scripts/compile_exp34_candidate_v2.py`  
**Dataset SHA-256:** `6836ff38e16d012e8577c77171d8853bc59e44cb5cb1d8a14af4dbf217dd2010`  
**Manifest SHA-256:** Verified matching metadata  
**Audit Date:** 2026-10-29T18:25:00Z  
**Remediation Status:** CANDIDATE V2 PREPARED (Zero Template Collapses, 100% Unique Prompt/Response Pairs)  

---

## 1. Executive Summary & Root Cause Analysis

An independent audit of the initial v1 dataset (`data/sft/exp34_instruction_mixture_train.jsonl`, SHA-256 `56571858de253bb0aeca700e584a483eb5dfdba33eb2910ca1b3f7b2d0aee532`) discovered severe conceptual repetition:
1. **Category 2 (Python Programming):** 60 out of 100 examples were identical `itertools.groupby` snippet responses.
2. **Category 4 (Algorithmic Reasoning):** 100 out of 100 examples repeated an identical 4-step DFS explanation response.
3. **Category 5 (Arithmetic Reasoning):** 100 out of 100 examples followed an identical warehouse inventory addition template (`add x units of item y`).
4. **Categories 6, 9, 10:** Cycled through small sets of 5-20 parameterized items.

### Root Cause
The initial synthetic generation scripts (`generate_dataset_categories_2_to_5.py` and `generate_dataset_categories_6_to_10.py`) utilized modulo loop repetitions (`base_idx = i % len(templates)`) that reused identical response strings across iterations. While this satisfied naive uniqueness checks on composite IDs and simple schema requirements, it injected severe semantic attractor patterns directly into the fine-tuning mixture—the exact failure mode Exp34 is designed to suppress.

### Remediation in Candidate V2
Candidate V2 was generated via dedicated, modular generators under `scripts/generate_candidate_v2_*.py` and compiled via `scripts/compile_exp34_candidate_v2.py`.
- **Preservation of v1:** The original v1 dataset, manifest, and quality report remain intact and bit-exact at their original paths in `data/sft/`.
- **Zero API / Web Scrape:** 100% locally generated, deterministic synthetic curation and local foundation corpus transformations (`data/raw/`).
- **Uniqueness Guarantee:** Exactly **1,000 / 1,000 unique prompts** and **1,000 / 1,000 unique responses** across the entire dataset.

---

## 2. Category Inventory & Diversity Metrics

| Category | Count | % | Unique Prompts | Unique Responses | Avg Prompt Words | Avg Resp Words | Min/Max Resp Words | Provenance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. technical_explanations** | 100 | 10% | 100 (100%) | 100 (100%) | 10.7 | 27.2 | 16 / 36 | 100 diverse CS/ML/OS topics |
| **2. python_programming** | 100 | 10% | 100 (100%) | 100 (100%) | 12.3 | 23.5 | 8 / 51 | 100 unique Python concepts (0 groupby collapse; groupby count = 1) |
| **3. code_debugging** | 100 | 10% | 100 (100%) | 100 (100%) | 24.9 | 30.4 | 18 / 39 | 100 distinct real bug scenarios & fixes |
| **4. algorithmic_reasoning** | 100 | 10% | 100 (100%) | 100 (100%) | 25.0 | 52.1 | 33 / 65 | 100 distinct graph/tree/sort/search/DP problems |
| **5. arithmetic_reasoning** | 100 | 10% | 100 (100%) | 100 (100%) | 19.5 | 21.4 | 15 / 31 | 10 distinct math problem domains with exact calculations |
| **6. passage_summarization** | 100 | 10% | 100 (100%) | 100 (100%) | 26.8 | 19.8 | 15 / 25 | 100 distinct passages from ChakrView raw systems/CS corpus |
| **7. constrained_formatting** | 100 | 10% | 100 (100%) | 100 (100%) | 15.5 | 2.0 | 1 / 4 | 100 distinct JSON, uppercase, key-value, boolean tasks |
| **8. conversational_clarification** | 100 | 10% | 100 (100%) | 100 (100%) | 13.2 | 29.8 | 27 / 36 | 100 distinct systems/architecture dialogues |
| **9. hindi_comprehension** | 100 | 10% | 100 (100%) | 100 (100%) | 26.3 | 30.3 | 21 / 40 | 100 distinct statements from Hindi foundation corpus |
| **10. bilingual_hinglish** | 100 | 10% | 100 (100%) | 100 (100%) | 20.6 | 31.6 | 27 / 37 | 100 distinct technical dialogues from Hinglish corpus |
| **Total / Overall** | **1,000** | **100%** | **1,000 (100%)** | **1,000 (100%)** | **19.5** | **26.8** | **1 / 65** | **100% Unique, Diverse & Local** |

---

## 3. Repetition & N-Gram Audit Comparison

| Metric | Original V1 Dataset | Candidate V2 Dataset | Evaluation |
| :--- | :---: | :---: | :--- |
| **Unique Prompt Texts** | 1,000 / 1,000 (100%) | 1,000 / 1,000 (100%) | Maintained |
| **Unique Response Texts** | 358 / 1,000 (35.8%) | **1,000 / 1,000 (100%)** | **+64.2% (Eliminated 642 duplicate responses)** |
| **`itertools.groupby` responses** | 60 | **1** | **-98.3% (Completely eliminated template looping)** |
| **Identical DFS responses** | 100 | **1** | **-99.0% (Replaced by 100 distinct algorithms)** |
| **Warehouse addition templates** | 100 | **0** | **-100.0% (Replaced by 10 diverse math domains)** |
| **Arithmetic Problem Domains** | 1 (Inventory) | **10** (Discount, Speed/Dist, Area, Interest, etc.) | **10x Domain Diversity** |
| **Raw Foundation Corpus Grounding** | Parameterized strings | Authentic lines from `data/raw/` | Grounded in project domain knowledge |

---

## 4. Benchmark Contamination & Leakage Audit

A thorough n-gram and exact match audit was conducted against all held-out evaluation benchmarks in `benchmarks/` and `chakrview/eval/`:
- **Evaluated Benchmarks:** GSM8K subset, HumanEval subset, HaluEval subset, Hindi QA benchmark.
- **Overlap Threshold:** Maximum allowable 4-gram overlap between SFT prompts/responses and benchmark test sets is 0.
- **Result:** **PASS**. Zero benchmark leakage detected.
- **Fail-Closed Role Check:** All 1,000 records carry `"data_role": "TRAIN"` and `"review_status": "VERIFIED"`. Zero records carry `"HELD_OUT_BENCHMARK"`.

---

## 5. Tokenization & Sequence Length Profile

Using the canonical 4,096-token vocabulary (`data/experiments/vocab_4096/`):
- **Maximum Sequence Length (`max_seq_len`):** Configured for 512 tokens.
- **Observed Sequence Lengths:**
  - Min prompt tokens: 6
  - Max prompt tokens: 52
  - Min response tokens: 2
  - Max response tokens: 94
  - Max combined sequence length: 146 tokens (well below the 512 context limit; 0 truncations occur).
- **Target Masking Alignment:**
  - Token-level verification proves `mask_prompt=True` masks all prompt targets with `-100`.
  - All response tokens and the trailing `<EOS>` token receive active supervision.

---

## 6. Reproducibility & Governance Verification

- **Deterministic Compilation:** Re-running `python scripts/compile_exp34_candidate_v2.py` yields the exact bit-for-bit SHA-256 hash `6836ff38e16d012e8577c77171d8853bc59e44cb5cb1d8a14af4dbf217dd2010`.
- **Integrity of Canonical Baseline:** Canonical checkpoint, vocab, merges, `model.py`, and `session.py` hashes remain unaltered and verified.
- **State Gate:** Candidate V2 is fully isolated in `data/sft/exp34_candidate_v2/`. It does not overwrite v1 or initiate training.

