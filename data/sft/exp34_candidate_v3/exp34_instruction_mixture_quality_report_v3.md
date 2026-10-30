# Exp34 Instruction Mixture Candidate V3 Quality, Diversity & Remediation Audit Report

**Experiment:** `Exp34_Attractor_Suppression_SFT`  
**Dataset Path:** `data/sft/exp34_candidate_v3/exp34_instruction_mixture_train_v3.jsonl`  
**Manifest Path:** `data/sft/exp34_candidate_v3/exp34_instruction_mixture_manifest_v3.json`  
**Compiler Script:** `scripts/compile_exp34_candidate_v3.py`  
**Dataset SHA-256:** `e09439b7aee8c1f56c9d6a105f062a76f02c04b106b1f0c8e4a4a8a5d277848d`  
**Audit Date:** 2026-10-29T19:30:00Z  
**Audit Status:** **READY FOR HUMAN REVIEW**  

---

## 1. Executive Summary & Root Cause Resolution

Following the independent audit failure of Candidate V2, Candidate V3 was engineered with a strict **diversity-first architecture**.
The root cause of earlier failures was **parameterized loop generation**, where generator loops recycled a single conceptual answer (e.g. 80 mutable default bug fixes in Category 3, 80 BST DFS traversals in Category 4, 10 cycled conversations in Category 8, and 39 micro-batch size dialogues in Category 10).

In Candidate V3:
- **Zero Loop Parameterization:** Every single record in Categories 1, 2, 3, 4, 7, and 8 was individually authored across distinct conceptual topic families.
- **Corpus Grounding:** Categories 6 (Summarization), 9 (Hindi), and 10 (Hinglish) draw directly upon 100 unique, authentic statements from ChakrView's committed raw foundation and dialogue corpora (`data/raw/`).
- **Elimination of Attractor Templates:** All boilerplate response tails (such as `" This architectural choice directly supports predictable and reproducible model behavior."`) were completely removed.
- **Lineage Preservation:** V1 (`data/sft/exp34_instruction_mixture_train.jsonl` at `56571858de...`) and V2 (`data/sft/exp34_candidate_v2/` at `6836ff38e1...`) remain completely untouched.

---

## 2. Category Diversity & Topic Inventory

| Category | Count | Topic Families / Coverage | Historical Defect | V3 Status |
| :--- | :---: | :--- | :--- | :--- |
| **1. technical_explanations** | 100 | OS, Hardware, Networking, Distributed Systems, ML Architecture (100 distinct topics) | None | Verified diverse |
| **2. python_programming** | 100 | Built-ins, Iterators, Collections, Decorators, Context Managers, Generators, AST (100 distinct concepts) | V1 `groupby` collapse | Verified diverse (`groupby` = 1) |
| **3. code_debugging** | 100 | 20 distinct bug families (Slicing, Mutable State, Concurrency, Numeric/Types, Scoping, Dunders, Exceptions, IO, Regex, Async, Typing, Memory, Packaging, Functional, Math, Unicode, Mocking, Identity, System) | V2 80x `cache={}` collapse | **100% Remediated** (`mutable default` = 1) |
| **4. algorithmic_reasoning** | 100 | 10 algorithmic families (Graph Shortest Paths, Trees & Disjoint Sets, Dynamic Programming, Sorting & Searching, Strings, Greedy, Number Theory, Complexity Theory, Computational Geometry, Backtracking) | V2 80x BST DFS collapse | **100% Remediated** (BST DFS = 0) |
| **5. arithmetic_reasoning** | 100 | 10 distinct word problem domains (Discount, Speed/Distance, Perimeter/Area, Simple Interest, Ratio, Work Rate, Compound Interest, Averages, Profit, Circumference) | V1 warehouse addition | Verified diverse & calculated |
| **6. passage_summarization** | 100 | 100 distinct passages from `english_foundation.txt`, `english_computer_science.txt`, `english_systems.txt` with 5 varied prompt framings and customized takeaways | Fixed prefix wrapping | Remediated (4 varied summary frames) |
| **7. constrained_formatting** | 100 | JSON schemas, Key-Value/INI/.env, Markdown tables/checklists, Identifier casing transformations, CIDR/Hex/Interval/LaTeX representations | V2 4-item cycling collapse | **100% Remediated** (0 pipeline_buffer repeats) |
| **8. conversational_clarification** | 100 | Disambiguation, Missing Context, ML Systems, Hyperparameters, Data Governance, Indian Computing, Conversational Troubleshooting | V2 10-topic cycle + tail boilerplate | **100% Remediated** (0 boilerplate tails) |
| **9. hindi_comprehension** | 100 | 100 distinct authentic lines from `hindi_foundation.txt`, `hindi_computing.txt`, `hindi_science_culture.txt`, `hindi_dialogue.txt` | Modulo prefix cycling | Remediated (Grammatical variation) |
| **10. bilingual_hinglish** | 100 | 100 distinct authentic lines from `hinglish_foundation.txt`, `hinglish_tech.txt`, `mixed_foundation.txt`, `mixed_multilingual.txt` | V2 39x micro-batch loop | **100% Remediated** (Micro-batch loop = 1) |

---

## 3. Repetition & Failure Pattern Comparison Across Versions

| Pattern / Metric | Candidate V1 | Candidate V2 | Candidate V3 (Current) | Status |
| :--- | :---: | :---: | :---: | :--- |
| **Total Records** | 1,000 | 1,000 | **1,000** | Balanced (100/cat) |
| **Unique Prompts** | 1,000 (100%) | 1,000 (100%) | **1,000 (100%)** | Verified |
| **Unique Responses** | 358 (35.8%) | 1,000 (100%) | **1,000 (100%)** | Verified |
| **Max Template Frequency** | 100x (DFS) | 80x (`cache={}`) | **10x** (Arithmetic domain) | **-87.5% drop in clustering** |
| **`itertools.groupby` responses** | 60 | 1 | **1** | Maintained |
| **`mutable default cache={}` bug fixes** | 0 | 80 | **1** | **Remediated** |
| **BST range DFS traversals** | 0 | 80 | **0** | **Remediated** |
| **Boilerplate conversational tail** | 0 | 100 | **0** | **Remediated** |
| **Pipeline buffer formatting tasks** | 0 | 25 | **0** | **Remediated** |
| **Micro-batch size Hinglish dialogues** | 0 | 39 | **1** | **Remediated** |

---

## 4. Benchmark Leakage & Quality Gate Verification

- **Production Validator (`chakrview/training/dataset_validator.py`):** **PASSED** with 0 errors.
- **Benchmark Evaluation Tasks Checked:** All 44 tasks (32 intelligence + 12 coding).
  - Exact Prompt Matches: **0**.
  - Exact Response / Gold Matches: **0**.
  - Significant 4-gram Overlaps: **0**.
- **`SFTDataset` Compatibility:** Successfully loaded with canonical tokenizer (`data/experiments/vocab_4096`).
- **Exact Reproducibility:** Deterministic compilation verified across independent compiler invocations (`e09439b7aee8c1f56c9d6a105f062a76f02c04b106b1f0c8e4a4a8a5d277848d`).

