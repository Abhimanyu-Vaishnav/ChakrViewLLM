# Exp34 Instruction Mixture Dataset Quality & Audit Report

**Experiment:** `Exp34_Attractor_Suppression_SFT`  
**Dataset Path:** `data/sft/exp34_instruction_mixture_train.jsonl`  
**Manifest Path:** `data/sft/exp34_instruction_mixture_manifest.json`  
**Audit Date:** 2026-10-29T18:00:00Z  
**Compiler Script:** `scripts/compile_exp34_dataset.py`  
**Dataset SHA-256:** `56571858de253bb0aeca700e584a483eb5dfdba33eb2910ca1b3f7b2d0aee532`  

---

## 1. Purpose & Scope

This dataset was created to support the controlled two-arm experiment `Exp34_Attractor_Suppression_SFT`.
The experiment investigates whether response-only supervised fine-tuning suppresses the model's observed attractor-collapse tendency (repeating generic attractor phrases) and improves instruction-following utility.

- **Target Size:** Exactly 1,000 instruction-response examples.
- **Composition:** 10 balanced categories (100 examples per category).
- **Source Constraints:** 100% locally generated, deterministic synthetic curation. Zero external APIs, zero hosted LLMs, zero pretrained weights.
- **Licensing:** CC0-1.0 (Public Domain Dedication). All content verified locally.

---

## 2. Category Distribution & Provenance

| Category | Example Count | Percentage | Primary Language | Source Provenance |
| :--- | :---: | :---: | :---: | :--- |
| **1. technical_explanations** | 100 | 10.0% | English | Factual curation (OS, networking, ML architectures, DS/Algo) |
| **2. python_programming** | 100 | 10.0% | English / Python | Synthesized idiomatic functions, classes, decorators |
| **3. code_debugging** | 100 | 10.0% | English / Python | Bug identification, root cause explanations, fixes |
| **4. algorithmic_reasoning** | 100 | 10.0% | English | Step-by-step logic, searching, traversals, proofs |
| **5. arithmetic_reasoning** | 100 | 10.0% | English | Verified arithmetic word problems, addition, subtraction, multiplication |
| **6. passage_summarization** | 100 | 10.0% | English | Independent scientific/historical passage summarization |
| **7. constrained_formatting** | 100 | 10.0% | English | Structured JSON, bullet lists, boolean flags, case transformation |
| **8. conversational_clarification**| 100 | 10.0% | English | Polite assistance, architectural boundaries, system clarification |
| **9. hindi_comprehension** | 100 | 10.0% | Hindi (Devanagari)| Grammatically correct technical & computing concepts in Hindi |
| **10. bilingual_hinglish** | 100 | 10.0% | Hinglish (hi-en) | Natural technical dialogue mixing English technical terms with Hindi syntax |
| **Total** | **1,000** | **100.0%** | Multi | **100% Synthetic Verified Local Curation** |

---

## 3. Quality & Contamination Audit

### Schema & Formatting
- **Valid JSONL lines:** 1,000 / 1,000 (100%)
- **Required fields present:** `id`, `prompt`, `response`, `category`, `source_type`, `language`, `license`, `data_role`, `review_status`
- **Missing or null fields:** 0
- **Empty or whitespace-only prompts/responses:** 0
- **Line ending hygiene:** Strict LF (`\n`), no trailing blank lines at EOF.

### Duplicate Detection
- **Exact duplicate prompts:** 0 (1,000 unique prompts)
- **Exact duplicate prompt-response pairs:** 0 (1,000 unique pairs)
- **Normalized duplicate rate:** 0% (every prompt addresses a distinct query, scenario, or index)

### Benchmark Contamination & Leakage Audit
- Evaluated against all 32 capability intelligence benchmark tasks (`evaluation/intelligence_suite/intelligence_tasks.json`) and 12 coding benchmark tasks (`evaluation/coding_suite/coding_tasks.json`).
- **Exact prompt matches with benchmark tasks:** 0
- **High 4-gram overlap matches with benchmark tasks:** 0
- **Leakage check status:** **PASS** (Zero contamination detected by `validate_sft_dataset_file(..., check_leakage=True)`).

*Note on Leakage Check Limitations:* Lexical 4-gram matching verifies that benchmark prompt text and gold expected answers were not copied into the training mixture. It does not preclude semantic overlap on general programming concepts (e.g. factorial or recursion), which are standard curriculum topics.

---

## 4. Reproducibility Audit

- **Generation Command:**
  ```powershell
  .\.venv\Scripts\python.exe -m scripts.compile_exp34_dataset --output_jsonl data/sft/exp34_instruction_mixture_train.jsonl --output_manifest data/sft/exp34_instruction_mixture_manifest.json
  ```
- **Bit-Exact Verification:** Recompilation into separate temporary paths yielded identical SHA-256 hashes (`56571858de253bb0aeca700e584a483eb5dfdba33eb2910ca1b3f7b2d0aee532`).
- **Automated Regression Test:** Covered by [`tests/test_exp34_dataset_generation.py`](file:///d:/Project/ChakrView/tests/test_exp34_dataset_generation.py).

---

## 5. Audit Gate Summary

| Gate | Criterion | Status |
| :--- | :--- | :---: |
| **G1: Artifact Integrity** | Canonical checkpoint, vocab, merges, and `v0.8.0` tag unchanged | **PASS** |
| **G2: Schema Compliance** | Valid JSONL with all required SFT training fields | **PASS** |
| **G3: Target Count** | Exactly 1,000 high-quality examples without low-quality padding | **PASS** |
| **G4: Category Balance** | Exactly 100 examples across each of 10 categories | **PASS** |
| **G5: Duplicate Rate** | 0 exact duplicate prompts or pairs | **PASS** |
| **G6: Benchmark Isolation** | 0 lexical leaks against intelligence and coding benchmark suites | **PASS** |
| **G7: Reproducibility** | Deterministic pipeline; byte-for-byte identical across runs | **PASS** |
| **G8: SFT Engine Loader** | Successfully loads into `SFTDataset` and produces valid causal tensors | **PASS** |

**Final Dataset Status:** **PASS (Ready for Human Review)**

