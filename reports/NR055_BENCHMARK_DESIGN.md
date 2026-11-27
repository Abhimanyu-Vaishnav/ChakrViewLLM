# NR-055: BENCHMARK V2 SPECIFICATION & DESIGN REPORT

**Mission Identifier:** NR-055  
**Date:** November 2026  
**Auditor:** ChakrView Independent Forensic Engineering Team  
**Scope:** Architectural design, domain taxonomy, constraint definitions, and isolation manifest for Benchmark V2 (`benchmark_v2_100.json`).

---

## 1. Executive Summary

Previous benchmark suites (such as the 50-prompt heldout evaluation in NR-053) relied heavily on automated substring matching that awarded full points to malformed outputs containing single digits or common letters.

**Benchmark V2** establishes an independently authored, multi-category evaluation suite consisting of **95 high-quality prompts** across 8 core cognitive and linguistic capabilities. Every prompt incorporates explicit expected answers, acceptable alternative answers, rubric types, and structural constraints.

---

## 2. Category Distribution & Taxonomic Breakdown

| Category | Identifier | Prompt Count | Ratio | Key Evaluation Capability |
| :--- | :--- | :---: | :---: | :--- |
| **Factual Knowledge** | `FQA` | 15 | 15.8% | General world knowledge (capitals, geography, science, currencies) |
| **Arithmetic & Numbers** | `MTH` | 15 | 15.8% | Basic arithmetic operations (+, -, *, /) and word problem math |
| **Instruction Following** | `INS` | 15 | 15.8% | Constraint adherence (single-word, formatting, case, repetition) |
| **Hindi Language** | `HIN` | 15 | 15.8% | Conversational and factual queries in native Devanagari script |
| **Logical Reasoning** | `LOG` | 10 | 10.5% | Deductive logic, transitivity, common sense physics |
| **Reading Comprehension**| `RC` | 10 | 10.5% | Passage extraction and contextual entity resolution |
| **Hinglish Language** | `HNG` | 10 | 10.5% | Colloquial romanized Hindi/English conversational queries |
| **Context Retention** | `CTX` | 5 | 5.3% | Immediate-context extraction and conversational consistency |
| **Total** | | **95** | **100%** | |

### Difficulty Distribution:
- **Easy:** 53 prompts (55.8%)
- **Medium:** 32 prompts (33.7%)
- **Hard:** 10 prompts (10.5%)

### Language Distribution:
- **English (`en`):** 70 prompts (73.7%)
- **Hindi (`hi`):** 15 prompts (15.8%)
- **Hinglish (`hi-en`):** 10 prompts (10.5%)

---

## 3. Rubric Types & Strict Constraints

To eliminate the vulnerabilities of naive substring matching, each prompt specifies a mandatory `rubric_type`:

1. `numerical_exact`:
   - Extracts all numbers using regex `[-+]?\d*\.?\d+`.
   - Requires target number to be uniquely matched or explicitly positioned as the final answer in an equation.
   - Rejects extraneous random numbers.

2. `strict_exact`:
   - Enforces formatting constraints (e.g. `single_word`, `case_insensitive`, `spelling`).
   - Rejects conversational preamble if the prompt strictly demanded a single token.

3. `token_set_match`:
   - Requires all mandatory set elements to be present (e.g. standard traffic light colors).

4. `exact_or_substring`:
   - Requires whole-word or whole-phrase boundary matching (`\b<target>\b`).
   - Prevents substring hits like `'ice'` matching inside `'notice'`.
   - Applies length penalties for excessive verbose hallucination.

---

## 4. Benchmark Isolation & Provenance

- **Storage Location:** `data/evaluation/benchmark_v2/benchmark_v2_100.json`
- **Manifest:** `data/manifests/benchmark_v2_manifest.json`
- **Contamination Guard:** This benchmark is strictly quarantined. No training or fine-tuning run has access to this directory.
