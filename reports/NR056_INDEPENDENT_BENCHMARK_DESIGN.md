# NR-056: INDEPENDENT BENCHMARK V3 DESIGN & EVALUATION REPORT

**Mission Identifier:** NR-056  
**Date:** November 28, 2026  
**Auditor:** ChakrView Independent Forensic Engineering Team  
**Scope:** Design, construction, manifest sealing, and execution of the 300-prompt Benchmark V3 evaluation suite across all 5 candidate models.

---

## 1. Executive Summary

To establish the genuine scientific capability frontier and eliminate all remaining contamination vulnerabilities present in prior benchmarks, **Benchmark V3** was designed and sealed as an independent 300-prompt evaluation suite (`data/evaluation/benchmark_v3/benchmark_v3_300.json`).

Benchmark V3 features:
1. **Uncontaminated Entities:** Fictional contexts, novel numerical equations, and global entities absent from the ~624k training corpus.
2. **Balanced Distribution:** 300 prompts evenly distributed across 10 distinct linguistic and cognitive tasks.
3. **Sealed Integrity:** SHA-256 manifest recorded prior to evaluation (`f4cec38066a2ccbd791c5282e92352cca718466cf55947b0e6d9d54a020f499d`).
4. **Independent Raw Generation Capture:** All 1,500 model outputs (300 prompts $\times$ 5 candidates) were persisted to `experiments/forensics_nr056/raw_generations/`.

---

## 2. Category Distribution & Taxonomic Composition

| Capability Category | Prompt Prefix | Items | Percentage | Task Type Tested |
| :--- | :--- | :---: | :---: | :--- |
| **Factual Knowledge** | `V3-FQA` | 35 | 11.7% | Unseen and global entity knowledge (world capitals, currencies, science) |
| **Basic Arithmetic** | `V3-MTH` | 35 | 11.7% | Discrete operations (+, -, *, /) with randomized operands |
| **Multi-Step Arithmetic** | `V3-WRD` | 30 | 10.0% | Multi-step word problems requiring intermediate summation |
| **Logical Reasoning** | `V3-LOG` | 30 | 10.0% | Syllogistic deduction, transitivity, common sense physics |
| **Reading Comprehension** | `V3-RC` | 35 | 11.7% | Novel entity extraction from synthetic passages |
| **Instruction Following** | `V3-INS` | 35 | 11.7% | Codeword constraints, uppercase formatting, negative exclusions |
| **English Fluency** | `V3-ENG` | 30 | 10.0% | Grammatical pluralization, syntax transformations |
| **Hindi Language** | `V3-HIN` | 35 | 11.7% | Devanagari queries spanning science, dates, and common sense |
| **Hinglish Language** | `V3-HNG` | 20 | 6.7% | Colloquial mixed Roman Hindi/English QA |
| **Context Retention** | `V3-CTX` | 15 | 5.0% | Multi-turn PIN and secret code retention |
| **Total** | | **300** | **100.0%** | |

---

## 3. Benchmark V3 Candidate Performance Summary

All 5 candidates were evaluated using greedy deterministic decoding ($T=0.0$, max tokens 32):

| Candidate Name | Parameters | Training Corpus | Raw Sum (/900) | Raw Mean (/3.0) | Normalized Score (%) | Score Dist `[0, 1, 2, 3]` | 95% Bootstrap CI (Norm %) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **NR052 38M SFT** | 37.50M | Corpus V4 | **11** | **0.0367** | **1.22%** | `[295, 0, 4, 1]` | `[0.22%, 2.44%]` |
| **RC3 Baseline** | 17.06M | Corpus V3 | 8 | 0.0267 | 0.89% | `[296, 0, 4, 0]` | `[0.22%, 1.78%]` |
| **NR051 17M SFT** | 17.06M | Corpus V3 PT | 4 | 0.0133 | 0.44% | `[298, 0, 2, 0]` | `[0.00%, 1.11%]` |
| **NR051 38M SFT** | 37.50M | Corpus V3 PT | 0 | 0.0000 | 0.00% | `[300, 0, 0, 0]` | `[0.00%, 0.00%]` |
| **NR052 17M SFT** | 17.06M | Corpus V4 PT | 0 | 0.0000 | 0.00% | `[300, 0, 0, 0]` | `[0.00%, 0.00%]` |

---

## 4. Key Scientific Revelations on Clean Evaluation

1. **Near-Zero True Capability on Unseen Data:** When evaluation items are rigorously isolated from the training corpus, model performance across all 5 candidates drops to between **$0.0\%$ and $1.22\%$**.
2. **Absence of Algorithmic Generalization:**
   - 0 out of 35 basic arithmetic problems were solved.
   - 0 out of 30 multi-step word problems were solved.
   - 0 out of 35 novel entity reading comprehension passages were solved.
3. **The Pretraining Token Scale Boundary:** At 17M and 38M parameter scales trained on only $\sim 1\text{M}$ tokens, the models lack sufficient linguistic world knowledge or algorithmic depth to answer unseen questions.
