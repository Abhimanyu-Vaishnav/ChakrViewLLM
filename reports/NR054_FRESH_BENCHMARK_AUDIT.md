# NR-054: FRESH 50-PROMPT BENCHMARK & EVALUATOR AUDIT

**Mission Identifier:** NR-054  
**Date:** November 2026  
**Auditor:** ChakrView Independent Forensic Engineering Team  
**Scope:** Prompt design, category distribution, keyword vulnerability, and scoring rubric audit of `fresh_heldout_benchmark_50.json`.

---

## 1. Executive Summary

Mission NR-053 introduced a 50-prompt heldout evaluation benchmark (`fresh_heldout_benchmark_50.json`) to test generalization on unseen queries across 5 categories:
- Hindi (`HIN-01` to `HIN-10`)
- Hinglish (`HNG-01` to `HNG-10`)
- General Knowledge & Factual QA (`FQA-01` to `FQA-10`)
- Instruction Following (`INS-01` to `INS-10`)
- Context Recall & Reasoning (`REC-01` to `REC-10`)

This audit scrutinized both the benchmark prompt construction and the automated scoring rubric used in `experiments/forensics_nr053/fresh_eval_results_50.json`.

**Key Finding:** The benchmark scoring rubric suffered from severe single-token substring vulnerability. Many outputs that were syntactically malformed or hallucinatory received maximum scores (Score 3) because common single characters or short words (`'ice'`, `'red'`, `'अ'`, `'2'`) matched anywhere in the output. When verified factual correctness is enforced, only **2 out of 50 prompts** (`FQA-01` and `FQA-03`) were genuinely answered correctly by the models.

---

## 2. Benchmark Composition & Diversity Analysis

| Category | Prompt Count | Description | Quality Assessment |
| :--- | :--- | :--- | :--- |
| **Hindi (`HIN`)** | 10 | Daily conversational queries in Devanagari script | Clear, realistic, representative |
| **Hinglish (`HNG`)** | 10 | Colloquial mixed Roman Hindi/English | Representative of colloquial Indian texting |
| **Factual QA (`FQA`)** | 10 | Clear factual questions (capitals, science, geography) | High quality, unambiguous answers |
| **Instruction (`INS`)** | 10 | Constrained generation (e.g. "Reply with only one word") | Good test of instruction adherence |
| **Recall/Reasoning (`REC`)**| 10 | Multi-step reasoning or context extraction | Appropriate difficulty for SLMs |

**Assessment:** The prompt set itself is a well-designed, legitimate heldout diagnostic tool. However, the automated grading rubric was excessively permissive.

---

## 3. Rubric Vulnerability & False Positive Forensic Breakdown

The automated scoring rubric in NR-053 applied regex and substring matching on pre-defined keyword lists:
- `score = 3` if mandatory answer keywords were detected.
- `score = 1` if partial topic keywords matched.
- `score = 0` if neither matched.

### Identified False Positive Categories:

1. **Short Numerical Characters (e.g., `'2'` in `INS-02`)**:
   - Prompt `INS-02`: *"What is 5 minus 3? Reply with only the number."*
   - Target answer: `2`.
   - Keyword: `'2'`.
   - Vulnerability: Any hallucinated output containing the character `'2'` (e.g. dates, timestamps, random numbers like `2024`, `page 2`) triggered a Score 3.
   - Model Output (`NR052_38M_SFT`): `"The answer is 2."` -> Genuine hit, but in other runs models outputting `"Chapter 2 was published in 2021"` also received Score 3.

2. **Substrings inside Common Words (e.g., `'ice'` in `FQA-07`)**:
   - Prompt `FQA-07`: *"What is solid water called?"*
   - Target answer: `ice`.
   - Vulnerability: Words such as `device`, `notice`, `service`, `police`, `practice` contain the substring `ice`. If the model outputted `"Notice that water..."`, it triggered a false positive match.

3. **Short Hindi Vowels/Consonants (e.g., `'अ'` in `HIN-07`)**:
   - Target keyword: `'अ'` (or short inflections).
   - Vulnerability: In Devanagari script, `'अ'` appears across hundreds of common Hindi prefixes (`अनुसार`, `अच्छा`, `अधिक`). Incoherent Hindi babble matching this common glyph was awarded Score 3.

---

## 4. Reconciled Score Distribution vs Verified Factuality

When outputs from `experiments/forensics_nr053/fresh_eval_results_50.json` were audited by human forensic review for semantic truthfulness:

| Model | Automated Score-3 Hits | Audited Factually Correct Hits | False Positive Rate |
| :--- | :--- | :--- | :--- |
| **RC3 Historical Baseline** | 3 (`FQA-01`, `FQA-03`, `INS-02`) | 2 (`FQA-01` Tokyo, `FQA-03` Pacific) | 33.3% |
| **NR051 17M SFT** | 3 (`FQA-01`, `FQA-03`, `HIN-07`) | 2 (`FQA-01` Tokyo, `FQA-03` Pacific) | 33.3% |
| **NR051 38M SFT** | 3 (`FQA-01`, `FQA-03`, `HIN-07`) | 2 (`FQA-01` Tokyo, `FQA-03` Pacific) | 33.3% |
| **NR052 17M SFT** | 3 (`FQA-01`, `FQA-03`, `INS-06`) | 2 (`FQA-01` Tokyo, `FQA-03` Pacific) | 33.3% |
| **NR052 38M SFT** | 4 (`FQA-01`, `FQA-03`, `INS-02`, `HIN-07`) | 2 (`FQA-01` Tokyo, `FQA-03` Pacific) | 50.0% |

### Truly Solved Prompts across all models:
1. `FQA-01`: *"What is the capital of Japan?"* -> Correctly answers: `Tokyo` (or `Tokyo is the capital of Japan`).
2. `FQA-03`: *"Which is the largest ocean on Earth?"* -> Correctly answers: `Pacific Ocean`.

---

## 5. Audit Recommendations for Future Benchmarking

1. **Word-Boundary Matching:** Strictly enforce regex word boundaries (`\b`) to prevent substring triggers like `ice` in `notice`.
2. **Length Penalties & Instruction Filters:** For instructions requiring single-token answers, penalize responses longer than 5 tokens.
3. **LLM-as-a-Judge or Human In-The-Loop:** Automated keyword heuristics must be cross-validated against strict exact match or human-labeled truth sets.
