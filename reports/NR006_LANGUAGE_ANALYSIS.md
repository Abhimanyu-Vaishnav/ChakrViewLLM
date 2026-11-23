# CHAKRVIEW RESEARCH REPORT: CV-GENESIS-NR-006
## LANGUAGE-SPECIFIC CONVERSATIONAL ANALYSIS REPORT

**Mission ID:** CV-GENESIS-NR-006  
**Date:** 23 November 2026  
**Evaluator:** Principal Multilingual NLP Engineer  
**Model Under Test:** Genesis SFT Best (`ee1b91e...`) vs. Canonical Pretrained Baseline  

---

### 1. Quantitative Breakdown Across Languages (300 Blind Prompts)

| Language Segment | Total Prompts | Correct & Relevant | Partially Correct | Incorrect | Unrelated / Hallucinatory | Unsupported Claim | Success Rate (Strict) | Coherence Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **English (`en`)** | 166 | 30 (18.1%) | 105 (63.3%) | 17 (10.2%) | 6 (3.6%) | 4 (2.4%) | **18.1%** | **81.4%** |
| **Hinglish (`hinglish`)**| 68 | 7 (10.3%) | 44 (64.7%) | 4 (5.9%) | 7 (10.3%) | 3 (4.4%) | **10.3%** | **75.0%** |
| **Devanagari Hindi (`hi`)**| 66 | 2 (3.0%) | 58 (87.9%) | 4 (6.1%) | 0 (0.0%) | 2 (3.0%) | **3.0%** | **90.9%** |

---

### 2. Empirical Linguistic Findings

#### A. English Performance
- **Capabilities Established:**
  - Strong natural greetings and assistant identity awareness (`"I am ChakrView, an indigenous neural assistant developed from first principles."`).
  - Python code generation syntax (`def`, indentation, `return`).
  - Basic definition grounding (e.g., RAM vs. storage, photosynthesis).
- **Failure Modes:**
  - Longer reasoning chains drift into loosely associated scientific jargon.

#### B. Hinglish Performance
- **Capabilities Established:**
  - Natural colloquial conversational replies (`"Sab badhiya hai bhai! Aap bataiye aaj kis code ya project par kaam karna hai?"`).
  - Code and tech advice trigger correct conceptual tokens (e.g. `virtual environment`, `modular`, `dependencies`).
- **Failure Modes:**
  - Occasional code-switching syntax breakdown where English nouns clash with Hindi verbs in short completions.

#### C. Devanagari Hindi Performance
- **Capabilities Established:**
  - 100% grammatically valid Devanagari Unicode sequences without phoneme scrambling (a major milestone compared to NR-001/NR-003).
  - Polite greetings and basic sentence structures (`"नमस्ते! मैं ChakrView हूँ..."`).
- **Critical Diagnosis — Language Imbalance Impact:**
  - The SFT dataset was 75.6% English and only 11.8% Hindi.
  - While English demonstrates 18.1% strict factual correctness, Hindi drops to 3.0% strict correctness despite high surface fluency (87.9% partially correct).
  - The model frequently retrieves unrelated Hindi phrases learned from legal, administrative, or privacy passages (e.g. `"भावा निजी पासवर्ड की जानकारी..."` or `"इस तथ्य को रेखांकित करता है..."`) rather than directly answering domain questions like `"भारत की राजधानी क्या है?"`.
  - **Verdict:** True multilingual competence requires scaling the Hindi SFT corpus to parity with English (aiming for $\ge 35\%$ Hindi tokens in the next training cycle).
