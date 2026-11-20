# UCIP-17R Independent Dataset Diversity & Forensic Comparison Audit

**Mission ID:** UCIP-17R  
**Project:** ChakrView  
**Repository:** `D:\Project\ChakrView`  
**Dataset Path:** `data/sft/ucip17/` vs `data/sft/ucip14r/`  
**Auditor:** Antigravity Senior Verification Auditor  
**Audit Timestamp:** 2026-11-20T18:37:45+05:30  

---

## 1. Executive Summary

This independent audit evaluated the reconstructed `data/sft/ucip17/` dataset against the baseline `data/sft/ucip14r/` across all 15,000 records.

### Key Comparative Metrics (12,000 Training Records):

| Dimension / Metric | UCIP-14R Baseline | UCIP-17 Reconstructed | Net Difference / Impact |
|---|---:|---:|:---:|
| **Total Training Records** | 12,000 | 12,000 | Unchanged |
| **Exact Unique Prompts** | 2,044 (17.0%) | **3,974 (33.1%)** | **+$94.4\%$ increase** in prompt diversity |
| **Exact Unique Responses** | 1,851 (15.4%) | **3,141 (26.2%)** | **+$69.7\%$ increase** in response diversity |
| **Exact Unique Prompt-Response Pairs** | 2,044 (17.0%) | **3,977 (33.1%)** | **+$94.6\%$ increase** in pair diversity |
| **Exact Duplicate Responses** | 10,149 (84.6%) | **8,859 (73.8%)** | Reduced by 10.8 percentage points |
| **Top Response Repetition** | 491x (Gravity physics) | **308x** (Security refusal) | Reduced concentration peak |
| **Mean Word Count** | 16.9 words | **12.8 words** | Improved short-response representation |
| **Min / Max Word Count** | 1 / 36 words | **1 / 25 words** | Tighter sequence distribution |

---

## 2. Category-Level Diversity Comparison

The reconstruction substantially improved diversity in algorithmic, structured, and contextual categories, but reveals lingering concentration in static explanatory categories:

| Category ID | Description | UCIP-14R Unique Responses | UCIP-17 Unique Responses | Diversity Status |
|---|---|---:|---:|:---:|
| `ARITH` | Arithmetic & Calculations | 704 / 722 (97.5%) | **730 / 742 (98.4%)** | Highly Diverse |
| `FORMAT` | Structured JSON Output | 468 / 468 (100.0%) | **476 / 478 (99.6%)** | Highly Diverse |
| `COPY` | Verbatim Alphanumeric Tokens | 473 / 474 (99.8%) | **471 / 476 (98.9%)** | Highly Diverse |
| `BIND` | Relational Entity Recall | 90 / 611 (14.7%) | **567 / 599 (94.7%)** | **Massive Gain (+80%)** |
| `CORR` | User Correction | 4 / 472 (0.8%) | **335 / 457 (73.3%)** | **Massive Gain (+72.5%)** |
| `CTX` | Context Reading & Extraction | 45 / 733 (6.1%) | **403 / 746 (54.0%)** | **Substantial Gain (+47.9%)** |
| `LOGIC` | Deductive Deduction | 11 / 602 (1.8%) | **72 / 596 (12.1%)** | Moderate Gain |
| `QA` | Open Domain QA | 4 / 828 (0.5%) | **15 / 819 (1.8%)** | Moderate Gain |
| `PYTHON` | Code Syntax Blocks | 8 / 954 (0.8%) | **12 / 945 (1.3%)** | Moderate Gain |
| `EXPL` | Physical Conceptual Explanations | 3 / 841 (0.4%) | **12 / 846 (1.4%)** | Moderate Gain |
| `INST` | Instruction Following | 11 / 964 (1.1%) | **12 / 969 (1.2%)** | Template Bound |
| `CONV` | General Persona & Greetings | 4 / 721 (0.6%) | **10 / 714 (1.4%)** | Template Bound |
| `CLARIF` | Ambiguity Handling | 4 / 485 (0.8%) | **7 / 476 (1.5%)** | Template Bound |
| `ENG` | English Proficiency | 4 / 723 (0.6%) | **4 / 726 (0.6%)** | Template Bound |
| `HIN` | Hindi Dialogue | 5 / 951 (0.5%) | **4 / 973 (0.4%)** | Template Bound |
| `HING` | Hinglish Code-Switching | 4 / 481 (0.8%) | **4 / 472 (0.8%)** | Template Bound |
| `UNCERT` | Uncertainty & Refusal | 4 / 614 (0.7%) | **4 / 586 (0.7%)** | Template Bound |
| `SAN` | Classical Sanskrit | 5 / 356 (1.4%) | **3 / 380 (0.8%)** | Template Bound |

---

## 3. Lexical Distribution & Concentration Findings

1. **Top Repeated Responses:**
   - The top response in UCIP-17 is the refusal template: *"I do not possess access to private encryption keys, confidential credentials, or enclave passcodes"* (**308 occurrences** across 12,000 records, 2.57%).
   - The top 10 repeated responses in UCIP-17 account for **2,245 records (18.7%)**, down from 28.3% in UCIP-14R.
2. **Opening Phrasing:**
   - The most common 3-word opening in UCIP-17 is `"According to the"` (372x, 3.1%), followed by `"Thank you for"` (341x, 2.8%) and `"I do not"` (308x, 2.6%).
3. **Partition Independence:**
   - Verified 100% mutual exclusivity: $Train \cap Val = \emptyset$, $Train \cap Dev = \emptyset$, $Val \cap Dev = \emptyset$.
