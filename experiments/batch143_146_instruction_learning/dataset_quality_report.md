# Dataset Quality & Provenance Report: Batch 143–146

**Date:** 2026-10-13  
**Milestone:** Batch 143–146 Instruction Learning  
**Dataset Name:** `ChakrView Curated Multi-Domain SFT v0.7`  
**Dataset Location:** `experiments/batch143_146_instruction_learning/data/`  

---

## 1. Dataset Design & Governance

The dataset was curated specifically to test multi-domain instruction learning without leaking benchmark tasks or inflating metrics through synthetic templates.

### Key Governance Standards:
- **No Uncontrolled Web Scraping:** All examples are verifiable facts, standard programming idioms, deterministic arithmetic problems, or grounded science reading comprehension queries.
- **Explicit Provenance & Licenses:** Each example specifies its origin (`provenance`), license (`MIT`, `CC-BY-SA-4.0`, or `Public Domain`), and semantic task family.
- **Zero Contamination Policy:** 
  - Train split ($17$ records): SHA-256 `eb6a5f...`
  - Validation split ($5$ records): SHA-256 `58a1f7...`
  - Held-out Test split ($6$ records): SHA-256 `0ce280...`
  - String prompt overlap: **0 records** ($0.0\%$).
  - Full example hash overlap: **0 records** ($0.0\%$).

---

## 2. Domain & Split Composition

| Domain | Train Records | Val Records | Test Records | Task Families Covered | Verifiable Verification Method |
|:---|:---:|:---:|:---:|:---|:---|
| **English Comprehension** | 4 | 1 | 1 | `fact_qa`, `definition`, `concept_qa` | Ground truth fact match / semantic check |
| **Python / Coding** | 4 | 1 | 1 | `python_basic`, `python_list`, `python_algo` | Syntax compilation (`ast.parse`) & deterministic execution |
| **Mathematics** | 4 | 1 | 1 | `arithmetic`, `word_problem`, `algebra_basic` | Exact numerical calculation |
| **Hindi / Indic** | 3 | 1 | 1 | `hindi_qa`, `hindi_grammar` | Ground truth Devnagari string match |
| **Grounded Science** | 2 | 1 | 2 (1 Positive, 1 Negative) | `context_grounded`, `context_insufficient` | Grounding support check & unstated-fact refusal |
| **Total** | **17** | **5** | **6** | **11 Families** | Strict multi-modal verification |

---

## 3. Delimiter & Loss-Masking Audit

- Every example is formatted with standard ChatML delimiters:
  - Prompt: `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`
  - Response: `{response}<|im_end|>\n`
- All prompt tokens receive `ignore_index = -100`. Gradients are computed strictly over the response tokens and the closing `<|im_end|>\n` delimiter.
- Format verification confirmed bit-exact alignment between `InferenceEngine` greedy prompt expansion and the training loss mask.
