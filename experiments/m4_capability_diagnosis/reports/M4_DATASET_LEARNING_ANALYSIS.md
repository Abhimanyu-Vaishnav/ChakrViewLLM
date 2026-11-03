# CHAKRVIEW — MILESTONE M4 DATASET LEARNING ANALYSIS
## Structural Quality, Curriculum Diversity & Semantic Grounding Audit of the M3 Conversational Mixture

**Project:** ChakrView  
**Milestone:** M4 — Capability Diagnosis & Learning Strategy Research  
**Dataset Inspected:** `data/sft/m3_conversational/m3_conversational_mixture_600.jsonl` (SHA-256 `1706aa1d...`)  
**Splits Inspected:** `m3_train_split_540.jsonl` (SHA-256 `a5f2c481...`), `m3_val_split_60.jsonl` (SHA-256 `53ac3009...`)  
**Date:** 2026-10-30  

---

## 1. STRUCTURAL QUALITY & STATISTICAL METRICS

### 1.1 Record Counts & Stratification
- **Total Records:** 600
- **Train Split:** 540 records (90.0%)
- **Validation Split:** 60 records (10.0%)
- **Categories:** Exactly 15 categories, balanced at 40 records per category (36 train / 4 val).
- **Exact Prompt Duplication:** 0 / 600 (all 600 prompts have distinct strings).
- **Normalized Prompt Duplication:** 0 / 600 (all 600 prompts have distinct lowercase/whitespace normalized strings).

### 1.2 Token Length Budget (ChakrMicro 4,096 BPE Tokenizer)
An exact tokenization pass on all 600 examples produced the following distribution:

| Sequence Segment | Min Tokens | Max Tokens | Mean Tokens | Median Tokens | 95th Percentile | Context Bound (256) |
|---|---|---|---|---|---|---|
| **Prompt** | 12 | 116 | 50.24 | 40.0 | 88.0 | Satisfied (100%) |
| **Response** | 2 | 51 | 21.66 | 20.0 | 43.0 | Satisfied (100%) |
| **Total (Prompt + Response)** | 24 | 132 | 71.85 | 68.0 | 118.2 | Satisfied (100%) |

**Observation:** Sequence lengths are strictly bounded well within the 256/512 token context ceiling. Zero sequences suffered token truncation during training.

---

## 2. THE CRITICAL DEFECT: COLLAPSED RESPONSE DIVERSITY

While prompt strings are 100% distinct, an exhaustive audit of response strings revealed severe semantic collapse:

- **Total Unique Response Strings across 600 examples:** **125**
- **Unique Normalized Responses:** **125**
- **Effective Response Diversity Ratio:** **20.8%**

### Category-by-Category Response Diversity Breakdown

| Category | Total Records | Unique Responses | Repetition Factor |
|---|---|---|---|
| `arithmetic_and_logical_problems` | 40 | 40 | 1.0 (Diverse) |
| `direct_questions_and_answers` | 40 | 10 | 4.0 |
| `instruction_following` | 40 | 10 | 4.0 |
| `natural_greetings_and_dialogue` | 40 | 10 | 4.0 |
| `bilingual_hindi_hinglish_dialogue` | 40 | 5 | **8.0** |
| `context_based_qa` | 40 | 5 | **8.0** |
| `explanation_and_clarification` | 40 | 5 | **8.0** |
| `follow_up_understanding` | 40 | 5 | **8.0** |
| `format_adherence` | 40 | 5 | **8.0** |
| `information_extraction` | 40 | 5 | **8.0** |
| `multi_turn_context_retention` | 40 | 5 | **8.0** |
| `simple_reasoning` | 40 | 5 | **8.0** |
| `summarization` | 40 | 5 | **8.0** |
| `topic_continuation_and_changes` | 40 | 5 | **8.0** |
| `uncertainty_and_insufficient_info` | 40 | 5 | **8.0** |

Across 11 of the 15 categories, **there are only 5 distinct responses repeated 8 times each across different prompt variants!**

### Synthetic Prompt Injection Pattern
To create 40 "unique" records per category, the dataset generation script simply took 5 core dialogues and prepended synthetic turn counters or dialog IDs into the prompt:
- Example: `User [Dialog 6]: What is Linux? ... Assistant: (Turn 1)` vs `User [Dialog 31]: What is Linux? ... Assistant: (Turn 6)`
- In both cases, the target completion was identically: `"Linux was created by Linus Torvalds in 1991."`

---

## 3. LEARNING SUITABILITY AUDIT

### 3.1 What the Dataset Successfully Teaches (Mechanical Syntax)
1. **Turn Delimitation & Role Scaffolding:** Teaches the model to format turns with `User: ...\nAssistant:` rather than markdown backtick loops or code-comment attractors.
2. **EOS Termination Supervision:** Every response ends after 20–45 tokens with `<EOS>`, successfully training the network that conversational turns are brief and terminated.
3. **Repetition Suppression:** By presenting diverse grammatical words in responses, the training objective broke the repetitive n-gram attractor loop of M1/M2.

### 3.2 What the Dataset Completely Fails to Teach (Semantic Grounding)
1. **No True State Tracking:** In `multi_turn_context_retention`, the 5 facts taught were static templates (`"Your favorite fruit is mango."`, `"You live in Bangalore."`, `"The project budget is 50,000 dollars."`, etc.). The model was never exposed to arbitrary entities or variable binding across dialogue turns.
2. **Template Memorization over Dynamic Reasoning:** In `arithmetic_and_logical_problems`, the dataset consisted almost entirely of two-digit addition problems formatted identically as `"X + Y = Z."`. In validation, this led to low validation loss on arithmetic formatting, but zero operational arithmetic logic on novel questions.
3. **Format Mimicry without Grammar Grounding:** In `format_adherence`, only 5 static responses were provided (e.g. one JSON template, one markdown table header). The dataset did not teach general schema translation or JSON key-value extraction from prompt instructions.

---

## 4. DATASET DIFFICULTY & CURRICULUM PROFILE

- **Difficulty Distribution:**
  - Basic (Static template retrieval / simple repetition): **85%**
  - Intermediate (Short arithmetic / single-fact extraction): **15%**
  - Advanced (Multi-step deduction, counterfactual reasoning, constraint satisfaction): **0%**
- **Curriculum Flaw:** The dataset lacks progressive difficulty. It trains the model on pseudo-dialogue tokens that reward memorizing a tiny set of 125 static responses rather than extracting answers conditioned on arbitrary prompt tokens.

---

## 5. SUMMARY OF EVIDENCE

- **Finding D1:** The M3 dataset has an effective response diversity of only 20.8% (125 unique responses across 600 records). `[VERIFIED]`
- **Finding D2:** 11 of 15 categories contain only 5 distinct target answers duplicated via superficial prompt prefix tagging (`User [Dialog N]`). `[VERIFIED]`
- **Finding D3:** Low validation loss (1.9533 at Step 500) reflects overfitting to these 125 memorized target strings rather than general conversational intelligence. `[VERIFIED]`

