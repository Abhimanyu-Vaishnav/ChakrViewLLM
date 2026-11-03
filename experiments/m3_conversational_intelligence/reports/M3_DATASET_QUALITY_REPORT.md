# ChakrView Milestone M3 — Dataset Quality & Provenance Report

**Dataset:** `m3_conversational_mixture_600.jsonl`  
**Training Split:** `m3_train_split_540.jsonl` (540 records, SHA-256 `a5f2c481...`)  
**Validation Split:** `m3_val_split_60.jsonl` (60 records, SHA-256 `53ac3009...`)  
**Manifest:** `m3_split_manifest.json` (SHA-256 `3c7ad754...`)  
**License:** CC0-1.0 (Public Domain Dedication)  

---

## 1. Composition & Category Stratification

The mixture consists of exactly **600 examples** distributed evenly across 15 target conversational categories (40 records per category):
1. `natural_greetings_and_dialogue` (36 train / 4 val)
2. `direct_questions_and_answers` (36 train / 4 val)
3. `instruction_following` (36 train / 4 val)
4. `follow_up_understanding` (36 train / 4 val)
5. `multi_turn_context_retention` (36 train / 4 val)
6. `context_based_qa` (36 train / 4 val)
7. `simple_reasoning` (36 train / 4 val)
8. `arithmetic_and_logical_problems` (36 train / 4 val)
9. `summarization` (36 train / 4 val)
10. `information_extraction` (36 train / 4 val)
11. `explanation_and_clarification` (36 train / 4 val)
12. `format_adherence` (36 train / 4 val)
13. `uncertainty_and_insufficient_info` (36 train / 4 val)
14. `topic_continuation_and_changes` (36 train / 4 val)
15. `bilingual_hindi_hinglish_dialogue` (36 train / 4 val)

---

## 2. Quality Principles Enforced

- **Format Harmonization:** All dialogic examples follow `User: ...\nAssistant: ...` prompt structures to naturally condition conversational turn taking.
- **Length Budget:** Target response lengths average 20–45 tokens, ensuring frequent `<EOS>` supervision to teach clean conversational termination.
- **De-biasing Code:** Minimizes code block boilerplate to eliminate the markdown backtick loop attractor observed in M2.
- **Zero Leakage:** Complete intersection check between the 540 training prompts, 60 validation prompts, and the 60 benchmark evaluation prompts confirmed $0$ leakage ($\text{Train} \cap \text{Bench} = \emptyset$).

