# CBR-SFT-01 — DATASET AUDIT REPORT

---

## 1. Dataset Origin & Split Integrity
- **Corpus:** Exp34 Candidate v3 (`data/sft/exp34_candidate_v3/`)
- **Train File:** `exp34_train_split_900.jsonl` (900 records, 490,143 bytes, SHA-256: `d42c56d1c77a8d6cb8b75216909cc34e49f4346724b6aa276ba4b1d298f8a104`)
- **Val File:** `exp34_val_split_100.jsonl` (100 records, 54,784 bytes, SHA-256: `98803283f26788f26940027c7a467e6101b1d85616b9d2da861857f3387e56ca`)
- **ID Overlap:** 0 records (Disjoint)
- **Prompt Overlap:** 0 records (Disjoint)
- **Evaluation Leakage:** Verified 0 prompt collision against M3 60-task suite (`m3_conversational_tasks.json`).

---

## 2. Category Distribution & Stratification

Each category contains exactly 90 training records (10.0%) and 10 validation records (10.0%):
1. `algorithmic_reasoning` (90 train / 10 val)
2. `arithmetic_reasoning` (90 train / 10 val)
3. `bilingual_hinglish` (90 train / 10 val)
4. `code_debugging` (90 train / 10 val)
5. `constrained_formatting` (90 train / 10 val)
6. `conversational_clarification` (90 train / 10 val)
7. `hindi_comprehension` (90 train / 10 val)
8. `passage_summarization` (90 train / 10 val)
9. `python_programming` (90 train / 10 val)
10. `technical_explanations` (90 train / 10 val)

---

## 3. Response Diversity Verification
- **Unique Prompts:** 900 / 900 (100.0%)
- **Unique Responses:** 900 / 900 (100.0%)
- In contrast to M3's 20.8% response diversity (which repeated 5 answers across 11 categories), Exp34 v3 provides unique conversational completions for every single prompt.

