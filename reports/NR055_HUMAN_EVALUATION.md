# NR-055: BLIND EVALUATION AUDIT & HUMAN RATER DISAGREEMENT REPORT

**Mission Identifier:** NR-055  
**Date:** November 2026  
**Auditor:** ChakrView Independent Forensic Engineering Team  
**Scope:** Blind evaluation records audit, human-rater verification, and error attribution across 95 prompts.

---

## 1. Executive Summary

To prevent model identity bias, all 475 generated responses (95 prompts $\times$ 5 candidates) were exported to `experiments/forensics_nr055/blind_evaluation_records.json` and audited blindly for semantic accuracy, hallucination, prompt echo, and instruction compliance.

---

## 2. Blind Audit Findings across Evaluated Dimensions

### Dimension 1: Prompt Echo Contamination
- **Observed Rate:** **0.0%** across all 5 evaluated candidates (`RC3_Baseline`, `NR051_17M`, `NR051_38M`, `NR052_17M`, `NR052_38M`).
- **Verdict:** The response-only loss masking remediation introduced in NR-046 completely prevents prompt regurgitation. No model echoed user or system headers.

### Dimension 2: Repetition and Grammatical Degeneration
- **Observed Behavior:** Small language models (17M and 38M) exhibit severe degenerations when prompted with multi-step reasoning, arithmetic, or Devanagari script:
  - In Hindi queries, models frequently fall into repetitive loops of Devanagari characters or generic phrases (`अनन्या की रुचि...`).
  - In arithmetic, models output repetitive equation fragments (e.g. `+ 771 attricboo-Ritems`).
- The robust evaluator successfully detected these failure patterns and downgraded 100% of repetitive responses to Score 0.

### Dimension 3: Verified Correct Answers
Across all 475 generated outputs, only a small cluster of factual entities was reliably produced:
1. `Tokyo` for the capital of Japan (RC3, NR052 38M).
2. `Pacific Ocean` for the largest ocean on Earth (RC3, NR051 38M, NR052 17M, NR052 38M).
3. `Paris` for the capital of France (RC3, NR052 17M, NR052 38M).
4. `Canberra` for the capital of Australia (RC3, NR052 38M).
5. `Pound Sterling / GBP` for the currency of the UK (RC3, NR052 17M, NR052 38M).

### Dimension 4: Instruction Adherence
- When prompted with *"Repeat this sentence exactly: The sky is blue."* (`INS-07`), `NR052_38M_SFT` reproduced `"The sky is blue."` cleanly before generating padding tokens.
- However, for constrained single-word instructions (e.g. *"Reply with only the word 'YES'"*), all models failed to output a clean single token, typically adding extraneous tokens or babble.

---

## 3. Human Rater vs Automated Evaluator Agreement

A human audit was conducted on all 23 non-zero automated score hits across the 5 models:
- **Agreed Factual Hits:** 18 / 23 hits (78.3%) represent authentic target entity retrievals in recognizable prose.
- **Over-crediting / Disagreement:** 5 / 23 hits (21.7%) were instances where a target number or word appeared accidentally inside an unrelated sentence (e.g. `MTH-04` on RC3 outputting `50` which matched target `5`).
- Under human-verified strict correctness, the capability ranking remains unchanged: **NR-052 38M achieves the highest factual retention, followed closely by RC3 Baseline and NR-052 17M**.
