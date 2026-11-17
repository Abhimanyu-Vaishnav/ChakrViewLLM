# UCIP-08: Owner Reality Benchmark Report

**Project:** ChakrView  
**Experiment Branch:** `exp/cbr-01-conversational-breakthrough`  
**Specification:** Frozen 100-Probe Held-Out Evaluation Standard  
**Benchmark Artifact:** `experiments/cognition/ucip_08/sealed_owner_reality_100.json`  
**Authoritative SHA-256:** `7607eaf4635232822cdd0b1faad15543664e7c7b339ecd0fda5a7a8e3766ae15`  
**Date:** November 17, 2026  

---

## 1. Design Principles of the Owner Reality Benchmark

The Owner Reality Benchmark was created to establish an uncompromising, contamination-free ground truth for ChakrView's autoregressive language capabilities.

Key constraints applied during construction:
1. **Zero Contamination:** Stems were cross-checked against `ucip06_train.jsonl`, `ucip07_train.jsonl`, and `ucip08_train.jsonl`. No exact prompt overlaps exist.
2. **Multi-Domain Realism:** Covers everyday conversational exchanges, Hindi/Hinglish multilingual prompts, technical computer science questions, word problems in arithmetic, and instruction following.
3. **Calibrated Evaluator Rules:**
   - Single-character arithmetic numbers are matched on whole-word/digit boundaries (`\b9\b` will NOT match `19` or `90`).
   - Technical explanations require multiple independent co-occurring technical concepts.
   - Devanagari Hindi probes require genuine grammatical verbs and Devanagari script density.
   - Outputs exhibiting repetitive degenerate looping (3+ identical tokens) are penalized with an immediate fail.

---

## 2. Category Distribution

```
Domain                          Probe Count
-------------------------------------------
English Conversation                     15
English Explanations                     10
Devanagari Hindi                         15
Roman Hindi                              10
Hinglish                                 15
Exact Copying                            10
Elementary Arithmetic                    10
Instruction Following                    10
Semantic Consistency                      5
-------------------------------------------
TOTAL                                   100
```

---

## 3. Detailed Results by Model

### 3.1 UCIP-06 Medium
- **Score:** 9 / 100 (9.0%)
- **Passes:**
  - English Conversation: "Thank you for explaining that so clearly.", "Good afternoon! How is everything going?"
  - English Explanations: "What does an electric motor do?"
  - Devanagari Hindi: "कंप्यूटर में सीपीयू (CPU) का क्या काम है?", "सूर्य हमें क्या प्रदान करता है?"
  - Roman Hindi: "Subah jaldi uthne ke kya fayde hain?"
  - Hinglish: "Cache memory kya hoti hai aur ye kyu fast hoti hai?", "Unit testing karna kyu important hai?"
  - Semantic Consistency: "Explain how an electric vehicle is powered..."

### 3.2 UCIP-07 Medium
- **Score:** 12 / 100 (12.0%)
- **Passes:**
  - English Conversation: "What are your main capabilities?", "Thank you for explaining that so clearly.", "Good afternoon! How is everything going?", "Tell me a fun fact about space."
  - Devanagari Hindi: "आपका मुख्य उद्देश्य क्या है?", "सूर्य हमें क्या प्रदान करता है?", "भारत की राजधानी का नाम क्या है?", "चंद्रमा रात में क्यों चमकता है?", "इंटरनेट का हमारे जीवन में क्या उपयोग है?"
  - Roman Hindi: "Subah jaldi uthne ke kya fayde hain?"
  - Hinglish: "Cache memory kya hoti hai aur ye kyu fast hoti hai?"
  - Elementary Arithmetic: "What is 5 plus 14?" (Answer: `5 plus 14 is 19.`)

### 3.3 UCIP-08 Candidate
- **Score:** 10 / 100 (10.0%)
- **Passes:**
  - English Conversation: "Thank you for explaining that so clearly.", "Good afternoon! How is everything going?"
  - Devanagari Hindi: "आपका मुख्य उद्देश्य क्या है?", "सूर्य हमें क्या प्रदान करता है?", "चंद्रमा रात में क्यों चमकता है?", "इंटरनेट का हमारे जीवन में क्या उपयोग है?"
  - Roman Hindi: "Subah jaldi uthne ke kya fayde hain?"
  - Hinglish: "Python language itni popular kyu hai?", "Unit testing karna kyu important hai?"
  - Elementary Arithmetic: "You have 18 books and sell 9 of them. How many books remain?" (Answer: `You will have 9 apples left.` -> Correct integer `9` extracted).

---

## 4. Evaluator Verification Audit

Both baseline checkpoints and the recovery candidate were evaluated with `evaluator_owner_reality_100.py`. The evaluation scripts, prompts, raw outputs, and category statistics are permanently logged in:
- `experiments/cognition/ucip_08/eval_ucip06_medium_owner_reality.json`
- `experiments/cognition/ucip_08/eval_ucip07_medium_owner_reality.json`
- `experiments/cognition/ucip_08/eval_ucip08_medium_owner_reality.json`
