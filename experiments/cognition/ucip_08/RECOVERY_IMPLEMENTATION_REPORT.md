# UCIP-08: Recovery Implementation Report

**Project:** ChakrView  
**Experiment Branch:** `exp/cbr-01-conversational-breakthrough`  
**Target:** Implementation of Neural Recovery Pipeline & Calibrated Evaluator  
**Date:** November 17, 2026  

---

## 1. Overview

The UCIP-08 recovery implementation was designed to address three fatal flaws identified during the forensic audit:
1. **Contaminated / Permuted Training Data:** Datasets that repeated a narrow set of 80 concepts across 10 prefixes.
2. **Evaluator Leakage and False Positives:** Evaluators accepting partial numbers, substrings, and single unigram hits.
3. **Unregularized Cross-Entropy Training:** Over-fitting on token identities that encouraged immediate EOS or loop degenerate states.

---

## 2. Implemented Components

### 2.1 Independent Frozen Benchmark
- **File:** `experiments/cognition/ucip_08/sealed_owner_reality_100.json`
- **SHA-256:** `7607eaf4635232822cdd0b1faad15543664e7c7b339ecd0fda5a7a8e3766ae15`
- **Specification:** 100 high-difficulty, multi-domain probes (15 English Conversational, 10 Explanations, 15 Devanagari Hindi, 10 Roman Hindi, 15 Hinglish, 10 Exact Copying, 10 Elementary Arithmetic, 10 Instruction Following, 5 Semantic Consistency).

### 2.2 Strict Calibrated Evaluator
- **File:** `experiments/cognition/ucip_08/evaluator_owner_reality_100.py`
- **Fixes Applied:**
  - Regex-based word boundary and integer extraction: Prevents `'7'` matching inside `'17'`.
  - Multi-keyword AND logic: Explanations require $\ge 2$ mandatory core concepts rather than single unigram hits.
  - Degeneracy & Repetition Rejection: Outputs containing 3+ consecutive identical tokens are automatically disqualified.
  - Script & Alphabet Strictness: Hindi probes require valid Devanagari density and grammatical verb markers.

### 2.3 Clean Balanced Training Dataset
- **File:** `experiments/cognition/ucip_08/build_clean_dataset.py`
- **Outputs:** `ucip08_train.jsonl` (734 samples) and `ucip08_val.jsonl` (82 samples).
- **Zero-Overlap Guarantee:** Formally checked and asserted $0\%$ stem and prompt overlap with the sealed 100-probe benchmark.
- **Coverage:** Arithmetic operations across diverse entities (books, apples, pens, cards, toys), clear multi-turn style responses, Hinglish technical explanations, Devanagari conversational pairs, and exact reproduction strings.

### 2.4 Regularized Training Pipeline
- **File:** `experiments/cognition/ucip_08/train_ucip08.py`
- **Regularization:**
  - Label smoothing: $\epsilon = 0.05$ applied to cross-entropy loss to prevent overconfidence in repetitive token states.
  - Prompt loss masking: Loss computed strictly over assistant response tokens, ignoring prompt prefix tokens.
  - Gradient clipping: Max norm $1.0$ for numerical stability on CPU.
  - Optimizer: AdamW ($lr=2\times 10^{-5}$, weight decay $0.01$).

---

## 3. Execution Verification

The pipeline was executed deterministically on CPU.
- Candidate Best checkpoint saved at step 125 with validation loss of **1.4452** (down from 1.9126 at step 25).
- Model artifact: `experiments/cognition/ucip_08/checkpoints/candidate_best.pt`
- Release artifact: `releases/ucip_08_candidates/chakrview_ucip08_medium_model.pt`
