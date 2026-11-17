# UCIP-09: Generation Pipeline & Autoregressive Diagnostics Report

**Project:** ChakrView  
**Experiment Branch:** `exp/cbr-01-conversational-breakthrough`  
**Date:** November 17, 2026  
**Artifact File:** `experiments/cognition/ucip_09/pipeline_diagnostics_raw.json`  

---

## 1. Executive Summary & Critical Pipeline Discovery

During Phase B, an end-to-end investigation of the autoregressive generation pipeline was conducted across 100 benchmark probes from the frozen Owner Reality Suite under 5 controlled decoding arms.

### 1.1 The Prompt Formatting Mismatch (Root Cause of Step-0 EOS)
Our comparative trace revealed why bare generation produces immediate empty string / `[EOS]` termination:
- **Training Convention:** Training data targets are formatted with turn prefixes: `User: <text>\nAssistant:<response><EOS>`.
- **Bare Direct Inference:** When evaluating with raw prompt strings (`"Hello ChakrView. How are you?"`), the model's next-token distribution places $>99.5\%$ probability on `[EOS]` (Token ID 1) because it has learned that bare sentences without dialogue markers are complete sequences.
- **Formatted Turn Inference:** When wrapped with `User: <prompt>\nAssistant:`, the model immediately transitions to assistant response generation (average generation length increases from 1.2 tokens to **27.8 tokens**).
- **Implication:** Any evaluation or runner that passes unformatted prompts to conversational checkpoints measures structural formatting collapse rather than semantic capability.

---

## 2. Decoding Factorial Experiment (5 Controlled Arms)

We systematically evaluated five decoding conditions across all 100 probes on both UCIP-07 Medium and UCIP-08 Medium:

| Arm Name | Repetition Penalty | No-Repeat N-Gram Size | Description |
|---|---|---|---|
| **Arm A (Raw Greedy)** | 1.00 (Off) | 0 (Off) | Pure unconstrained argmax decoding |
| **Arm B (RepPen Only)** | 1.15 | 0 (Off) | Chat CLI repetition penalty alone |
| **Arm C (NoRepeat3 Only)** | 1.00 (Off) | 3 | Hard 3-gram suppression alone |
| **Arm D (Chat CLI Default)** | 1.15 | 3 | Full default interactive configuration |
| **Arm E (Conservative)** | 1.05 | 0 (Off) | Soft penalty without hard n-gram bans |

### Benchmark Pass Rate Comparison Across Arms

| Decoding Arm | UCIP-07 Medium Accuracy | UCIP-08 Medium Accuracy | Decoding Dynamics Observed |
|---|---|---|---|
| **Arm A (Raw Greedy)** | **12.0%** (12 / 100) | 10.0% (10 / 100) | Highest grammatical fluency, but vulnerable to loop traps on explanations |
| **Arm B (RepPen 1.15)** | 10.0% (10 / 100) | 11.0% (11 / 100) | Suppresses high-frequency glue words (`of`, `the`, `is`) |
| **Arm C (NoRepeat3)** | 10.0% (10 / 100) | **13.0%** (13 / 100) | Breaks phrase stuttering cleanly in UCIP-08 |
| **Arm D (Default Chat CLI)** | 8.0% (8 / 100) | 11.0% (11 / 100) | Compound suppression causes lexical divergence into rare tokens |
| **Arm E (Conservative 1.05)** | 11.0% (11 / 100) | **13.0%** (13 / 100) | Balanced token diversity without tail-token corruption |

### Analysis of Decoding Behavior:
1. **Chat CLI Degradation on UCIP-07:** The combination of `rep_penalty=1.15` and `no_repeat_ngram=3` degraded UCIP-07 accuracy from **12.0% down to 8.0%** (a 33% relative performance loss). Penalizing both simultaneously forces the small vocabulary model to search for distant synonyms, injecting nonsensical vocabulary like `"dogance"` and `"copatience"`.
2. **Optimal Decoding Configuration:** **Arm E (Conservative RepPen 1.05)** and **Arm C (NoRepeat 3 alone)** achieve the highest score (13.0% on UCIP-08), preserving natural language syntax while preventing repetition loops.

---

## 3. Token-Level Probability & Entropy Diagnostics

For representative probes across domains, next-token entropy $H = -\sum p \log p$ was tracked at step 0 (first generated token after `Assistant:`):

| Category & Prompt | UCIP-07 Step-0 Entropy | UCIP-08 Step-0 Entropy | Top-1 Candidate Token (UCIP-08) | Top-1 Prob |
|---|---|---|---|---|
| **English Greeting:** `"Hi! Can you introduce yourself?"` | 0.38 nats | 0.80 nats | `' '` (Leading space) | 89.5% |
| **Explanation:** `"Explain difference between RAM and storage"` | 2.43 nats | 1.66 nats | `' '` (Leading space) | 66.1% |
| **Devanagari Hindi:** `"आपका मुख्य उद्देश्य क्या है?"` | 3.35 nats | 3.97 nats | `' न'` (`Na` - starting 'Namaste') | 16.4% |
| **Exact Copying:** `"Repeat word for word: The sun ..."` | 2.59 nats | 2.91 nats | `' S'` | 29.0% |
| **Elementary Arithmetic:** `"What is 15 plus 7?"` | 0.03 nats | 0.99 nats | `' 1'` (Starting tens digit) | 90.9% |

### Key Diagnostic Observations:
- **Low Uncertainty on English & Arithmetic:** First-token entropy is low ($< 1.0$ nats) on standard English greetings and arithmetic prefixes, demonstrating that the model reliably recognizes task structure.
- **High Uncertainty on Multilingual & Zero-Shot Copying:** Step-0 entropy on Hindi is high ($> 3.9$ nats), indicating severe vocabulary competition across the 4,096 BPE subwords when generating non-English characters.
- **Detokenization Safety:** Identified that invalid byte sequences in partial subwords could raise `UnicodeDecodeError` in strict mode; detokenization must consistently use `errors="replace"` to guarantee runtime stability.
