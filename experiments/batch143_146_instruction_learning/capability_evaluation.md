# Capability Evaluation & Generalization Report: Batch 143–146

**Date:** 2026-10-13  
**Milestone:** Batch 143–146 Instruction Learning & Capability Growth  
**Device:** CPU (PyTorch 2.6+, 14 threads)  

---

## 1. Quantitative SFT Performance Across Experimental Arms

Instruction cross-entropy loss was evaluated on the active answer tokens (excluding masked prompt tokens) across unseen validation and held-out test splits. Catastrophic forgetting was monitored on held-out Stage C raw pretraining text.

| Evaluation Split | Arm A (Baseline) | Arm B (Constant SFT) | Arm C (Warmup+Cosine SFT) | Arm D (Conservative SFT) |
|:---|:---:|:---:|:---:|:---:|
| **Validation Instruction Loss** | 5.4594 | **2.6350** ($-2.8244$) | 3.0761 ($-2.3833$) | 4.0118 ($-1.4476$) |
| **Validation Perplexity** | 234.96 | **13.94** | 21.67 | 55.25 |
| **Held-Out Test Instruction Loss** | 5.0877 | **2.7641** ($-2.3236$) | 3.1184 ($-1.9693$) | 3.9069 ($-1.1808$) |
| **Held-Out Test Perplexity** | 162.02 | **15.86** | 22.61 | 49.74 |
| **Stage C Raw Language Loss** | **3.3053** | 3.3480 ($+0.0427$) | 3.3230 ($+0.0177$) | 3.3101 ($+0.0048$) |
| **Stage C Retention Perplexity** | **27.26** | 28.45 | 27.74 | 27.39 |

### Core Empirical Findings:
1. **Dramatic Instruction Alignment on Teacher-Forced Loss:**
   The canonical baseline (which was only trained on raw web text) exhibits high entropy on ChatML instruction completions ($5.0877$ test loss, PPL $162.02$). Supervised fine-tuning with exact prompt masking dramatically aligns response predictions: Arm B achieves $2.7641$ test loss (PPL $15.86$), and Arm C achieves $3.1184$ (PPL $22.61$).
2. **Minimal Catastrophic Forgetting under Warmup + Cosine (Arm C):**
   Arm C's retention loss on raw Stage C text degraded by only $+0.0177$ nats ($3.3053 \to 3.3230$, PPL $27.26 \to 27.74$). In contrast, un-annealed constant learning rate (Arm B) experienced double the drift ($+0.0427$ nats).
3. **Conservative Arm D Proves Under-tuned:**
   Arm D ($lr=2\times 10^{-5}$) was too conservative for 50 steps, leaving test loss at $3.9069$.

---

## 2. Qualitative Free-Running Generation Analysis

While teacher-forced loss improved substantially, inspecting greedy autoregressive generations reveals critical behavioral differences:

### Baseline (Arm A): Complete Format Failure
When prompted with ChatML (`<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`), the baseline model has no concept of assistant role completion. It immediately outputs Wikipedia wikitext table artifacts:
- *Query:* `"What is the capital of Germany?"`
- *Baseline output:* `|-\n|200\n|200\n|2005\n|-\n|2005...`
- *Query:* `"Write a Python function that returns the absolute value of a number."`
- *Baseline output:* `|-\n| scope=&quot;row&quot; | \n|-\n| scope=`

### Fine-Tuned (Arm B & C): Format Learned, Semantic Capacity Limited
After 50 steps of SFT, the model completely ceases outputting wikitext table markers and adopts assistant-like linguistic structures and terminal delimiters:
- *Code Query:* `"Write a Python function that returns the absolute value of a number."`
  - *Arm B:* `def get_end|>` (Recognized function definition syntax and terminated with delimiter!).
- *Hindi Query:* `"भारत का राष्ट्रीय पशु कौन सा है?"`
  - *Arm B:* `जलवायु\nचित्र:\nचित्रकाल दिल्ली माना जाता है।`
  - *Arm C:* `जलवायु` (terminates cleanly, though factually incorrect).
- *Math & Science Queries:*
  - Models still exhibit repetition loops (`"The total of the number of the..."`), demonstrating that 3.44M parameters is too small to reliably memorize factual world knowledge or execute multi-step arithmetic purely in its weights.

---

## 3. Grounded vs. Parametric Capability Summary

| Capability Dimension | Baseline (Arm A) | Arm B (Constant SFT) | Arm C (Warmup SFT) | Assessment |
|:---|:---:|:---:|:---:|:---|
| **ChatML Formatting** | Complete Failure | **Success** | **Success** | SFT successfully teaches role boundary comprehension. |
| **EOS / Delimiter Termination** | Never (0/6) | 16.7% (1/6) | 16.7% (1/6) | Emergence of `<|im_end|>` stopping behavior. |
| **Code Syntax Prompting** | Wikitext junk | Python token `def` | General English | Python idiom activation confirmed. |
| **Factual Accuracy (Parametric)** | 0% | 0% | 0% | Model parameters do not store reliable facts. |
| **Pretraining Retention** | **100% (Baseline)** | 98.7% | 99.5% | Retention protected by warmup schedule. |
