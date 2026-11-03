# CHAKRVIEW — M5.1 ROOT-CAUSE ANALYSIS & NEXT EXPERIMENT DESIGN
## Definite Forensic Diagnosis and Protocol for Algorithmic Induction Pretraining (AIP)

**Project:** ChakrView  
**Milestone:** M5.1  
**Date:** 2026-10-30  

---

## 1. ROOT-CAUSE CLASSIFICATION & RANKING

| Rank | Potential Root Cause | Status | Empirical Evidence |
|---|---|---|---|
| **1** | **Absence of Pretrained Induction Attention Circuits** | **CONFIRMED** | Mean sensitivity delta is only $\mathbf{0.00014}$. Swapping the entity in the prompt causes virtually zero change in output token probability at the answer position. The attention heads have no functional circuit mapping $[A][B] \dots [A] \to [B]$. |
| **2** | **Superficial Objective Domination by Syntactic Template Tokens** | **CONFIRMED** | **78.96%** of supervised response tokens are static syntax words (`"At station"`, `"resides in"`). Template loss dropped from 5.08 to 1.38, masquerading as learning while entity token loss remained high (2.44). |
| **3** | **Tokenizer Subword Fragmentation of Dynamic Entities** | **CONFIRMED** | 100% of tested entity names and cities fragment into 3 to 6 subword tokens (e.g. `Meera` $\to$ `[Me, er, a]`). This requires multi-step sequential copy operations rather than single-step token lookup. |
| **4** | **Free Autoregressive Decoding Divergence** | **STRONGLY INDICATED** | Teacher forcing achieves **72.0%** token accuracy, but free generation drifts immediately into generic template sequences because the model cannot commit to the first entity token. |
| **5** | **Evaluator Heuristics / Scoring Artifacts** | **REFUTED** | Comprehensive scan revealed **0 false negatives**. The 0/200 failure rate is 100% genuine. |

---

## 2. EXPLICIT ANSWERS TO MANDATED QUESTIONS

1. **Can the model's token probabilities respond to changed entities?**  
   **NO.** Mean probability change when swapping prompt entities was a negligible $\mathbf{0.00014}$ ($0.014\%$).
2. **Does the model rank the correct entity above incorrect alternatives?**  
   **NO.** Mean rank of the target token across diagnostic probes is **205.5** out of 4,096.
3. **Does teacher-forced evaluation show learning even when free generation fails?**  
   **YES.** Under teacher forcing, Arm B predicts the correct next token **72.0%** of the time, proving it learned response templates, even though free generation fails on entity values.
4. **Are dynamic entity tokens receiving meaningful gradient supervision in the existing training objective?**  
   **PARTIALLY.** Entity tokens receive gradients, but represent only **16.53%** of response loss. Gradients are dominated by the 78.96% template tokens.
5. **Does the tokenizer materially contribute to the failure?**  
   **YES.** Subword fragmentation requires compound multi-step induction copying.
6. **Is decoding responsible for any portion of the failure?**  
   **YES.** Greedy decoding suffers from error accumulation on subword fragments.
7. **Does the evaluator reliably measure extraction?**  
   **YES.** Audited with 0 false negatives.
8. **Is there evidence that the architecture itself is the primary bottleneck?**  
   **NO.** The primary bottleneck is the lack of inductive attention circuits in the base weights and tokenizer fragmentation, not the parameter count itself.

---

## 3. RECOMMENDED NEXT EXPERIMENT: EXP36 — ALGORITHMIC INDUCTION PRETRAINING (AIP)

### 3.1 Research Question
Can ChakrMicro (3.44M parameters) learn dynamic token copying when pretrained or continued-pretrained on synthetic, algorithmic sequence-copying tasks where 100% of the loss is applied strictly to copied target spans?

### 3.2 Core Experimental Design
- **Key Innovation:** Pure algorithmic copy tasks without natural language fluff:
  - Task 1: `COPY: [A] [B] [C] -> [A] [B] [C]`
  - Task 2: `LOOKUP: key1=val1, key2=val2. key1 -> val1`
  - Task 3: `REVERSE: [A] [B] [C] -> [C] [B] [A]`
- **Dataset:** 5,000 synthetic algorithmic sequences.
- **Loss Masking:** 100% of loss applied exclusively to the copied target tokens. Zero template tokens.
- **Controls:** Compare ChakrMicro with and without AIP before SFT.
- **Success Criteria:**
  1. Mean sensitivity delta $> 0.20$ on paired prompts.
  2. Grounded extraction accuracy on single-token slot-filling $> 50.0\%$.

