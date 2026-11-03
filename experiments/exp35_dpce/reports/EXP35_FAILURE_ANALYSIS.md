# CHAKRVIEW — EXP35 FAILURE ANALYSIS & SCIENTIFIC DIAGNOSIS
## Investigation of Why 100% Unique Extractive SFT Failed to Produce Dynamic Generalization

**Experiment:** Exp35-DPCE  
**Milestone:** M5  
**Result:** 0/200 (0.0%) Grounded Extraction on Unseen Benchmarks  
**Date:** 2026-10-30  

---

## 1. THE PHENOMENON: LOSS DIVERGENCE VS GENERALIZATION GAP

During 500 SFT steps on the 1,000-record DPCE dataset:
- **Arm A (M3 Step 250 base):** Train loss dropped from 5.2928 to **2.0855**; Val loss dropped from 5.2928 to **2.5216**.
- **Arm B (M2 Arm A base):** Train loss dropped from 4.9529 to **1.0980**; Val loss dropped from 4.9529 to **1.4006** (a massive $71.7\%$ reduction).
- **Physical Generation Stability:** Both arms maintained superb physical generation metrics:
  - 4-gram repetition collapsed to **0.0207** (Arm B) and **0.0308** (Arm A).
  - Unique token ratio remained at **0.9033** (Arm B) and **0.9068** (Arm A).
  - Clean `<EOS>` termination reached **86.7%** (26/30 prompts).

**Yet, on the independent 200-task benchmark, both arms achieved exactly 0/200 (0.0%) extraction accuracy.**

---

## 2. FORENSIC ROOT-CAUSE ANALYSIS

### 2.1 The N-gram Vocabulary Fragmentation Bottleneck
- ChakrMicro uses a **4,096-vocabulary Byte-Pair Encoding (BPE)** tokenizer.
- In this compact vocabulary, common Indian names (e.g., `Meera`, `Zoya`, `Tariq`, `Bhavna`) and synthetic cities (`Kashipur`, `Vrindavan`) **do not exist as atomic single tokens**.
- For example:
  - `Meera` is tokenized into 3 separate subword tokens: `[2260, 283, 100]` (`Me`, `er`, `a`).
  - `Zoya` is tokenized into: `[87, 1021, 62]` (`Z`, `oy`, `a`).
- In order to "extract" an entity like `Meera` across sequence context, a causal transformer must execute **multi-step attention copy operations across multiple consecutive subword token steps**.
- At a hidden dimension of $d_{\text{model}}=192$ with 6 heads ($d_{\text{head}}=32$), the attention heads lack the representation dimensionality to implement both induction head copying and subword concatenation simultaneously without massive pretraining exposure.

### 2.2 Template Syntax Mimicry vs Causal Variable Binding
Inspection of Arm B's generations reveals that the model successfully learned the **syntactic surface patterns** of the training data:
- Training example: `"Officer Aarav (Badge 201) is stationed in Somgarh."`
- Model prediction: `"At station 271 is stationed in Chandra."`
- In every case, the model outputs grammatically plausible sentences starting with `"At station..."`, `"Officer..."`, or `"User..."`, matching the top starting tokens of the training responses.
- However, **the specific entity names and numbers inside the completion are not causally copied from the prompt context**. Instead, the model outputs synthetic fragments sampled from its internal unigram/bigram priors.

### 2.3 The Pretraining Representation Horizon
- Fine-tuning (SFT) can only **steer and align representations that already exist** in the base model.
- Because ChakrMicro's pretraining was limited to 10,000 steps on a small foundational corpus, the base weights never developed general **induction heads** (the two-layer attention circuit responsible for copy-paste operations $[A][B] \dots [A] \to [B]$).
- Without pre-existing induction circuits, 500 steps of SFT simply teaches the model to memorize high-frequency template bigrams rather than dynamically copying novel entities from prompt context.

---

## 3. SCIENTIFIC VERDICT & IMPLICATIONS FOR THE PROJECT

1. **The Null Hypothesis is Supported:**  
   Providing 100% unique responses in SFT is necessary, but **insufficient on its own** to induce dynamic variable binding in a model that lacks pre-trained induction circuits.
2. **Architecture vs Data Reality:**  
   The bottleneck is not merely parameter count, but the **lack of algorithmic copying supervision in pretraining**.
3. **Strategic Pivot:**  
   Future capability improvements cannot rely on SFT alone. Pretraining must explicitly incorporate algorithmic copying and induction tasks, or the runtime must utilize external memory/retrieval contracts to supply grounded entity spans directly.

