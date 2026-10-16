# Prediction Failure and Behavioral Analysis

## Batches 1341–1380: Qualitative Analysis of ChakrMicro Next-Token Predictions

### 1. Quantitative Performance Overview

| Metric | Uniform Random | Unigram Baseline | ChakrMicro-10k |
|---|---|---|---|
| **Cross-Entropy Loss** | 8.3178 | 6.6266 | **5.0630** |
| **Perplexity (PPL)** | 4096.00 | 754.93 | **158.07** |
| **Top-1 Accuracy** | 0.024% | 4.80% | **12.85%** |
| **Top-5 Accuracy** | 0.122% | 13.60% | **30.70%** |
| **Top-10 Accuracy** | 0.244% | 19.21% | **40.39%** |

#### Core Takeaway:
ChakrMicro achieves a **79% reduction in perplexity** relative to the empirical unigram baseline (158.07 vs 754.93) and a **96% reduction** relative to uniform random (158.07 vs 4096.00). In 40.39% of cases, the ground-truth next token is ranked in the model's top 10 choices. This establishes statistically that the 10.8M parameter core **did learn valid multi-token transition patterns and syntactic structure**, rather than merely memorizing fixed sentences.

---

### 2. Failure Mode Taxonomy in Autoregressive Generation

Despite learning basic token transition priors, qualitative inspection of predictions and open-ended greedy generation exposes several key failure modes:

#### Failure Mode 1: Repetitive Token Loops Under Greedy Decoding
- **Symptom:** When generated autoregressively without sampling or temperature penalties, the model falls into cyclic loops:
  - *Prompt:* `ChakrMicro is` $\rightarrow$ `ChakrMicro is. The first of the world of the worl`
  - *Prompt:* `# ChakrMicro` $\rightarrow$ `# ChakrMicrosocial of the word of the world. The c`
- **Root Cause:** In small models (10.8M params), high-probability bigram and trigram transitions dominate the greedy argmax selection. Without repetition penalties, the model lacks sufficient context depth to escape attractor cycles.

#### Failure Mode 2: Code vs. Markdown Indentation Mismatch
- **Symptom:**
  - *Prompt:* `The system` $\rightarrow$ `The system\n    pass\n    >>> linked_list.`
- **Root Cause:** The pretraining mixture in Batches 16–21 included Python code snippets alongside text. The prompt triggered an indentation transition (`\n    pass`), mistaking prose for a Python docstring.

#### Failure Mode 3: Subword Boundary and Technical Identifier Incompletion
- **Symptom:**
  - *Context:* `Refund and Early ` $\rightarrow$ *Target:* `C` (from `Cancellation`). Model predicted `L` (prob 0.0857), `M` (prob 0.0566), `P` (prob 0.0525). `C` was ranked 8th (prob 0.0354).
- **Root Cause:** Technical domain vocabulary is fragmented into small 1-3 byte token chunks. While the model correctly predicts capitalized letters following title words, it assigns mass across all common capital initials rather than sharply focusing on the domain term.

#### Failure Mode 4: Format-Induced Punctuation Bias
- **Symptom:** After `#`, the model frequently predicts another `#` (prob 0.3202) or double space `  ` (prob 0.3096) before space ` ` (prob 0.1409).
- **Root Cause:** Strong markdown header formatting in training data makes the `#` sequence heavily biased toward markdown title nesting (`##`).

---

### 3. Subsystem Breakdown of Perplexity

| Category | Evaluated Documents | Evaluated Tokens | ChakrMicro Loss | ChakrMicro PPL | Unigram PPL |
|---|---|---|---|---|---|
| **Storage** | 5 | 609 | 4.8872 | **132.58** | 719.12 |
| **Billing** | 4 | 580 | 4.8618 | **129.26** | 755.52 |
| **Network** | 5 | 585 | 5.0182 | **151.14** | 711.66 |
| **Security** | 4 | 545 | 5.0747 | **159.92** | 727.60 |
| **SOP** | 4 | 549 | 5.1764 | **177.04** | 863.09 |
| **Compute** | 5 | 591 | 5.2336 | **187.47** | 764.67 |

- **Lowest Perplexity:** Billing (129.26) and Storage (132.58), where structured bullet points and consistent phrasing provide high predictability.
- **Highest Perplexity:** Compute (187.47) and SOP (177.04), which feature diverse hardware terms (`AVX2`, `RoPE`, `SwiGLU`) and process checklists.
