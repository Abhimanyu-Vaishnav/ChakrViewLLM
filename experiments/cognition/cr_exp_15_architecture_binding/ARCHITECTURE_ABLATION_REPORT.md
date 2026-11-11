# CR-EXP-15 Architecture Ablation Report
## Head Dimension, Capacity Scaling, and In-Context Routing Capacity

**Mission ID:** CR-EXP-15  
**Date:** November 2026  
**Status:** Completed & Empirically Verified  

---

## 1. Executive Summary

CR-EXP-15 investigated the structural mechanism of entity binding in ChakrMicro. Across historical experiments (CR-EXP-12 through CR-EXP-14), ChakrMicro persistently failed strict subject-value association on unseen entities (scoring 0/1,100 on non-abstention probes). CR-EXP-15 systematically decoupled and ablated the primary structural and optimization hypotheses:

1. **Training Objective & Curriculum:** Does curriculum progression (pure copying $\to$ pairwise selection $\to$ static binding $\to$ multi-turn state tracking) resolve entity binding?
2. **Attention Head Dimension ($d_{\text{head}}=32$ vs $d_{\text{head}}=64$):** Does doubling the subspace rank per head enable pointer routing or in-context induction?
3. **Model Width & Parameter Capacity ($192\text{-dim } / 3.44\text{M}$ vs $256\text{-dim } / 5.77\text{M}$):** Does scaling parameter capacity by $+67.5\%$ unlock dynamic state tracking?
4. **Pretraining Priors vs Scratch Training:** How do fine-tuning on top of language model checkpoints compare to fresh architecture arms?

### Definitive Empirical Results on Sealed 1,200-Probe Benchmark

| Architecture Arm / Candidate | Model Parameters | $d_{\text{model}}$ | Heads | $d_{\text{head}}$ | Best Val Loss | Non-Abstention Strict Acc (Denom=1050) | Abstention Acc (Denom=150) | Overall Acc (Denom=1200) |
|---|---|---|---|---|---|---|---|---|
| **CR-12 Parent Baseline** | 3,443,136 | 192 | 6 | 32 | 5.7251 | **0.000%** (0/1050) | **0.000%** (0/150) | **0.000%** (0/1200) |
| **Candidate A (Standard LM Control)** | 3,443,136 | 192 | 6 | 32 | 5.0537 | **0.000%** (0/1050) | **0.000%** (0/150) | **0.000%** (0/1200) |
| **Candidate B (Copy & Select Curriculum)** | 3,443,136 | 192 | 6 | 32 | 5.1228 | **0.000%** (0/1050) | **0.000%** (0/150) | **0.000%** (0/1200) |
| **Candidate C (Static Binding Curriculum)** | 3,443,136 | 192 | 6 | 32 | 4.9690 | **0.000%** (0/1050) | **0.000%** (0/150) | **0.000%** (0/1200) |
| **Candidate D / Arm A (Dynamic Updates)** | 3,443,136 | 192 | 6 | 32 | **1.8418** | **0.000%** (0/1050) | **0.000%** (0/150) | **0.000%** (0/1200) |
| **Arch Arm B ($d_{\text{head}}=64$)** | 3,443,136 | 192 | 3 | 64 | 6.0375 | **0.000%** (0/1050) | **0.000%** (0/150) | **0.000%** (0/1200) |
| **Arch Arm C ($256\text{-dim}, d_{\text{head}}=64$)** | 5,767,424 | 256 | 4 | 64 | 5.2227 | **0.000%** (0/1050) | **0.000%** (0/150) | **0.000%** (0/1200) |

---

## 2. Key Scientific Findings

### Finding 1: Changing Head Dimension ($d_{\text{head}}=32 \to 64$) Does Not Induce Pointer Routing From Scratch
Halving head count from 6 to 3 while doubling head dimension from 32 to 64 (Arch Arm B) keeps total parameters identical ($3.44\text{M}$), but trained from scratch it achieves a validation loss of only $6.0375$ (vs $1.8418$ for Candidate D fine-tuned on parent). On the sealed benchmark, Arm B collapsed into repeating syntax tokens (`Assistant: Assistant: Assistant:`). In an autoregressive transformer without pre-trained induction circuits, wider head subspaces do not autonomously discover entity-value pointer mechanisms during short fine-tuning.

### Finding 2: Scaling Width to 256 Dimensions ($5.77\text{M}$ Parameters) Alone Is Insufficient
Arch Arm C expanded model width by 33% ($192 \to 256$) and parameter count by 67.5% ($3.44\text{M} \to 5.77\text{M}$). Its validation loss reached $5.2227$, showing rapid optimization, but its generative output suffered severe modal collapse (`AAAAAAAAAAAAAAA...`). Capacity scaling without structured weight initialization or multi-phase pretraining fails to establish lexical grounding.

### Finding 3: The Parametric Prior Bottleneck
Candidate D (fine-tuned with dynamic updates) achieved the lowest validation loss ($1.8418$) and learned the conversational and syntactic frame (`<Entity> is currently in <Location>`), producing syntactically pristine multi-turn answers. However, when tested on unseen entity names and cities, Candidate D output memorized training entities and cities (`Amita is currently in Aurangabad.`, `Kavita is currently in Amritsar.`) instead of copying the entities from the prompt context.

This proves that **the failure mode is not representational incapacity, but attention routing vs parametric memory dominance**. The model relies entirely on its feedforward and embedding weights (parametric memorization) to output entities, rather than dynamically copying tokens from the query context via induction heads.

