# UCIP-16 Next Experiment Proposal: Corrective Strategy for Conversational Alignment

**Mission ID:** UCIP-16  
**Project:** ChakrView  
**Status:** PROPOSAL ONLY — TRAINING NOT AUTHORIZED  
**Date:** 2026-11-20  

---

## 1. Classification & Ranking of Observed Failure Modes

Based on measurable empirical evidence gathered in UCIP-16, the failure mechanisms are ranked by strength of evidence:

| Rank | Failure Dimension | Strength of Evidence | Empirical Evidence / Finding |
|:---:|---|:---:|---|
| **1** | **Dataset Template Collapse** | **Definitive (100%)** | 84.6% of training responses are exact duplicates. 13 of 18 categories have fewer than 10 unique response strings. The model memorized 15 templates rather than learning instruction following. |
| **2** | **Replay Token Domination** | **Definitive (100%)** | Unmasked Stage C replay sequences accounted for 75.5% of all supervised training tokens, drowning out dialogue gradients and creating pretraining web URL/article attractors. |
| **3** | **Evaluator Fallback Inflation** | **Definitive (100%)** | The reported 70.33% score was caused by an invalid `len(text) > 15` rule in `mixture_sft_engine.py`. True strict semantic accuracy is 0%. |
| **4** | **Autoregressive Markov Attractors** | **High** | Token 898 (`"The "`) has a logit of 5.36 post-prompt vs. `<EOS>` at -0.73 (rank 375), reliably trapping greedy decoding into cyclical loops. |
| **5** | **Architecture Parameter Scale** | **Moderate** | At 3.44M parameters ($d_{model}=192$), parametric factual recall and complex syntax are severely capacity-constrained without external retrieval. |

---

## 2. Top Three Prioritized Corrective Experiments (UCIP-17 Design)

### Experiment A: Diverse Instruction Corpus Construction (Mandatory Precondition)
- **Problem:** Training on 15 repeated template sentences cannot produce conversational capability.
- **Action:** Re-engineer the SFT training corpus so that **100% of responses are unique**:
  - Integrate diverse public-domain conversational dialogues, diverse linguistic variations, varied sentence lengths, and syntactically varied QA pairs.
  - Enforce a strict diversity floor: maximum response multiplicity $\le 1$ across the entire dataset.
- **Estimated Records:** 10,000–15,000 unique prompt-response pairs.

### Experiment B: Balanced Effective Token Loss Formulation
- **Problem:** Stage C replay dominated 75.5% of token gradients, washing out instruction tuning.
- **Action:**
  - Balance loss by **active token count**, not sequence count.
  - Set replay token budget to strictly **10%–15% of active tokens** (e.g. 1 replay chunk per 16 SFT sequences, or dynamic loss weighting $\lambda_{\text{replay}} = 0.15$).
  - Train exclusively from the approved `checkpoint_step_800.pt` base.

### Experiment C: Contrastive Prefix Regularization & Early EOS Supervision
- **Problem:** The model assigns logit 5.36 to generic tokens (`"The "`) and fails to emit `<EOS>` when an answer is complete.
- **Action:**
  - Add explicit negative log-likelihood penalties for generic opening attractors (`"The first of..."`).
  - Introduce short single-turn and concise QA records where `<EOS>` is emitted within 10–20 tokens to force the model to prioritize turn termination.
