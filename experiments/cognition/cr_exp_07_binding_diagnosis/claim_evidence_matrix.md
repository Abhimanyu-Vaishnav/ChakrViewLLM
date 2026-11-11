# Claim-Evidence Matrix — CR-EXP-07

**Experiment ID**: CR-EXP-07  
**Investigation Focus**: Subject Identity & Context Binding Diagnosis  
**Date**: 2026-11-06  

---

## 1. Primary Empirical Findings & Evidence Status

| # | Scientific Assertion / Hypothesis | Status | Direct Causal / Empirical Evidence |
|---|---|---|---|
| **E1** | **ChakrMicro represents subject identity at intermediate hidden layers.** | **CONFIRMED** | Changing context subject from Ravi to Aman produces a 15.43% relative divergence in Layer 5 residual stream representations and a 0.12–0.48 logit shift under causal replacement. |
| **E2** | **ChakrMicro represents associated contextual values at intermediate layers.** | **CONFIRMED** | Changing context value from Delhi to Jaipur produces a 7.49% divergence at Layer 5 residual representations, and causal value activation replacement shifts logits towards the source value. |
| **E3** | **ChakrMicro forms an algebraic representation of the subject-value interaction.** | **CONFIRMED (Representational Only)** | A 2x2 counterfactual Difference-in-Differences design yields a non-zero interaction tensor ($\|\text{Interaction}\|_2 = 0.3494$, 2.29% of mean vector norm). The geometry reflects the paired binding. |
| **E4** | **The subject-value interaction is successfully translated into final output logits.** | **REFUTED** | Despite the 2.29% representational interaction, target entity ranks remained $>50$ across all 2x2 conditions (ranks 57 to 239). The top-1 prediction is overwhelmingly dominated by the conversational prior token (`' '`, $P > 0.73$). |
| **E5** | **The model follows counterfactual reassignment during text generation.** | **REFUTED** | In Stage 5, prompting with 5 different cities for Ravi (Delhi, Jaipur, Mumbai, Chennai, Pune) produced identical generic text: `"Please result me know what task you are working on"`. Reassignment success was **0.0%**. |
| **E6** | **Causal activation replacement at entity positions directly alters the top-1 generation.** | **REFUTED** | In Stage 6, swapping subject or value activations at Layers 0, 2, 4, 5 shifted logit vectors, but **never shifted the top-1 generated token** away from conversational filler. |
| **E7** | **Contextual information weakens due to positional decay ("lost in the middle").** | **REFUTED** | Context position tests across Beginning, Middle, and End show near-identical target ranks (32 vs 30 vs 29) and near-zero probabilities ($0.00018$ vs $0.00019$ vs $0.00031$). The failure is position-invariant. |
| **E8** | **The dominant failure mechanism is output-head prioritization overridden by memorized priors.** | **CONFIRMED** | Stage 7 information-flow tracing confirms contextual SNR is preserved through Layer 5 (10–14% SNR, 7.5% logit divergence). The contextual signal reaches the final layer but is drowned out by high-frequency background token logits. |
| **E9** | **The model preserves subject identity across distractors or unseen templates.** | **REFUTED** | In Stage 8, across 450 independent out-of-distribution prompts (distractors, reassignments, unseen entities, unseen templates), exact relational retrieval accuracy was **0.0%**. |

