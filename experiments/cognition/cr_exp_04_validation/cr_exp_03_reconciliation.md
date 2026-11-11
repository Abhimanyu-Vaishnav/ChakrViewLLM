# Reconciliation of CR-EXP-03 Mechanistic Causality Findings

## 1. Executive Summary
CR-EXP-03 conducted mechanistic interpretability and causal intervention experiments on ChakrMicro v0.1 to understand why earlier fine-tuning runs (such as CR-EXP-01 Stage A) suffered from output mode collapse and contextual suppression. This document reconciles those mechanistic findings with prior hypotheses and with the new conversational pilot evaluated in CR-EXP-04.

---

## 2. Key Mechanistic Hypotheses Evaluated

| Earlier Diagnostic Hypothesis (from CR-EXP-01 / CR-EXP-02) | CR-EXP-03 Empirical Interventions | Reconciled Finding |
|---|---|---|
| **Layer 5 FFN acts as an isolated, active negative filter** suppressing context. | Swapping or ablating Layer 5 FFN alone shifted final contextual divergence from 2.72% to only 2.83% and did not recover prediction. | **REFUTED**: Suppression is not localized to Layer 5 FFN. |
| **FFN outputs are completely invariant (static).** | Contextual deltas $\|\Delta \text{FFN}\|_2$ ranged between 0.45 and 1.09 across layers. | **REFUTED**: FFN outputs do carry context-dependent signal. |
| **FFN vectors directly oppose attention vectors.** | Cosine similarity $\cos(\Delta \text{Attn}, \Delta \text{FFN})$ ranged between $-0.15$ and $+0.18$. | **REFUTED**: Attention and FFN deltas are approximately orthogonal. |
| **Cumulative Residual Norm Inflation / Drown-Out.** | Post-FFN residual norms grew from 5.74 (Layer 0) to 25.52 (Layer 5), while contextual delta stayed near ~1.0. | **CONFIRMED**: Over 90% of FFN output energy consists of invariant background bias vectors. Across 6 layers, cumulative invariant addition dilutes the contextual signal-to-noise ratio from 19.2% down to 2.72%. |
| **No weight decay was used in Stage A training.** | Code audit of `train_stage_a.py` line 125 revealed `weight_decay=0.01`. | **RESOLVED**: Weight decay was present in the optimizer call, but mathematically ineffective ($10^{-6}$ decay per step over 100 steps) against unregularized gradient norms. |
| **Candidate hash discrepancy in report.** | Checkpoint search confirmed candidate SHA-256 is `1ac891fbb21af0945751e85f946d3020deb3f01d21969ed298205ba176b9facf`. | **RESOLVED**: A secondary hash string in CR-EXP-02 was a compositional transcription artifact; the underlying disk checkpoint is authentic. |

---

## 3. Implications for Conversational Pilot (`conversational_pilot_best_val.pt`)
1. **Residual Stream Signal Preservation**:
   By training with diverse conversational turns and lower learning rate scheduling in the 12-hour pilot, the model avoided single-token attractor collapse.
2. **Contextual Routing vs. Format Bias**:
   While the candidate learned smooth conversational turns and persona representations without collapsing to `'Cube'` or `'Mumbai'`, the mechanistic constraint identified in CR-EXP-03 remains relevant:
   - In a 3.44M parameter model (d_model=192, 6 layers), deep multi-hop routing without architectural memory or pointer mechanisms remains severely bandwidth-limited.
   - The model learns the surface response distribution (persona, greetings, phrasing) before it acquires deep contextual retrieval for out-of-distribution entity pairs.

