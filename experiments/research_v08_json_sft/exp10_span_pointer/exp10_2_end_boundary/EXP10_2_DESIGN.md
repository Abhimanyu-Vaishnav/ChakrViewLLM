# ChakrView Experiment 10.2 — Held-Out Span End-Boundary Diagnostic & Correction Design

**Date**: 2026-10-21  
**Experiment**: Diagnostic and Controlled Start-Conditioned Span Pointer Variant (`exp10_2_end_boundary`)  
**Parent Checkpoint**: Canonical Parent Checkpoint `Step-10000` (`experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`)  
**SHA-256 (Canonical Parent)**: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`  

---

## 1. Problem Statement & Failure Analysis

In Experiment 10, the factorized span pointer achieved:
- **Validation**: 100.0% Value EM (98.99% Span EM)
- **Test A**: 100.0% Value EM (98.65% Span EM)
- **Test B (Held-Out Entities)**: 75.67% Value EM (74.00% Span EM)

The audit revealed that **69 of the 73 Test B failures (94.5%)** were **Premature End Truncations**, where:
1. Start position accuracy was **97.67%** (start selection almost never failed).
2. The predicted span was a **strict prefix** of the target entity (e.g. predicting `softmax_` instead of `softmax_prob`, or `key_vec` instead of `key_vector`).
3. In all 73 failures, the expected end position was within the context window ($e_{\text{gt}} < T$) and within the span constraint ($e_{\text{gt}} - s_{\text{gt}} + 1 \le 8$).
4. In the zero-training oracle-start diagnostic, providing the gold start position yielded the exact same **75.67% end accuracy**.

### Root Cause
In Experiment 10, the end-pointer scoring function was factorized as:
$$\text{Score}_{\text{end}}(j) = \frac{(W_{q,\text{end}} h_{\text{query}})^T (W_{k,\text{end}} h_j)}{\sqrt{d}}$$
where $h_{\text{query}} = h_{T-1}$ is the prompt query vector at the end of the input.

Crucially, **$\text{Score}_{\text{end}}(j)$ was independent of the chosen start position $s$**. The query vector $h_{\text{query}}$ only represents "extract the requested entity", not "given that the entity starts at position $s$, where does it terminate?". On held-out compound words (e.g. `softmax_prob`, `key_vector`), the independent end scorer placed high probability on intermediate punctuation/subword boundaries (`_`, `vec`, `dim`) rather than integrating the contextual continuation from the start position.

---

## 2. The Smallest Justified Correction: Start-Conditioned End Pointer

To resolve this factorization limitation without altering the core architecture or introducing large models:
1. **Backbone**: Frozen or shared ChakrMicro contextual hidden states $H \in \mathbb{R}^{T \times d}$.
2. **Start Pointer**: Identical to Experiment 10:
   $$\text{Score}_{\text{start}}(i) = \frac{(W_{q,\text{start}} h_{\text{query}})^T (W_{k,\text{start}} h_i)}{\sqrt{d}}$$
3. **Start-Conditioned End Query**:
   For each candidate start position $s$, the effective query for the end boundary is conditioned on **both the prompt instruction query $h_{\text{query}}$ and the hidden state at the selected start position $h_s$**:
   $$q_{\text{end}}(s) = W_{q,\text{end}} [h_{\text{query}} + h_s]$$
4. **Conditional End Scoring**:
   $$\text{Score}_{\text{end}}(j \mid s) = \frac{q_{\text{end}}(s)^T (W_{k,\text{end}} h_j)}{\sqrt{d}} - \lambda \cdot (j - s)$$
   where $(j - s)$ provides relative span-length awareness, and candidate masks enforce $s \le j \le s + L_{\max}$.
5. **Joint Inference**:
   $$(\hat{s}, \hat{e}) = \operatorname{argmax}_{s \le j \le s + L_{\max}} \left( \text{Score}_{\text{start}}(s) + \text{Score}_{\text{end}}(j \mid s) \right)$$

---

## 3. Predeclared Evaluation & Baselines

We evaluate on identical records from Experiment 10:
1. **Original Exp 10 Factorized Span Pointer** (Unconditional end scorer)
2. **Exp 10 Oracle-Start Upper Bound** (Diagnostic)
3. **Exp 10.2 Start-Conditioned Span Pointer Variant**

Metrics reported:
- Exact span match ($s^* \land e^*$).
- Decoded value exact match.
- Start position accuracy.
- End position accuracy.
- Premature end truncation rate.
