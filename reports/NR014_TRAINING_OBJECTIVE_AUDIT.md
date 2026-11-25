# CHAKRVIEW RESEARCH REPORT: NR-014 TRAINING OBJECTIVE FORENSIC AUDIT
**Mission ID:** CV-GENESIS-NR-014  
**Date:** 2026-11-25  
**Audit Harness:** `tests/test_training_objective_nr014.py`  

---

## 1. Executive Summary & Verification Suite
Under Phase 1 of Mission `CV-GENESIS-NR-014`, we conducted an exhaustive mathematical and unit-level forensic verification of the causal autoregressive language modeling objective in ChakrGenesis-17M.

All 4 automated regression tests passed cleanly (`Ran 4 tests in 3.137s, OK`):
1. **Causal Attention Masking Invariant:** Verified that logits at step $t \le 2$ are bit-for-bit invariant ($|\Delta| = 0.0$) when modifying future tokens at $t > 2$. Zero future information leakage.
2. **Exact Shifted Autoregressive Alignment:** Confirmed that `shift_logits = logits[..., :-1, :]` matches `shift_labels = labels[..., 1:]` with manual cross-entropy decomposition up to 5 decimal places.
3. **Loss Masking on Prompt and Padding:** Confirmed that prompt tokens and `<PAD>` tokens with label `=-100` contribute exactly zero loss and zero gradient.
4. **Gradient Flow & Weight Updates:** Verified that backpropagation across the 10 Transformer layers and RMSNorm modules produces non-zero, finite gradients without vanishing or exploding.

---

## 2. Invariant Specifications Confirmed
- $\text{Input Tensor}: x = [t_0, t_1, \dots, t_{T-1}]$
- $\text{Target Tensor}: y = [\ell_0, \ell_1, \dots, \ell_{T-1}]$, where prompt positions have $\ell = -100$
- $\text{Shifted Criterion}: \mathcal{L} = \text{CrossEntropy}(\hat{x}_{0:T-2}, y_{1:T-1}, \text{ignore\_index}=-100)$
- `<BOS>` = 0, `<EOS>` = 1, `<PAD>` = 2.
