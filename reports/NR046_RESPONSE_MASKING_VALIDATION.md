# CHAKRVIEW NR-046: RESPONSE-MASKED SFT VALIDATION REPORT

**Mission Reference:** `CHAKRVIEW NR-046`  
**Phase:** WP-2 (Response-Masked SFT Supervision)  
**Date:** 2026-11-27  
**Validation Suite:** `experiments/genesis_stage_rc2_046/test_response_masking_unit.py`  

---

## 1. Objective

To empirically verify that response-only loss supervision functions strictly as designed before launching RC2 training:
1. Prompt tokens contribute zero loss.
2. Target token shift aligns properly with causal language modeling.
3. Assistant response tokens (including the very first token emitted given the full prompt context) are supervised.
4. Padding tokens are assigned `-100` and contribute zero loss.
5. Gradients are finite and stable across backpropagation.
6. A controlled toy sample converges to near-zero loss.

---

## 2. Mathematical & Tensor Alignment Specification

For an input sequence consisting of:
$$\mathbf{x}_{\text{prompt}} = [p_0, p_1, \dots, p_{P-1}], \quad \mathbf{x}_{\text{response}} = [r_0, r_1, \dots, r_{R-1}]$$
Concatenated full sequence:
$$\mathbf{X} = [p_0, \dots, p_{P-1}, r_0, \dots, r_{R-1}]$$

Under standard autoregressive causal training with sequence length $L = P + R$:
- **Model Input:** $\mathbf{X}_{0:L-1} = [p_0, \dots, p_{P-1}, r_0, \dots, r_{R-2}]$
- **Labels:** $\mathbf{Y} = [\ell_0, \dots, \ell_{L-2}]$ where:
$$\ell_i = \begin{cases} -100 & \text{if } i < P - 1 \quad (\text{predicting prompt tokens } p_1 \dots p_{P-1}) \\ r_{i - (P - 1)} & \text{if } i \ge P - 1 \quad (\text{predicting response tokens } r_0 \dots r_{R-1}) \\ -100 & \text{if } i \ge \text{actual sequence length (padding)} \end{cases}$$

Notice critically that at position $i = P - 1$, the model has consumed all prompt tokens $[p_0, \dots, p_{P-1}]$. Its prediction at this step is the first response token $r_0$. Thus, index $P-1$ in labels receives $r_0$ (active supervision), while all prior indices receive $-100$.

---

## 3. Unit Test Verification Results

| Test ID | Verification Target | Expected Condition | Observed Result | Status |
| :---: | :--- | :--- | :--- | :---: |
| **UT-01** | Prompt Token Masking | Target IDs $< P-1$ are `-100` | Target IDs $0 \dots 17$ are `-100` | **PASSED** |
| **UT-02** | First Response Token | Target ID at $P-1$ is $r_0$ | Target ID at index 18 is token ID `55` ('4') | **PASSED** |
| **UT-03** | Padding Inactivation | All padding indices are `-100` | Verified: all padding tokens mapped to `-100` | **PASSED** |
| **UT-04** | Gradient Non-finiteness | No NaN or Inf in model parameters | All gradients finite and bounded | **PASSED** |
| **UT-05** | Loss Attribution | Loss computed solely on response | Computed loss: $8.4323$ (response-only) | **PASSED** |
| **UT-06** | Convergence Overfit | 2-sample controlled overfit $< 0.1$ | Initial loss $8.3755 \to \mathbf{0.0139}$ in 80 steps | **PASSED** |

---

## 4. Conclusion

The response-masking implementation completely resolves the unmasked prompt gradient flaw identified in Stage-L/RC1.
