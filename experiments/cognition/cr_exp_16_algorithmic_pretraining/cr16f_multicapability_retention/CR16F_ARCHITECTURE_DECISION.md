# CR-EXP-16F: Architectural Decision & Capacity Bottleneck Analysis

## 1. Architectural Verdict: ChakrMicro v0.1 Frozen & Bottleneck Formally Characterized

In accordance with strict project safety guardrails, **no changes have been made to the canonical ChakrMicro v0.1 architecture**. The architecture remains frozen ($3,443,136$ parameters, $L=6, d=192, H=6, d_{\text{mlp}}=512, V=4096$).

However, the empirical evidence accumulated through CR-16A, CR-16C, CR-16D, CR-16D-R, CR-16E, CR-16E-R, CR-16E-R2, and CR-16F decisively establishes that **ChakrMicro v0.1 has reached a fundamental capacity boundary**.

---

## 2. Evidence of the 3.4M Parameter Capacity Bound

1. **Mutual Exclusivity of Copying and Invariant Retrieval:**
   - CR-16A proved ChakrMicro can learn exact sequence copying (66%).
   - CR-16C proved ChakrMicro can learn single-position retrieval (>80% on in-distribution single keys).
   - CR-16F tested 25% and 50% replay in a controlled $2 \times 2$ factorial setup. Replay completely failed to preserve copying ($0 - 1\%$), while retrieval remained severely suppressed ($4 - 14\%$) due to capacity sharing.
   - The same attention heads and projection matrices cannot simultaneously implement an identity-preserving token delay line and a content-addressable dynamic lookup across variable positions.
2. **The Zero-Shot Unseen-Value Wall (CAT7 = 0.0%):**
   - Across every experiment conducted to date, zero-shot retrieval of unseen values has consistently remained at 0/150 ($0.0\% \pm 0.0\%$).
   - A 192-dimensional residual stream with 6 heads cannot generalize variable-binding symbolically to unobserved vocabulary tokens without dedicated architectural inductive biases (such as pointer networks, dual-memory architectures, or drastically higher dimension).
3. **The Abstention Trilemma:**
   - Training with negative abstention targets ($\lambda=1.00$) triggers a global false-abstention collapse on answerable queries (>70% false abstentions).
   - Attenuating negative loss ($\lambda=0.25$) reduces false abstentions but degrades unanswerable recall below 50%.
   - The model lacks the representational dimensionality to decouple query confidence estimation from token generation.

---

## 3. Formal Recommendation for Future Milestones

1. **Immediate Scope (CR-16 Series):** Maintain ChakrMicro v0.1 strictly frozen as the reference micro-architecture baseline.
2. **Subsequent Architectural Roadmap (ChakrMicro v0.2 / ChakrMini):**
   - When the user authorizes an architectural evolution milestone, exploration must focus on:
     - Scaling model capacity ($d \ge 384, L \ge 12, H \ge 8, \approx 15-30\text{M}$ parameters).
     - Auxiliary pointer/copy attention heads or dedicated gating mechanisms for uncertainty/abstention.
     - Separating memory retrieval routing from generation head logits.

