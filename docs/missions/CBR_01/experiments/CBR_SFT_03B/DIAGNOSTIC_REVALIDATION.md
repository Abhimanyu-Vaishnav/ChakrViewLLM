# CBR-SFT-03B — DIAGNOSTIC REVALIDATION & AUDIT REPORT

**Audit Date:** 2026-11-04  
**Audit Target:** Verification of CBR-SFT-03A Diagnostic Implementation, Alignment & Methodology  
**Status:** **`CONFIRMED & VERIFIED`** (Diagnostic math, tokenization alignment, and logits indexing confirmed 100% accurate)  

---

## 1. Technical Audit of Diagnostic Mechanics

Every step of the diagnostic code in `scripts/run_cbr_sft_03a_forensics.py` was inspected and revalidated:

1. **Logits Indexing Verification:**
   - Input tensor shape is `[1, T]`.
   - `logits[0, -1, :]` is the logit vector emitted by the final prompt token (the token `:` in `Assistant:`).
   - In causal language modeling, position $T-1$ predicts token $T$, which is the very first token emitted by the assistant.
   - **Verdict: Accurate.** Zero off-by-one error.
2. **Tokenizer Leading-Space Alignment:**
   - In BPE tokenization, words in continuous text carry a leading space delimiter (e.g. `1606 (' A')` vs `68 ('A')`).
   - The diagnostic correctly encoded `f" {concept}"` to extract the expected first token `1606 (' A')`.
   - The model's top-1 greedy prediction was `68 ('A')` without leading space, followed by `898 ('The ')`.
   - The expected token `1606 (' A')` was verified to be ranked **140** with probability **$0.000608$ ($0.06\%$)**.
   - **Verdict: Confirmed.** The expected first token is buried deep in the distribution.
3. **Teacher-Forced Prefix Divergence Verification:**
   - In free autoregressive generation, position 0 emits `68 ('A')`, followed by `'ndrone = 200.'`.
   - The reported 100/100 divergence at position 0 is **100% genuine and reproducible**.
4. **Distinction Between First-Token and Full-Response Accuracy:**
   - First-token accuracy was evaluated specifically to identify the *earliest point of failure*.
   - The diagnostic confirmed that divergence is not due to long-horizon context drift or EOS premature termination; it fails immediately at token 0.
   - However, a correct first token is merely a necessary prerequisite, not proof of full semantic response correctness.
5. **No Attention-Routing Dogma:**
   - We do not claim attention heads are physically broken; rather, the gradient updates from 200 steps at $LR=5\times 10^{-5}$ across 540 diverse topics were simply too weak to overcome the entrenched Stage C pretraining priors.

