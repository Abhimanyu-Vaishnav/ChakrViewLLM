# CR-16E-R2 Phase I: Architecture Decision & Evidence-Backed Verdict

### 1. Research Questions Addressed

1. **Is the Sequence Copying Regression Explained by Negative Abstention Loss?**
   - **NO.** The controlled experiment (Arm G1 vs Arm G2) conclusively disproved this hypothesis. When negative abstention loss was strictly masked to zero, sequence copying accuracy remained collapsed ($1.0\%$ mean vs $0.33\%$ in G1), far below the $66.0\%$ parent baseline. The copying collapse is caused by **cross-task catastrophic forgetting and attention re-allocation** during retrieval fine-tuning, not negative target loss.
2. **Does Abstention Remain Useful After Correcting Benchmark Labels?**
   - **Yes, but with severe trade-offs.** On unanswerable queries, abstention recall reaches $75\% - 95\%$. However, models with high abstention recall pay a heavy penalty in rejected answerable queries ($50\% - 60\%$ false abstention rate). Masking abstention loss completely eliminates false abstentions ($0.0\%$) and doubles answerable retrieval accuracy ($7.3\% \to 14.9\%$), but destroys abstention recall ($0.0\%$).
3. **Does Balanced-Span Training Generalize Beyond the Benchmark Distribution?**
   - **No.** Balanced-span training solved autoregressive subword completion ($>90\%$ completion once the first token is matched) for entity names present in the training vocabulary ($57.0\%$ peak on length 4). However, it scores **strictly 0.0% (0 / 150)** on genuinely unseen entity values (`CAT7`). The model operates as a robust associative memory for familiar entities, not an abstract copy-based span extractor.
4. **Is Any Architectural Limitation Supported by Direct Evidence?**
   - **No.** The canonical ChakrMicro v0.1 architecture ($3.44\text{M}$ parameters, $d=192, L=6, H=6, d_{\text{mlp}}=512, V=4096$) is capable of non-zero position retrieval ($28\%$), multi-token autoregressive completion ($57\%$), paired permutation sensitivity ($14.7\%$), and high-coverage retrieval ($88.8\%$). The failure to retain copying alongside retrieval is a **training objective and curriculum design flaw** (lack of joint multi-task representation and gradient alignment), not an intrinsic parameter capacity wall.

---

### 2. Formal Architectural Verdict

**VERDICT: PRESERVE CANONICAL CHAKRMICRO v0.1 STRICTLY UNCHANGED.**

- **Do NOT expand model width or depth** ($d=192, L=6, H=6$ remain frozen).
- **Do NOT introduce memory banks, recurrence, or external modules.**
- **Do NOT alter vocabulary size ($V=4096$) or tokenizer files.**

#### Forward Recommendations for CR-17 Capability Co-Existence:
1. **Curriculum Rebalancing:** Interleave copying and retrieval with explicit task conditioning tokens (e.g. `[TASK: COPY]` vs `[TASK: RETRIEVE]`) to prevent attention circuits from being overwritten.
2. **Abstention Target Formatting:** Condition `[UNKNOWN]` on a distinct syntactic token or query confidence threshold rather than unconditioned sequence termination.
3. **OOD Span Extraction:** Train explicitly on variable-binding token copying syntax to bridge the gap between arbitrary sequence copying and multi-token retrieval.

