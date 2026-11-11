# CR-16DR Systematic Failure Mode Analysis

### 1. Unified Failure Mode Taxonomy
Through the forensic audit, Part B diagnostic interventions, and 5-seed replication across 20 models, four distinct failure mechanisms have been isolated:

```
Failure Hierarchy in Consolidated ChakrMicro:
1. Multi-Token Subword Autoregressive Completion Drift (~50% of benchmark errors)
2. Absolute Positional Attention Lock-In (~30% of benchmark errors)
3. Strong Copying Bias Suppressing Abstention (~15% of benchmark errors)
4. Distractor Context Proximity Capture (~5% of benchmark errors)
```

---

### 2. Comprehensive Diagnostics of Each Failure Mode

#### A. Multi-Token Subword Completion Drift (G2 & G4)
- **Mechanism:** When evaluated on unseen values (e.g. `Mercury`, `Shield`, `Spring`, `Cotton`), logit inspection proves that the model **correctly points to the value and predicts the first subword token with $>86\%$ probability** (`M`, `S`, `Co`).
- However, because the value was never trained as an end-to-end multi-token span, the causal LM prior overrides continuation:
  - `M` $\to$ `an` $\to$ `g` $\to$ `o` (`Mango`)
  - `S` $\to$ `hi` $\to$ `p` (`Ship`)
- **Remedy:** Pretrain with multi-token value spans and subword boundary delimiters.

#### B. Absolute Positional Attention Lock-In
- **Mechanism:** In single-fact queries (`Store: K = V. Query: K.`), the target fact resides at a constant token offset.
- When tested with fact permutations (`Store: K2 = V2. Store: K1 = V1. Query: K1.`), models scored **0/100 on both orders**. Models succeeded only when the target fact was in position 1 (46%–58%), and completely failed when it was in position 2 (0%).
- The model attends by absolute position rather than performing an invariant content-based associative search.
- **Remedy:** Train with variable-length contexts and randomized fact orderings.

#### C. Loss of Calibrated Abstention (Over-Eager Copying)
- **Mechanism:** Replay copying examples train the network to copy tokens from context.
- When an unbound key is queried (CAT9), the attention distribution is diffuse, and the model samples one of the visible values in context rather than emitting `[UNKNOWN]`.
- **Remedy:** Include explicit negative copying samples (`Copy: corrupted -> [UNKNOWN]`) and negative key-value queries during consolidation.

