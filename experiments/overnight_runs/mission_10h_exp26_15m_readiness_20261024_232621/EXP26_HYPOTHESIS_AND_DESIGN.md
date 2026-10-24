# EXP25 FORENSIC REVIEW & EXP26 HYPOTHESIS SPECIFICATION
## Mission: 10-Hour Autonomous Overnight Engineering Run

### 1. Independent Exp25 Audit Summary
The complete artifacts of `exp25_vocabulary_generalization_diagnostic_20261024_232014` were independently audited and verified:
- **Canonical Checkpoint Hash**: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (Bit-exact match).
- **Test 1 (In-Distribution Copying, N=40)**:
  - Exact Matches (EM): 33 / 40 (82.5%) [VERIFIED from raw predictions]
  - Paired-Lookup Accuracy (PLA): 28 / 40 (70.0%) [VERIFIED from raw predictions]
  - EOS Termination: 40 / 40 (100.0%) [VERIFIED]
- **Test 2A (Held-Out Letters M-P, m-p, N=32)**:
  - Exact Matches (EM): 0 / 32 (0.0%) [VERIFIED]
  - Paired-Lookup Accuracy (PLA): 0 / 32 (0.0%) [VERIFIED]
  - Modal Token: `"E"` (15.6%), all emissions belong to the 40 training SFT tokens.
- **Test 2B (Held-Out Digits & Symbols, N=32)**:
  - Exact Matches (EM): 0 / 32 (0.0%) [VERIFIED]
  - Paired-Lookup Accuracy (PLA): 0 / 32 (0.0%) [VERIFIED]
  - Modal Token: `"5"` (15.6%), all emissions belong to the 40 training SFT tokens.
- **Test 2C (Held-Out Letters Q-X, N=32)**:
  - Exact Matches (EM): 0 / 32 (0.0%) [VERIFIED]
  - Paired-Lookup Accuracy (PLA): 0 / 32 (0.0%) [VERIFIED]
  - Modal Token: `"G"` (46.9%), collapsed to training SFT tokens.

### 2. Forensic Diagnosis
Why did the 3.44M causal LM achieve 82.5% in-distribution copying while failing completely (0.0%) on held-out single characters?
1. **Unembedding Vector Bottleneck**:
   In standard autoregressive language modeling with cross-entropy loss, the final token prediction is formed by:
   $$\text{logits} = H_L \cdot W_{\text{out}}^T$$
   During 300 steps of SFT on 40 distinct tokens, only the unembedding vectors $W_{\text{out}}[t]$ corresponding to those 40 tokens receive large positive gradient updates aligned with the context extraction hidden state $H_L$.
2. **Associative Projection Failure**:
   For any held-out token $t' \notin \text{SFT}$, its unembedding vector retains its baseline pretraining geometry. Because the model must emit an argmax logit over 4,096 tokens, the fine-tuned tokens have systematically larger dot-products with the query state than any untouched held-out token, regardless of the token appearing in the context prompt.
3. **Conclusion**:
   Standard cross-entropy SFT over token identities is an **associative lookup table**, not a context-pointer copying mechanism.

---

### 3. Hypothesis Table

| Hypothesis | Theoretical Mechanism | Evidence Needed | Cheapest Discriminating Test |
|---|---|---|---|
| **H1: Output Unembedding Bottleneck** | Cross-entropy updates exclusively elevate seen token unembeddings, starving held-out logits. | Logit distribution of held-out tokens in context vs top-k seen tokens. | Inspect top-5 logits on Test 2A: if target token rank is near random unigram prior, H1 is supported. |
| **H2: Model Capacity Bottleneck (15M Scaling)** | A 3.44M model lacks sufficient capacity to separate task routing from universal copying representations; a 15M model has sufficient subspace capacity. | Training loss trajectory, in-distribution EM, and held-out transfer on ChakrMicro-15M under identical SFT protocol. | Train ChakrMicro-15M for 100-300 steps on the Exp25 dataset and evaluate on Test 1 vs Test 2A/2B/2C. |
| **H3: Pointer-Head Bypass** | Directly scoring context token positions via a pointer query bypasses the vocabulary unembedding matrix entirely, guaranteeing held-out copying. | Span-pointer / position-copying score on context tokens. | Evaluate existing Exp10 pointer model or position-matching head on single-token context copying. |

### 4. Selected Discriminating Experiment (Exp26)
We conduct a bounded, two-arm controlled study:
1. **Arm 1 (15M Parameter Scaling Test)**:
   Does increasing parameters by 4.5x (from 3.44M to 15.7M) allow the standard transformer to generalize copying to held-out tokens without changing the causal LM objective?
   - Hypothesis: Scaling reduces cross-entropy loss faster, but unembedding bias remains an issue without architectural pointer inductive bias.
   - Falsification Criterion: If 15M achieves > 0% held-out PLA on Test 2A, capacity was indeed a bottleneck. If it achieves 0% held-out PLA despite lower training loss, H1 (unembedding bottleneck) is verified.
2. **Arm 2 (Position-Scoring Copy Pilot)**:
   Test whether a context-attention copy mechanism reliably transfers to held-out characters when decoupled from the 4096-class vocabulary projection.
