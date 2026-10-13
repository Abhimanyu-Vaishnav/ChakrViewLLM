# Batch 171 — Initial Failure-Mode & Repository Audit
## Neural Copying Bottlenecks, Representation Trade-offs, and Pointer Design Requirements

### 1. Repository State & Baseline Integrity
- **Branch**: `dev/v0.6-v0.7-validation`
- **Head Commit**: `9669f90` (`feat(batch161-170): add learned representations, copy-induction experiments, and memory task engine`)
- **Canonical Model**: ChakrMicro v0.1 (`3,443,136` parameters, `4096` vocab, 6 layers, `d_model=192`, SwiGLU, RoPE, tied embeddings).
- **Canonical Checkpoint**: `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
  - **SHA-256**: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (Verified exact match, immutable).
- **Protected File**: `chakrview/runtime/session.py` remains uncommitted and unmodified.

### 2. Diagnosis of Copy-SFT 0% Exact Match Failure in Batch 166
In Batch 166, Arm A (Copy-SFT) achieved a significant syntactic breakthrough:
1. It completely eliminated the catastrophic wiki-markup collapse of the base checkpoint.
2. It learned clean turn termination (`<|im_end|>`).
However, its held-out exact match on unseen facts was 0.0%. Investigation confirms:
- **Causal Loss Shift & Labels**: The one-token causal shift in `build_chatml_example()` was mathematically correct ($X_{0:T-1} \to Y_{1:T}$).
- **The Core Bottleneck**: Without an explicit pointer/copy mechanism, a 3.44M standard transformer must route context tokens through 6 multi-head attention layers and project back through a static 4096-dimensional vocabulary embedding matrix.
- **Parametric Memorization vs. In-Context Routing**: Instead of dynamically routing attention to the position of the unseen fact in the context, the small network updates its internal feedforward weights to memorize the most frequent training values (e.g., emitting `Ganymede` for Neptune's moon or `3` for Gold's melting point).
- **Conclusion**: Pure vocabulary cross-entropy loss on a 3.44M transformer is insufficient to induce relational pointer copying zero-shot. An auxiliary pointer-generator attention head is required.

### 3. Diagnosis of Learned Representation Baseline (Batch 164)
- `ChakrBiEncoder` achieved 50% ranking accuracy on held-out test facts vs. 100% for the deterministic hash baseline.
- **Root Cause**: The bi-encoder was trained on a small sample of 10 fact triplets. While training triplet loss converged to 0.0, the representations overfit to surface syntax rather than generalized semantic proximity.
- **Action for Batches 171–180**: Scale the contrastive dataset, balance entity domains, and enforce calibrated gating so retrieval only falls back to pure semantics when confidence is high.
