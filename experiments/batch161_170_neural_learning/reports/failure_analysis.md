# Batch 170 — Consolidated Failure & Capability Analysis
## Learned Representations, Neural Copy Induction, Memory Tasks & Architectural Boundaries

### 1. Empirical Findings on Learned Representations (Batches 163–164)
- **Component**: `ChakrBiEncoder` (`chakrview/runtime/learned_embedding.py`) trained from scratch with Triplet Margin Loss on partitioned facts.
- **Result on Held-Out Test Entities**:
  - Deterministic Hash Baseline: 100% Ranking Accuracy (Margin separation: +0.1949).
  - Learned Bi-Encoder (from scratch, tiny dataset): 50% Ranking Accuracy (Margin separation: +0.0415).
- **Scientific Conclusion**:
  A tiny 2-layer transformer trained on 10 fact triplets easily overfits to training syntax and lacks generalized semantic transfer to held-out entities compared to the deterministic hash provider.
  **Decision**: We preserved both providers, integrating `LearnedEmbeddingProvider` into `RetrievalAdapter` as a configurable option while retaining the reliable baseline. In hybrid search, pure semantic fallback is calibrated with a high confidence threshold ($0.88$) to prevent false positive retrievals on absent queries.

### 2. Empirical Findings on Neural Copy Induction (Batches 165–166)
- **Component**: Fine-tuning Arm A (`ChakrMicro` 3.44M initialized from canonical `step_10000.pt`) using causal loss masking over target answer tokens and the termination token `<|im_end|>`.
- **Result on Held-Out Test Entities**:
  - Arm B (Zero-shot control): 0.0% Exact Match. Output collapsed into repetitive wiki markup (`|- \n | scope=...`).
  - Arm A (Copy-SFT): 0.0% Exact Match on unseen test facts, but **critical behavioral transition achieved**:
    1. Output markup collapse was completely eliminated.
    2. Model learned to generate concise candidate values followed by immediate `<|im_end|>` termination tokens.
    3. Failure mode transitioned from catastrophic repetition to hallucinating familiar training set entities (e.g. generating `Ganymede` for Neptune's moon, or `3` for Gold's melting point).
- **Scientific Conclusion**:
  Loss masking successfully taught the 3.44M model output length control and valid termination syntax. However, true zero-shot pointer copying across attention heads requires larger parameter scale or explicit positional copy-gate mechanisms rather than purely generic causal transformer blocks.

### 3. Empirical Findings on Memory-Guided Multi-Step Tasks (Batch 168)
- **Component**: `CognitiveTaskEngine` (`chakrview/runtime/task_engine.py`) coordinating multi-step execution:
  `Read/Retrieve` $\to$ `Extract` $\to$ `Verify` $\to$ `Record in Memory`.
- **Result**:
  - Workflow executed cleanly to completion.
  - State persisted to disk and proved 100% idempotent upon engine restart.
  - Final extracted facts verified against supporting evidence spans before being committed to persistent conversation history.

### 4. Adversarial Benchmark Battery (Batch 169)
- **Total Tests**: 12 adversarial scenarios covering greetings, held-out facts, distractors, conflicts, Hindi, memory isolation, context truncation, empty input, and task crash resumption.
- **Pass Rate**: 12 / 12 (100.0%).
- **Component Attribution**:
  - Conversational turns attributed to `neural_core`.
  - Grounded factual slots attributed to `deterministic_extractor`.
  - Absent knowledge attributed to `fallback_policy`.
  - Multi-step orchestration attributed to `task_engine`.
