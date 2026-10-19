# Next Experiment Proposal: Batch 2901–2940
## Balanced SFT Mixture, Anti-Regression Pretraining Replay & Checkpoint Tracking

### 1. Motivation & Identified Deficiencies

The Batch 2861–2900 pilot established that:
1. Pure instruction fine-tuning (`100%` instruction template) on a micro-model triggers catastrophic suppression of unformatted raw completion (emitting `<EOS>` immediately on 71.4% of raw prompts).
2. A 192-example dataset for 100 steps is sufficient to learn shallow syntactic patterns (valid JSON `[0]`), but insufficient to learn multi-hop context extraction, refusal behavior, or semantic reasoning.
3. Checkpoints were saved only on fixed intervals (`step % 25 == 0`), missing the lowest validation loss state at step 40.

**This proposal is documented only. Training requires separate explicit user authorization.**

---

### 2. Proposed Experiment Specifications

#### A. Data Mixture Architecture (Total: 400 Examples)
- **70% Curated Instruction Tasks (280 examples)**:
  - 40 Simple instruction & definitions
  - 40 Context entity extraction
  - 40 Context-grounded QA
  - 40 Structured JSON with varied schema types (objects, dicts, key-value)
  - 40 Honest refusal & uncertainty probes
  - 40 Short summarizations
  - 40 Deterministic text transformations
- **30% Pretraining Replay Continuations (120 examples)**:
  - Unformatted paragraphs, Python functions, and continuous prose sourced from the original pretraining corpora.
  - Formatted without `Instruction:` prefixes and without early `<EOS>` triggers to preserve the model's base continuation capability.

#### B. Architectural & Training Safeguards
- **Dedicated Best-Checkpoint Saving**:
  - Update `SFTExperimentEngine` to track `min_val_loss` across validation intervals and save `sft_best_checkpoint.pt` automatically whenever validation loss reaches a new minimum.
- **Evaluation Decoupling**:
  - Split evaluation metrics into **Exact Entity Match** (for extractive tasks) and **Parse Validity** (for JSON), while recording qualitative human review flags for open-ended summaries.
- **Compute Budget**:
  - 200 optimizer steps on CPU (~30 seconds on standard CPU hardware).

---

### 3. Implementation Readiness

The data generation scripts, replay collator, and best-checkpoint tracking can be prepared cleanly within standard project modularity, awaiting authorization.
