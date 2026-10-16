# Recommended Next Core Experiment

## Batches 1341+: Controlled Held-Out Perplexity and Next-Token Prediction Benchmark for ChakrMicro

### 1. Executive Summary and Strategic Rationale

Over the preceding 300 batches (Batches 1021–1340), ChakrView achieved a robust, deterministic, local knowledge assistant MVP with 98.6% benchmark accuracy, sub-1.1 ms CPU latency, 100% citation fidelity, and safe attribute-aligned false-premise refutation.

However, as documented in `NEURAL_CORE_CAPABILITY_INVENTORY.md`, almost all of this operational utility is provided by deterministic retrieval (BM25), string parsing, and session compaction. The neural core (`ChakrMicro`, 10.8M parameters) has remained largely unmeasured on true out-of-distribution language modeling.

Expanding deterministic regex heuristics further (e.g. attempting to parse all physical units, dates, or complex natural language negations via regex) is an engineering dead end. **The single most valuable next experiment is to establish a rigorous, reproducible, held-out language modeling evaluation protocol for ChakrMicro.**

---

### 2. Experiment Specification: Controlled Next-Token Prediction Benchmark

#### 2.1. Core Hypothesis
The canonical `ChakrMicro` checkpoint (step 10,000, 10.8M parameters) exhibits measurable token prediction capability (cross-entropy loss < 4.5, Perplexity < 90) on unseen in-domain technical documentation, demonstrating that it has learned general token transition priors rather than simply memorizing exact training sequences.

#### 2.2. Baseline Configuration
- **Model Checkpoint:** `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` (SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`).
- **Tokenizer:** Canonical vocabulary 4096 tokens (`release/chakrview-v0.5.0/data/experiments/vocab_4096`).
- **Architecture:** 6-layer Transformer decoder, 384 hidden dim, 6 query heads, 2 KV heads, 1536 intermediate FFN, max sequence length 512.
- **Reference Baseline:** Uniform random token baseline ($-\ln(1/4096) \approx 8.317$ cross-entropy loss, Perplexity = 4096) and unigram frequency baseline.

#### 2.3. Dataset & Split Design
- **Corpus Source:** Standardized technical documentation chunks from `experiments/batch1141_1180/data/corpus/` (30 documents, ~12,000 tokens total).
- **Strict Split Protocol:**
  - 80% Held-Out Validation Split: 24 documents never seen during any prompt tuning or slot SFT.
  - 20% In-Distribution Split: 6 documents used for reference calibration.
- **Data Integrity:** Dataset manifest with SHA-256 hashes must be generated before evaluation.

#### 2.4. Compute and Hardware Budget
- **Device:** Local CPU (Intel/AMD x86_64, AVX2).
- **Execution Budget:** < 30 seconds for full held-out evaluation across 10,000 tokens. Zero GPU required.
- **Training Budget:** 0 GPU hours. This experiment is an **evaluation-first audit** of existing weights before committing compute to re-training.

#### 2.5. Primary Evaluation Metrics
1. **Per-Token Cross-Entropy Loss:** Evaluated on sliding window with stride 256.
2. **Held-Out Perplexity ($PPL = \exp(Loss)$):** Target < 90.0 for basic language modeling coherence.
3. **Top-1 and Top-5 Token Prediction Accuracy:** Percentage of next tokens where the true token is in the top-k predicted logits.
4. **Repetition Penalty Sensitivity:** Measurement of token n-gram repetition degradation under greedy decoding.

#### 2.6. Failure and Halt Criteria
- **Catastrophic Failure:** Perplexity > 250 (indicates model weights have collapsed or forgotten language distribution outside narrow training prompts).
- **Severe Degeneration:** Model produces identical repetitive n-grams (>70% duplicate 3-grams) within the first 30 tokens of open-ended autoregressive decoding.
- **Protocol Action on Failure:** If Perplexity > 250, halt all downstream agentic planning tasks; prioritize fundamental masked/causal language model pretraining with a clean technical web/code corpus before any further tool-calling experiments.

---

### 3. Readiness and Preconditions

| Precondition | Status | Notes |
|---|---|---|
| Model architecture code in tree | **READY** | `chakrview/core/` and checkpoint loader present. |
| Canonical checkpoint verified | **READY** | Hash `96357705a6...` verified identical. |
| Tokenizer artifacts accessible | **READY** | Vocab 4096 loaded and verified. |
| Test environment ready | **READY** | Python 3.14.7 virtual environment with PyTorch 2.6+ CPU. |
| Training deferred | **ENFORCED** | No retraining will start until baseline evaluation script is committed and verified. |
