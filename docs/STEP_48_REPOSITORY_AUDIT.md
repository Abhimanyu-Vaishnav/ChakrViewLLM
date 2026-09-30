# ChakrView Step 48: Pretraining Repository Audit & Baseline State

- **Date**: 2026-09-30
- **Scope**: Step 48 Controlled Pretraining & Learning Validation
- **Target Architecture**: ChakrMicro v0.1 ($3,443,136$ parameters, vocab 4096, context 512)
- **Frozen Baseline SHA-256**: `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`
- **Baseline Invariant**: $\Delta W_{\text{baseline}} = 0$ strictly enforced

---

## 1. Audit Scope & Overview

Step 48 shifts the repository focus from readiness validation ("the model can train without crashing or corrupting state") to empirical learning validation ("does the model demonstrably learn linguistic/statistical regularities from data?").

Before running training experiments or drawing scientific conclusions, this audit inspects the entire pretraining pipeline, data representation, loss formulation, optimization mechanics, and evaluation methodology across 18 audit axes.

---

## 2. Detailed Audit Across 18 Pretraining Axes

### 1. Training Objective
- **Implementation**: Autoregressive causal language modeling.
- **Formulation**: Minimizes the negative log-likelihood of token $x_t$ given context $x_{<t}$:
  $$\mathcal{L}(\theta) = -\frac{1}{T} \sum_{t=1}^T \log P_\theta(x_t \mid x_{<t})$$
- **Target Shift**: Standard causal shift: $x_{0 \dots T-1} \to x_{1 \dots T}$.

### 2. Exact Loss Implementation
- **Module**: `chakrview/training/loss.py` (`CausalLoss`).
- **Mechanism**: Reshapes logits $[B, T, V] \to [B \cdot T, V]$ and targets $[B, T] \to [B \cdot T]$.
- **Reduction**: Mean over unmasked tokens.
- **Masking**: `ignore_index = 2` (`<PAD>`), ensuring padding tokens contribute zero gradient.
- **Safety**: Checked via `TrainingSafetyChecker.check_loss()` against NaN, Inf, and explosions ($> 1000.0$).

### 3. Dataset Format
- **Format**: Flat sequential binary files (`shard_*.bin`) storing unsigned 16-bit integers (`np.uint16` / big-endian or native little-endian as specified in metadata).
- **Metadata**: Each shard directory contains `metadata.json` recording:
  - `split` (`train`, `val`, `test`)
  - `total_tokens`
  - `total_documents`
  - `vocab_size` (4,096)
  - `tokenizer_checksum` (`7498d92adeef7c9e64d4d1bbd5b4b23f8774692d868742e3b98e30c584c4403f`)
  - Per-shard file sizes, token counts, and SHA-256 digests.

### 4. Token Ordering & Chunking
- **Module**: `chakrview/training/dataset.py` (`StreamingTokenDataset`).
- **Chunking**: For a target `sequence_length` $T$, chunks of length $T+1$ are sliced sequentially.
  - `input_ids = chunk[:-1]`
  - `target_ids = chunk[1:]`
  - `attention_mask = ones(T)`
- **Remainder Handling**: Configurable `drop_remainder=True/False`. When padded, `pad_token_id=2` is used with attention mask zeroed on padded indices.

### 5. BOS/EOS Special Token Handling
- **Special Tokens Contract**: Step 2 ADR-T06:
  - `<BOS>` = 0
  - `<EOS>` = 1
  - `<PAD>` = 2
  - Strict absence of `<UNK>`, `<MASK>`, `<SEP>`, `<NL>`.
- **Shard Delineation**: Documents in raw corpus ingestion were encoded with `<BOS>` at index 0 and `<EOS>` at terminal index.

### 6. Context Length Constraints
- **Model Invariant**: $T_{\max} = 512$ tokens (RoPE frequency cache precomputed up to 512).
- **Batch Sequences**: Sequences fed to `ChakrMicro` must satisfy $T \le 512$. For micro-experiments, $T=64$ or $T=128$ allows rapid iteration and dense gradient estimation on CPU.

### 7. Available Dataset Splits
The repository contains three tiers of tokenized shards:
1. `data/tokenized/train/` (1,877 tokens) & `data/tokenized/val/` (479 tokens): Minimal smoke test splits.
2. `data/tokenized/stage_b/`:
   - `train/`: 2 shards, 405,244 tokens, 7,258 documents.
   - `validation/`: 1 shard, 45,027 tokens.
   - `test/`: 1 shard, 50,030 tokens.
   - Manifest: `data/manifests/stage_b_manifest.json` (SHA-256: `9c516cbc317f15867ab2b42c8893bca821de396a26a710fd1570524f4299ba7b`).
3. `data/tokenized/stage_c/`:
   - `train/`: 27 shards, 6,651,093 tokens.
   - `validation/`: 1 shard, 250,000 tokens.
   - `test/`: 1 shard, 250,000 tokens.
   - Manifest: `data/manifests/stage_c_manifest.json` (SHA-256: `7fb98181aedffdba93ea07203acb0b21c79477bd62c9691485860ef3d209e3d0`).

### 8. Split Isolation & Data Leakage Risk
- `stage_b` and `stage_c` splits were created from separate document sets during raw ingestion.
- **Phase 2 Requirement**: Programmatically verify that train and validation shards have zero exact shard-level collisions and measure sequence-level overlap to guarantee no data leakage.

### 9. Dataset Manifest Identity
- Manifests define document IDs, sources, hashes, and token allocations across splits.
- All manifests are cryptographically hashed and checked.

### 10. Tokenizer Identity & Verification
- **Artifact**: `data/experiments/vocab_4096/` (`vocab.json`, `merges.json`, `config.json`).
- **Merges SHA-256 Checksum**: `7498d92adeef7c9e64d4d1bbd5b4b23f8774692d868742e3b98e30c584c4403f`.
- **Integrity**: Matches `stage_b` and `stage_c` manifests exactly.

### 11. Optimizer Configuration
- **Algorithm**: Decoupled Weight Decay AdamW (`torch.optim.AdamW`).
- **Parameter Partitioning**:
  - Weight Decay ($0.1$ default): 2D+ matrices (embeddings, attention projections, MLP projections).
  - No Decay ($0.0$): 1D tensors (RMSNorm gains $\boldsymbol{\gamma}$).
- **Betas**: $(0.9, 0.95)$, $\epsilon = 10^{-8}$.

### 12. Learning Rate Configuration
- **Base LR**: $1 \times 10^{-3}$ (or $5 \times 10^{-4}$ for conservative micro-runs).
- **Minimum LR**: $1 \times 10^{-4}$ ($10\%$ of peak).

### 13. Scheduler Configuration
- **Schedule**: Linear warmup followed by Cosine Annealing decay down to `min_lr`.
- **Warmup**: $5\text{--}10\%$ of total planned optimization steps.

### 14. Gradient Handling & Safety
- **Gradient Clipping**: Norm clipping at threshold $1.0$ via `torch.nn.utils.clip_grad_norm_`.
- **Pre-Clipping Safety**: `TrainingSafetyChecker.check_gradients()` verifies zero NaN and zero Inf across all parameter tensors before clipping and stepping.

### 15. Checkpoint Architecture
- **Atomicity**: Writes to temporary `.tmp` file and performs atomic `os.replace`.
- **Typing**: Explicit `checkpoint_type="training"`, recording `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `rng_state`, `tokenizer_checksum`, `dataset_manifest_hash`, `model_config`, and `parameter_count: 3443136`.
- **Validation**: Rejects corrupt payloads or inference-only checkpoints.

### 16. RNG & Reproducibility Control
- **Seed Utility**: `chakrview/training/seed.py` controls Python `random`, `numpy.random`, and `torch.manual_seed`.
- **State Capture**: Checkpoints save full RNG state (`torch.get_rng_state()`).

### 17. Evaluation Methodology
- **Module**: `chakrview/training/evaluator.py`.
- **Metrics**: Cross-entropy validation loss ($\mathcal{L}_{\text{val}}$) and Perplexity:
  $$\text{PPL} = \exp(\mathcal{L}_{\text{val}})$$
- **Context**: Evaluated under `torch.no_grad()` with `model.eval()`.

### 18. Test Suite Leakage Audit
- Unit tests use synthetic fixtures or isolated temporary directories (`tmp_path`).
- Baseline model weights in tests use isolated seeded instances without mutating production checkpoints.

---

## 3. Conclusions & Readiness for Step 48

The audit confirms:
1. The training infrastructure is sound and safe.
2. The `stage_b` dataset ($405,244$ train tokens, $45,027$ validation tokens) provides an ideal, highly tractable dataset for CPU-first pretraining experiments without introducing multi-hour delays.
3. The next required step is Phase 2: rigorous dataset integrity and split leakage verification.
