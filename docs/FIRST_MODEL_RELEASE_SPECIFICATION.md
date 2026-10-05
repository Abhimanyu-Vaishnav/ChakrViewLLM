# ChakrView First Model Release Specification

**Model Name**: ChakrMicro-v0.1-Indigenous-Smoke  
**Release Version**: 0.1.0-alpha  
**Date**: October 5, 2026  
**License**: Project Internal / Apache 2.0 (as established in repository)  

---

## 1. Executive Identity and Purpose

ChakrMicro v0.1 is the **first indigenous, CPU-first, from-scratch small language model release** of the ChakrView cognitive architecture.

It is designed to serve as:
1. An empirical proof of training viability without reliance on external pretrained model weights, Hugging Face backbones, or GPU dependencies.
2. The neural generation backbone for ChakrView's governed cognitive runtime, operating in concert with the Persistent Project Brain (PPB) and symbolic cognitive layers.
3. An honest, low-resource research baseline evaluated with rigorous transparency regarding its parameter capacity and fluency limits.

---

## 2. Architecture & Parametric Invariants

The neural architecture strictly adheres to the ratified `ChakrMicroConfig`:

- **Model Class**: `chakrview.model.transformer.ChakrMicroTransformer`
- **Total Parameters**: `3,443,136` (exact invariant)
- **Layers (Blocks)**: `6`
- **Embedding Dimension ($d_{model}$)**: `256`
- **Feedforward Dimension ($d_{ff}$)**: `1024` (4x expansion)
- **Attention Heads**: `8`
- **Head Dimension**: `32` ($256 / 8$)
- **Max Context Length**: `512` tokens
- **Vocabulary Size**: `4,096` tokens
- **Normalization**: Pre-LayerNorm (`eps=1e-5`)
- **Positional Embeddings**: Learned 1D embeddings ($512 \times 256$)
- **Activation Function**: GELU (approximate)
- **Tie Word Embeddings**: True (input embedding matrix shared with language modeling head)

---

## 3. Tokenizer Specification

- **Tokenizer Family**: Indigenous Byte-Pair Encoding (BPE)
- **Vocabulary Artifacts**:
  - `data/experiments/vocab_4096/vocab.json`
  - `data/experiments/vocab_4096/merges.json`
  - `data/experiments/vocab_4096/config.json`
- **Vocabulary Size**: `4,096`
- **Special Tokens**:
  - `<pad>`: ID `0`
  - `<bos>`: ID `1`
  - `<eos>`: ID `2`
  - `<unk>`: ID `3`
  - `<mask>`: ID `4`
- **Text Coverage**: Multilingual support for English, Hindi, Hinglish, Code identifiers, and Mathematical notation.

---

## 4. Training Objective & Optimization

- **Objective**: Causal Next-Token Language Modeling (Auto-regressive Cross-Entropy Loss)
  $$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log P(x_i \mid x_{<i})$$
- **Loss Module**: `chakrview.training.loss.CausalLoss` (with optional ignore index = -100)
- **Optimizer**: Decoupled AdamW (`chakrview.training.optimizer.build_optimizer`)
  - Weight decay applied exclusively to 2D matrix weights (linear layers, projections, embeddings)
  - Zero weight decay applied to 1D biases and LayerNorm scales
- **Hyperparameter Profile**:
  - Base Learning Rate ($\eta$): $5 \times 10^{-4}$
  - Minimum Learning Rate ($\eta_{min}$): $5 \times 10^{-5}$
  - Betas: $(0.9, 0.95)$
  - Epsilon ($\epsilon$): $1 \times 10^{-8}$
  - Weight Decay ($\lambda$): $0.01$
  - Gradient Clipping ($\|\mathbf{g}\|_2$): $1.0$
- **Scheduler**: Cosine decay with linear warmup (warmup fraction: 0.05–0.10)
- **Hardware Profile**: CPU-First single socket execution, FP32 deterministic math, bounded thread pool.

---

## 5. Checkpoint Format & Persistence

Checkpoints are serialized atomically via `chakrview.training.checkpoint.CheckpointManager`:
- Temporary write path: `<checkpoint_dir>/step_<N>.pt.tmp`
- Atomic rename to: `<checkpoint_dir>/step_<N>.pt`
- Pointer metadata: `<checkpoint_dir>/latest_checkpoint.json`
- Checkpoint Payload:
  - `step`: Global training step (integer)
  - `epoch`: Global epoch (integer)
  - `model_state_dict`: Model weights
  - `optimizer_state_dict`: AdamW momentum & velocity states
  - `scheduler_state_dict`: Learning rate schedule progress
  - `rng_state`: PyTorch, NumPy, and Python deterministic RNG seeds
  - `config`: Complete training & model configuration dictionary
  - `loss`: Moving average loss at save point

---

## 6. Neural Governance Boundaries

Strict cognitive separation is mandated:
1. **Training Boundary**: Weight modification is restricted strictly to `chakrview.training.trainer.Trainer.train_step()`.
2. **Inference Boundary**: In all evaluation, validation, interactive chat, and cognitive orchestration loops, $\Delta W \equiv 0$ is strictly verified.
3. **Canonical Baseline Protection**: The canonical baseline artifact (hash `c5571c9c5cb77386...`) is immutable and write-protected. Trained model checkpoints must reside in isolated directories (e.g., `artifacts/releases/` or `scratch/training_runs/`) and must never overwrite the canonical baseline.

---

## 7. Release Artifact Structure

A formal ChakrMicro release bundle must contain:
```
release/chakrmicro_v0.1/
├── checkpoint/
│   ├── model.pt                     # Trained model weights & config
│   └── latest_checkpoint.json       # Step & metadata pointer
├── tokenizer/
│   ├── vocab.json                   # BPE vocabulary
│   ├── merges.json                  # BPE merge rules
│   └── config.json                  # Tokenizer configuration
├── config/
│   ├── model_config.json            # Model architecture parameters
│   └── training_config.json         # Training hyperparameters
├── evaluation/
│   ├── eval_summary.json            # Loss, perplexity, token sanity metrics
│   └── delta_w_verification.json    # Proof of zero weight leakage
├── checksums.sha256                 # SHA-256 hashes of all artifacts
├── MODEL_CARD.md                    # Detailed model documentation & disclosures
└── README.md                        # Quickstart instructions for CPU inference
```

---

## 8. Known Limitations & Explicit Non-Goals

### Known Limitations
1. **Capacity Limit**: At 3.44M parameters with a 4,096 vocabulary, this model is a micro-scale language model. It cannot exhibit general world knowledge, complex multi-turn conversational mastery, or extensive commonsense reasoning.
2. **Context Window**: 512 tokens maximum sequence length.
3. **Fluency Boundary**: Open-domain fluent conversation is unproven and not guaranteed. The model is an indigenous research baseline.

### Explicit Non-Goals
1. **No Pretend AGI**: Will not claim human-level reasoning or GPT-4 parity.
2. **No External Backbones**: No fine-tuning on top of LLaMA, Mistral, BERT, or external weights.
3. **No GPU Requirement**: Must not fail on standard consumer CPU workstations without accelerator hardware.
