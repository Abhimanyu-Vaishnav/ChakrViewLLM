# Step 7 — Real-Corpus Baseline Freeze Record

**Document Version**: 1.0.0  
**Milestone**: Step 7 — Real-Corpus Baseline Pre-Training & Evaluation  
**Status**: FROZEN & AUDITED  
**Date**: 2026-09-28  

---

## 1. Baseline Status

> **OFFICIAL STATUS DECLARATION**:  
> Step 7 establishes the first real-corpus baseline for ChakrMicro v0.1. It is a research baseline, not a production model.

This document formally freezes the experimental configuration, inputs, empirical measurements, checkpoints, and verified contracts for Step 7. No further modifications to this baseline are permitted. All future pre-training experiments (starting with Step 8) will be compared directly against the metrics recorded herein.

---

## 2. Experiment Identity

* **Experiment Name**: `chakr_micro_stage_c_baseline`
* **Repository Lineage**:
  * Base Commit (Step 6.5 Stage C Ingestion): `3fa697c`
  * Pre-Training Run Execution Commit: `13412bc`
  * Visual Curve Artifacts Commit: `2a6a14a`
  * Audit & Freeze Commit: `HEAD`
* **Execution Date**: 2026-09-27
* **Deterministic Seed**: 42 (coordinated across Python `random`, NumPy, and PyTorch CPU)
* **Authoritative Configurations**:
  * JSON: [configs/chakr_micro_stage_c_baseline.json](file:///d:/Project/ChakrView/configs/chakr_micro_stage_c_baseline.json)
  * YAML: [configs/chakr_micro_stage_c_baseline.yaml](file:///d:/Project/ChakrView/configs/chakr_micro_stage_c_baseline.yaml)
  * Smoke Evaluation Suite: [configs/stage_c_smoke_prompts.json](file:///d:/Project/ChakrView/configs/stage_c_smoke_prompts.json)

---

## 3. Immutable Inputs

### Model (ChakrMicro v0.1 — Frozen)
* **Architecture**: Decoder-only causal Transformer
* **Layers ($N$)**: 6 stacked transformer blocks
* **Hidden Dimension ($d_{\text{model}}$)**: 192
* **Attention Heads ($n_{\text{heads}}$)**: 6 ($d_{\text{head}} = 32$)
* **FFN Hidden Dimension ($d_{\text{ff}}$)**: 512 (SwiGLU activation)
* **Context Length ($T$)**: 512 tokens
* **Normalization**: Pre-RMSNorm ($\epsilon = 10^{-5}$)
* **Positional Encoding**: Rotary Position Embeddings (RoPE, $\theta = 10000.0$)
* **Projections**: Bias-free linear layers throughout
* **Embedding Weight Tying**: Enabled ($W_{\text{out}} = E^T$)
* **Parameter Count**: Strictly **3,443,136 parameters** (56 named parameter tensors, 57 state dict entries)
* **Pretrained Weights**: None (trained from scratch with truncated normal initialization, $\sigma = 0.02$)

### Tokenizer (Byte-Level BPE — Frozen)
* **Vocabulary Size ($V$)**: 4096
* **Special Tokens**: BOS=0, EOS=1, PAD=2
* **Byte Primitives**: IDs 3–258 (256 raw byte octets)
* **Learned Merges**: IDs 259–4095 (3,837 merges)
* **Lossless Invariant**: $\text{Decode}(\text{Encode}(\text{text})) \equiv \text{text}$ across all Unicode octets
* **Artifact Directory**: `data/experiments/vocab_4096`
* **Artifact Checksum**: `7498d92adeef7c9e64d4d1bbd5b4b23f8774692d868742e3b98e30c584c4403f`

### Corpus (Stage C.1 Multi-Domain Dataset — Frozen)
* **Authoritative Manifest**: `data/manifests/stage_c_manifest.json` (SHA-256: `7fb98181aedffdba93ea07203acb0b21c79477bd62c9691485860ef3d209e3d0`)
* **Total Accepted Documents**: 5,534
* **Total Content Tokens**: 7,703,067
* **Total Sharded Tokens**: 7,708,601 (including 5,534 EOS boundary tokens)
* **Total Shards**: 32 binary shards (`uint16` little-endian, all 32 SHA-256 digests verified)
* **Token ID Bounds**: Strictly verified $1 \le \text{ID} \le 4094 < 4096$ across all tokens
* **Split Allocation**:
  * **Train**: 4,755 documents | 6,651,093 sharded tokens | 27 shards (`shard_00000.bin`–`shard_00026.bin`)
  * **Validation**: 378 documents | 465,954 sharded tokens | 2 shards (`shard_00000.bin`–`shard_00001.bin`)
  * **Test**: 401 documents | 591,554 sharded tokens | 3 shards (`shard_00000.bin`–`shard_00002.bin`)
* **Disjointness**: Document sets across train, validation, and test splits have exactly zero overlap ($\text{Overlap} = 0$).

---

## 4. Training Parameters

| Parameter | Frozen Value |
| :--- | :--- |
| **Total Steps** | 500 optimizer steps |
| **Tokens Processed** | 512,000 tokens (7.70% of 1 epoch of Stage C train split) |
| **Micro-Batch Size ($B$)** | 2 sequences |
| **Sequence Length ($T$)** | 512 tokens |
| **Tokens Per Step** | 1,024 tokens ($B \times T$) |
| **Optimizer** | AdamW ($\beta_1 = 0.9, \beta_2 = 0.95, \epsilon = 10^{-8}$) |
| **Weight Decay** | 0.01 (applied to 2D matrix projections; excluded from RMSNorm scales) |
| **Peak Learning Rate** | $5.0 \times 10^{-4}$ |
| **Minimum Learning Rate** | $5.0 \times 10^{-5}$ |
| **LR Schedule** | Cosine decay with 25 linear warmup steps |
| **Gradient Clipping** | 1.0 (L2 norm) |
| **Loss Function** | Causal Cross-Entropy with PAD token exclusion (`ignore_index = 2`) |
| **Device / Precision** | CPU / FP32 |

---

## 5. Measured Baseline Results

### Loss & Perplexity
* **Initial Step 1 Training Loss**: **8.3096** (PPL: 4,062.55) [measured]
* **Final Step 500 Training Loss**: **4.1756** (PPL: 65.08) [measured]
* **Training Loss Drop**: **4.1340** (**49.75% reduction**) [measured]
* **Initial Step 0 Validation Loss**: **8.3306** (PPL: 4,148.91) [measured]
* **Final Step 500 Periodic Validation Loss**: **4.1041** (PPL: 60.59) [measured]
* **Full Validation Split Loss (465,954 tokens)**: **6.3739** (PPL: **586.34**) [measured]
* **Full Test Split Loss (591,554 tokens)**: **6.3830** (PPL: **591.70**) [measured]
* **Generalization Alignment**: $|\text{Val} - \text{Test}| = \mathbf{0.0091}$ ($0.14\%$). Validation and test losses were closely aligned in this run, and no obvious validation/test divergence was observed.

### Diagnostic Domain Evaluation
Lightweight diagnostic evaluation on representative sample passages:
* **English**: Loss 4.5098 | PPL 90.90 (242 tokens)
* **Structured Data**: Loss 5.0519 | PPL 156.32 (71 tokens)
* **Mathematics**: Loss 5.6725 | PPL 290.77 (129 tokens)
* **Reasoning**: Loss 5.6838 | PPL 294.07 (157 tokens)
* **Hinglish**: Loss 7.2085 | PPL 1,350.92 (75 tokens)
* **Code**: Loss 8.5623 | PPL 5,230.63 (71 tokens)
* **Sanskrit**: Loss 9.0794 | PPL 8,772.48 (120 tokens)
* **Hindi**: Loss 9.1858 | PPL 9,757.61 (140 tokens)

### Hardware & Throughput Profile
* **Host CPU**: 13th Gen Intel(R) Core(TM) i7-13700H (14 physical cores, 20 logical threads)
* **Total Host RAM**: 31.64 GB
* **Peak Training Process Memory (RSS)**: **468.8 MB**
* **Peak Evaluation Process Memory (RSS)**: **520.4 MB**
* **Total Wall-Clock Training Time**: **589.1 seconds (~9.8 minutes)**
* **Average Training Throughput**: **869.1 tokens/second** (including online evals & checkpoints)
* **Full Validation Evaluation Duration**: 142.23 seconds (3,268.6 tokens/sec)
* **Full Test Evaluation Duration**: 184.45 seconds (3,197.8 tokens/sec)

---

## 6. Checkpoints & Resume Evidence

### Checkpoint Inventory
Located in `checkpoints/stage_c_baseline/`:
1. `checkpoint_0000100.pt`: 41,404,447 bytes | Step 100 | Verified
2. `checkpoint_0000200.pt`: 41,404,447 bytes | Step 200 | Verified
3. `checkpoint_0000300.pt`: 41,404,447 bytes | Step 300 | Verified
4. `checkpoint_0000400.pt`: 41,404,447 bytes | Step 400 | Verified
5. `checkpoint_0000500.pt`: 41,404,447 bytes | Step 500 | Verified (Final baseline checkpoint)

### Checkpoint Payload Schema
Every checkpoint contains:
* `model_state_dict`: 56 parameter tensors (57 state dict entries, 3,443,136 weights)
* `optimizer_state_dict`: AdamW moments for all parameters
* `scheduler_state_dict`: Cosine decay state
* `step`: Integer global step
* `rng_state`: CPU and NumPy RNG state tensors
* `config`: Complete training and model configuration dictionary

### Resume Verification
* Resumed from `checkpoint_0000200.pt` for 30 steps (steps 201–230).
* Maximum absolute loss discrepancy against uninterrupted reference run: **$0.00004959$** ($4.96 \times 10^{-5}$).
* Status: **PASS — Deterministic continuation within numerical tolerance** attributable to minor CPU FP32 floating-point accumulation ordering across process runs.

---

## 7. Qualitative Findings (Capability Smoke Tests)

Evaluated via greedy decoding ($\text{temperature}=0.0$, 14 standardized prompts in `configs/stage_c_smoke_prompts.json`):
* **Repetition**: Greedy decoding quickly enters repetitive loops (`the world the world...`), characteristic of early-stage models.
* **Partial syntax learning**: Punctuation, commas, and Markdown/JSON delimiters are partially emitted in appropriate relative positions.
* **Formatting preservation**: Indentation and structural formatting are preserved in structured data and code prompts.
* **Token-level continuation**: Subword transitions occur smoothly without crashing or producing out-of-vocabulary token IDs.
* **Multilingual degradation**: Devanagari prompts (Hindi, Sanskrit) frequently switch into Latin subwords and formatting fragments.
* **Code continuation weakness**: Fails variable binding and logical block completion.
* **Factual recall weakness**: Factual queries fail recall, defaulting to high-frequency text patterns.

*(Note: These are qualitative observations of raw next-token sampling from an early 500-step checkpoint, not measurements of intelligence, reasoning, comprehension, or lack thereof.)*

---

## 8. Known Limitations

1. **Early Budget Ceiling**: 500 steps exposed the model to 512,000 tokens ($\sim 7.7\%$ of 1 epoch of the training split). This is sufficient to validate learning convergence and infrastructure stability, but does not provide sufficient exposure for language fluency.
2. **Devanagari Subword Under-Exposure**: Hindi (loss 9.19) and Sanskrit (loss 9.08) exhibit significantly higher loss than English (loss 4.51). In early training, multi-byte UTF-8 sequences are under-represented relative to Latin characters.
3. **Repetitive Degeneration**: Under greedy decoding, generation collapses into repetitive n-grams.
4. **Code Syntax Incompleteness**: While Python indentation formatting is preserved, logical scoping and variable bindings remain incomplete.

---

## 9. Baseline Freeze Sign-Off

* **Neural Architecture Frozen**: ChakrMicro v0.1 (3,443,136 parameters)
* **Tokenizer Frozen**: Byte-Level BPE ($V=4096$)
* **Stage C Corpus Frozen**: 5,534 documents, 7,708,601 sharded tokens across 32 shards
* **Test Suite State**: 260 / 260 passing
* **Baseline Status**: **ACCEPTED & FROZEN**
