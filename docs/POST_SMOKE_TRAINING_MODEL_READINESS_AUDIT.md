# Post-Smoke-Training Model Readiness Audit

**Document Identity**: `POST_SMOKE_TRAINING_MODEL_READINESS_AUDIT.md`  
**Date**: October 5, 2026  
**Status**: EMPIRICALLY RATIFIED AUDIT  
**Baseline Hard Invariant**:
- Parameter Count: `3,443,136`
- Canonical Weight SHA-256: `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`
- Core Principle: Sovereign, indigenous, from-scratch, CPU-first neural intelligence. Cognitive self-evolution $\neq$ neural weight mutation. Canonical baseline $\Delta W \equiv 0$.

---

## 1. Deep Subsystem Audit Findings

### 1.1 Tokenizer
- **Implementation**: Indigenous BPE Tokenizer (`chakrview/tokenizer/tokenizer.py`, `chakrview/tokenizer/bpe.py`).
- **Artifacts**: Curated vocabulary in `data/experiments/vocab_4096/` (`vocab.json`, `merges.json`, `config.json`).
- **Vocabulary Size**: Exact `4,096` tokens.
- **Special Tokens**: `<bos>` (0), `<eos>` (1), `<pad>` (2), `<unk>` (3), `<mask>` (4).
- **Encode/Decode & Round-Trip**: Lossless UTF-8 byte fallback; validated by extensive property tests (`test_tokenizer_utf8_lossless.py`).
- **Compatibility**: Full handoff compatibility with `ChakrMicro` embedding layer ($4096 \times 256$) and causal head.

### 1.2 Corpus & Tokenized Shards
- **Available Datasets**:
  - `data/tokenized/train`: 1 shard, 1,877 tokens, 80 documents (micro-smoke).
  - `data/tokenized/val`: 1 shard, 479 tokens, 20 documents.
  - `data/tokenized/stage_b/train`: 2 shards, 405,244 tokens, 7,258 documents, 1,576 sequences (at length 256).
  - `data/tokenized/stage_b/validation`: 1 shard, 48,527 tokens, 891 documents, 188 sequences (at length 256).
  - `data/tokenized/stage_b/test`: 1 shard, 47,882 tokens, 853 documents.
  - `data/tokenized/stage_c/train`: 27 shards, 6,651,093 tokens, 4,755 documents, 25,859 sequences (at length 256).
  - `data/tokenized/stage_c/validation`: 2 shards, 465,954 tokens, 378 documents.
  - `data/tokenized/stage_c/test`: 3 shards, 591,554 tokens, 401 documents.
- **Separation & Leakage**: Shards are pre-partitioned strictly by document boundaries across train, validation, and test splits with explicit cryptographic checksums in manifests (`stage_b_manifest.json`, `stage_c_manifest.json`).
- **Quality**: Curated multi-domain mix (Python/TypeScript/C/Rust syntax, algorithms, natural language explanations, mathematical expressions).

### 1.3 Training Infrastructure
- **Engine**: `chakrview/training/trainer.py`, `dataset.py`, `optimizer.py`, `checkpoint.py`.
- **Target Shifting**: Explicitly constructed by `StreamingTokenDataset`:
  $$\text{sequence } [x_0, x_1, \dots, x_T] \implies \text{input\_ids } [x_0, \dots, x_{T-1}],\quad \text{target\_ids } [x_1, \dots, x_T]$$
- **Loss Computation**: `CausalLoss` cross-entropy on $[B, T, V]$ vs $[B, T]$, correctly ignoring padding index 2.
- **Gradient Handling & Accumulation**: Linear scaling ($L / \text{accum}$), L2 norm clipping ($\le 1.0$), gradient norm tracking.
- **Throughput & CPU Latency**: Benchmarked at **~62.6 ms per step** (16.0 steps/sec) on modern CPU at batch size 2, sequence length 256.

### 1.4 Evaluation Infrastructure
- **Metrics**: `evaluate()` in `evaluator.py` and `ReleaseEvaluator` in `release_evaluator.py`.
- **Loss & Perplexity**: Validation loss computed over held-out batches with mathematical perplexity $e^{\min(20, \text{loss})}$.
- **Assessment**: The current evaluation is scientifically grounded for computing empirical cross-entropy and token perplexity. However, it previously only ran for tiny numbers of batches in smoke tests. It must be run against a structured held-out evaluation split to prove true generalization beyond training sample memorization.

### 1.5 Inference Infrastructure
- **Trained Checkpoint Loading**: Implemented cleanly via `NeuralInferenceContract.from_checkpoint(...)`.
- **Inference Invariant**: $\Delta W \equiv 0$ strictly verified pre- and post-generation; tamper detection fails closed.
- **EOS Handling**: Greedy decoding and nucleus sampling with clean sequence stopping and max token budgets.

### 1.6 Governance & Boundary Verification
- **Separation**: The canonical baseline ($3.44\text{M}$ params, hash `c5571c...`) is immutable.
- **Cognitive Isolation**: Persistent Project Brain (PPB), investigation loops, episodic memory, strategy registries, and autonomous replanning operate exclusively on cognitive metadata and are architecturally barred from altering model weights.

---

## 2. Core Audit Questions (A through L)

### A. What is genuinely complete?
1. The indigenous Transformer model architecture (`ChakrMicro`, 6 layers, 256 hidden dimension, 8 heads, 3,443,136 parameters).
2. The BPE Tokenizer (v4096 vocabulary, round-trip lossless encoding/decoding, special tokens).
3. The dataset streaming engine (`StreamingTokenDataset` with dynamic binary shard memory mapping).
4. The AdamW optimizer builder with decoupled weight decay (2D matrices vs 1D biases/norms).
5. Cosine learning rate scheduling with linear warmup.
6. Causal cross-entropy loss with padding token masking.
7. Atomic checkpoint save, prune, pointer, and full-state resumption.
8. Deterministic RNG control (`seed=42`) producing bit-exact runs.
9. Post-training inference contract enforcing $\Delta W \equiv 0$.
10. Release packager bundling checkpoint, tokenizer, config, SHA-256 checksums, and model card.

### B. What is only infrastructure?
- The 14-step smoke training ran on only 1,877 tokens. It proved that the code executes, computes gradients, updates weights, and saves checkpoints without crashing. **It did not train a functional language model.**
- The model at 14 steps has only reduced cross-entropy loss from 8.35 to 7.94 on a handful of sequences. It cannot generate coherent sentences, predict syntax reliably, or perform open-domain reasoning.

### C. What has actually been learned by the neural model?
- **Empirical Reality**: In the 14-step smoke experiment, the model learned negligible semantic structure. It slightly adjusted weight magnitudes toward the marginal unigram/bigram token distribution of the tiny 1,877-token train shard.
- It has **NOT** learned English grammar, Python syntax, logic, or conversational conventions.

### D. What evidence currently proves learning?
1. Monotonic loss decrease on training batches ($8.3582 \to 7.9482$).
2. Gradient norms remaining bounded and non-zero ($0.94 \to 0.71$).
3. Bit-exact reproducibility across independent training runs with identical seeds.
4. Divergence of weight hash from initial random seed state ($H_{\text{trained}} \neq H_{\text{init}}$) without producing NaNs or Infinities.

### E. What evidence is still missing?
1. **Generalization to held-out data**: Decreasing validation loss on unseen shards over hundreds of training steps.
2. **Perplexity reduction**: Significant drop in validation perplexity (from random initialization $e^{8.31} \approx 4064$ down toward $< 100$).
3. **Syntactic next-token prediction sanity**: Ability to predict obvious syntax completions (e.g. `def `, `return `, `import `, standard English articles).
4. **Overfitting vs Generalization curves**: Evidence that training loss decreases while validation loss also decreases (rather than diverging).

### F. What is the minimum dataset/training/evaluation pipeline required for the first real trained model?
- **Dataset**: `Stage B` curated corpus (`data/tokenized/stage_b/train/` with 405,244 tokens, and `data/tokenized/stage_b/validation/` with 48,527 tokens).
- **Sequence Length**: 256 tokens (optimal balance of context capacity and CPU throughput).
- **Batch Size**: 2 micro-batch, 2 gradient accumulation steps (effective batch size = 4 sequences = 1,024 tokens per optimizer step).
- **Steps**: 500 – 1,000 steps (~512,000 – 1,024,000 tokens processed = 1.25 – 2.5 epochs of Stage B).
- **Evaluation**: Validation evaluated every 50 steps on 10 held-out validation batches (~2,560 tokens).
- **Wall-clock time**: At ~65 ms per step, 500 steps will take approximately **32–40 seconds** on CPU.

### G. What should remain unchanged?
- `ChakrMicro` architecture ($3,443,136$ parameters, 6 layers, 256 embedding dimension).
- Vocabulary size ($4,096$) and special token IDs.
- Causal loss implementation (`CausalLoss`).
- Canonical baseline artifact (`c5571c...`) and strict $\Delta W = 0$ inference enforcement.
- PPB and cognitive orchestration modules.

### H. What must be improved before long training?
1. **Validation Metric Step Storing**: Ensure `latest_val_metrics` are only attached to step records when evaluation actually occurs, or explicitly preserved as `last_val_loss` so step-level logs distinguish between train-only steps and evaluation intervals.
2. **Evaluator Batch Exhaustion**: Ensure `StreamingTokenDataset` when used in validation does not loop infinitely or cause hanging when `loop=False` is set.
3. **Syntactic Probe Evaluation**: Add an objective syntactic completion probe to `ReleaseEvaluator` that checks top-1 and top-5 accuracy on standard canonical tokens (e.g. language keywords and common words).

### I. What metrics should determine whether training is successful?
1. **Train Loss Descent**: $\mathcal{L}_{\text{train}}$ drops from $\sim 8.3$ to $\le 5.5$.
2. **Validation Loss Descent**: $\mathcal{L}_{\text{val}}$ drops monotonically from $\sim 8.3$ to $\le 6.0$ on unseen Stage B shards.
3. **Validation Perplexity**: Drops from $\approx 4,000$ to $\le 400$.
4. **Held-Out Generalization Gap**: $|\mathcal{L}_{\text{val}} - \mathcal{L}_{\text{train}}| < 1.0$ (no catastrophic overfitting).
5. **Numerical Stability**: 0 NaNs, 0 Infs, gradient norms $\le 1.0$.
6. **Inference Invariant**: $\Delta W \equiv 0$ on the resulting checkpoint.

### J. What would constitute a scientifically honest "First Usable Model" milestone?
A model release where:
1. It is trained on at least 400k tokens from scratch without pretrained weights.
2. Held-out validation perplexity is measurably reduced compared to random initialization.
3. It can generate syntactically coherent micro-structures (e.g., Python function headers, structured word associations).
4. Its limitations are documented in its Model Card with zero hyperbole:
   - "Capable of micro-scale syntactic pattern generation."
   - "Not a general conversational assistant."
   - "Requires cognitive PPB grounding for factual accuracy."

### K. What capabilities must explicitly remain UNPROVEN?
- Open-domain dialogue and freeform conversational fluency.
- Factual question answering without PPB retrieval.
- Complex multi-step reasoning without symbolic cognitive deliberator.
- Zero-shot instruction following on unseen tasks.

### L. What is the smallest safe experiment to validate the complete pipeline before long training?
- **Experiment Spec**: `Stage-B-50-Step-Generalization-Check`
- **Train Shard**: `data/tokenized/stage_b/train` (shards 0 & 1).
- **Val Shard**: `data/tokenized/stage_b/validation` (shard 0).
- **Sequence Length**: 128 tokens.
- **Steps**: 50 optimization steps.
- **Eval Interval**: Every 25 steps (evaluates step 0, 25, 50 on held-out validation data).
- **Expected Runtime**: < 5 seconds on CPU.
- **Pass Criteria**:
  - Step 0 $\mathcal{L}_{\text{val}} \approx 8.3 \pm 0.2$.
  - Step 50 $\mathcal{L}_{\text{val}} < 7.8$.
  - Both train loss and validation loss decrease.
  - Final checkpoint saved atomically with non-baseline weight hash.
  - Canonical baseline hash unchanged.

---

## 3. Cognitive vs Neural Architecture Boundary

```
+-------------------------------------------------------------------------+
|                       COGNITIVE ARCHITECTURE                            |
|                                                                         |
|  - Persistent Project Brain (PPB): Project memory, symbols, dependencies|
|  - Investigation Loop: Grounded source code analysis, AST inspection    |
|  - Deliberation & Critical Thinking: Multi-hypothesis evaluation        |
|  - Strategy Registry: Success/failure pattern adaptation                |
|  - Adaptive Budgeting: Token & time constraint allocation               |
|                                                                         |
|  NOTE: This symbolic layer provides grounding, facts, and structure.    |
|        It must NEVER be misrepresented as "neural model fluency".       |
+------------------------------------+------------------------------------+
                                     |
                                     | Structured Prompt Handoff
                                     v
+-------------------------------------------------------------------------+
|                     SOVEREIGN NEURAL BACKBONE                           |
|                                                                         |
|  - Model: ChakrMicro (3,443,136 parameters, 6 layers, 256 d_model)      |
|  - Tokenizer: BPE 4096 vocabulary                                       |
|  - Objective: Causal Next-Token Language Modeling                       |
|  - Current Status: Early Research Baseline                              |
|  - Inference Invariant: Delta W = 0 strictly verified                   |
|  - Weight Updates: EXCLUSIVELY in offline Trainer subsystem             |
+-------------------------------------------------------------------------+
```

The cognitive runtime must **never** be used as a camouflage for neural model limitations. The neural model must be measured, trained, and evaluated on its own merits as an autoregressive next-token predictor.

---

## 4. Architectural Component Classification

| Component | Classification | Rationale |
|---|---|---|
| `ChakrMicroTransformer` | **DO NOT TOUCH** | Ratified neural architecture. Invariant parameter count: 3,443,136. |
| Canonical Baseline Weights | **DO NOT TOUCH** | Invariant hash: `c5571c...`. Must remain bit-exact. |
| BPE Tokenizer & Vocab 4096 | **KEEP** | Complete, tested, lossless round-trip UTF-8. |
| `StreamingTokenDataset` | **KEEP** | Fully functional streaming reader for binary uint16 shards. |
| `CausalLoss` | **KEEP** | Standard cross-entropy with padding exclusion. |
| `Trainer` | **KEEP & REFINE** | Solid training engine. Minor enhancement: record validation metrics cleanly without carrying stale values across steps. |
| `ReleaseEvaluator` | **ADD PROBES** | Add token prediction accuracy probes on common language syntax to measure actual learning. |
| `PersistentProjectBrain` | **KEEP** | Core cognitive memory. Zero weight mutation guaranteed. |
| External Pretrained Backbones | **FORBIDDEN** | Strictly indigenous from-scratch policy. |

---

## 5. Next Empirical Experiment Plan

Before committing hours to multi-epoch Stage C pre-training, execute the **Stage-B-50-Step Generalization Experiment** in an isolated test suite:
- Verify that real held-out validation loss drops monotonically on Stage B data.
- Confirm that optimizer momentum, learning rate warmup/decay, and gradient clipping operate flawlessly on realistic sequences ($T=128$, $B=2$).
- Verify that post-training inference on the resulting checkpoint demonstrates consistent token prediction shifts without NaN/Inf pathologies.
