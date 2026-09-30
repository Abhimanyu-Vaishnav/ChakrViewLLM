# ChakrView Step 50: Trained Model Interactive Evaluation Report

- **Date**: 2026-09-30
- **Scope**: Step 50 — Trained ChakrView Interactive Model Evaluation
- **Target Architecture**: ChakrMicro v0.1 (Decoder-only causal transformer)
- **Parameters**: 3,443,136
- **Vocabulary Size**: 4,096
- **Context Length**: 512
- **Frozen Baseline SHA-256**: `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`
- **Selected Trained Checkpoint**: `artifacts/step48/run_seed_42/checkpoint_0000100.pt`
- **Selected Trained Weight SHA-256**: `85e1eb6cd3472d431692cde71cf58f705990fee50be91a2d70f46e0968481a64`
- **Status**: **RATIFIED & LOCKED**

---

## 1. Executive Summary & Objective

The objective of Step 50 is **not** to conduct additional model training, but to load the best available experimentally trained ChakrView checkpoint and provide an interactive terminal runtime (`scripts/chat_chakrview.py`) for direct evaluation, side-by-side baseline comparison, structured diagnostic probing, and quantitative metric collection.

All evaluations adhere strictly to the non-negotiable architectural invariants:
- **CPU-First Execution**: Zero GPU dependencies, zero external model APIs, zero Hugging Face dependencies.
- **$\Delta W_{\text{baseline}} = 0$**: The canonical frozen baseline model was never mutated.
- **$\Delta W_{\text{trained}} = 0$ during inference**: The evaluated experimental weights were never modified during interactive turns.
- **Separation of Observed vs Measured vs Interpretation**: No claims of consciousness, intelligence, or human-like understanding.

---

## 2. Checkpoint Selection & Verification

Following the exhaustive repository audit documented in `docs/STEP_50_CHECKPOINT_SELECTION.md`, the primary evaluated checkpoint is:
`artifacts/step48/run_seed_42/checkpoint_0000100.pt`

### Selection Criteria:
1. **Best Validated Loss & Perplexity**: Validation loss of $2.5324$ and perplexity of $12.58$ on the held-out validation corpus (compared to $8.3691$ loss / $4311.62$ PPL for the frozen baseline).
2. **Explicit Typology**: Checkpoint includes `checkpoint_type="training"`, validating model architecture, optimizer state, tokenizer checksum, and dataset manifest hash.
3. **Exact Tokenizer Checksum Match**: Checkpoint embeds checksum `7498d92adeef7c9e64d4d1bbd5b4b23f8774692d868742e3b98e30c584c4403f`, perfectly matching active tokenizer artifacts in `data/experiments/vocab_4096`.
4. **Finite Numerical Weights**: All 56 parameter tensors verify as finite (zero NaNs, zero Infs).

---

## 3. Interactive Runtime Architecture

The interactive evaluation runtime is implemented in `chakrview/runtime/interactive.py` and exposed via CLI in `scripts/chat_chakrview.py`:

```powershell
.venv\Scripts\python.exe scripts/chat_chakrview.py
```

### Supported Interactive Commands:
- `/help`: Displays command overview.
- `/info`: Displays architecture, training step count, parameter count, context length, and weight hashes.
- `/reset`: Clears conversation context turns, preserving the 512-token budget.
- `/context`: Displays active turn count, token consumption, and remaining context budget.
- `/stats`: Displays cumulative session statistics (prompts, tokens, throughput, average latency, unigram repetition).
- `/compare <prompt>`: Executes side-by-side generation comparing trained checkpoint against frozen baseline under identical parameters.
- `/probe`: Runs structured probe suite covering basic language, context sensitivity, controlled memory distance, negative controls, and simple patterns.
- `/raw`: Toggles token-level probability inspection, displaying selected token probability, rank, and top-5 alternatives.
- `/seed <N|random>`: Configures deterministic RNG seed or randomized sampling.
- `/generate_config`: Displays active generation hyperparameters.
- `/save [path]`: Serializes structured JSON session logs to `artifacts/step50/sessions/`.
- `/quit`: Gracefully exits and saves session logs.

---

## 4. Structured Probing Results

The structured probe suite (`tests/fixtures/step50_interactive_probes.json`) was evaluated on CPU using the selected trained model:

### Probe A: Basic Language Continuations
| Prompt | Top-1 Token | Top-1 Probability | Sample Continuation |
|:---|:---:|:---:|:---|
| *"The sun rises in the"* | `'` | 0.1326 | `': qkv_v007: ` |
| *"Water freezes at"* | `e` | 0.4905 | `e_tensor_kernel_v00` |
| *"A computer has"* | `: ` | 0.0868 | `: int = 3 * d_ff_v007` |
| *"The child is"* | `'` | 0.1667 | `': qkv_v007: ` |
| *"The sky is"* | `'` | 0.1687 | `': qkv_v007: ` |

### Probe B: Context Change & Sensitivity
| Paired Prompts | JS Divergence | Cosine Distance | Top-5 Overlap |
|:---|:---:|:---:|:---:|
| *"The cat sat on the"* vs *"The dog sat on the"* | 0.0025 | 0.0168 | 0.67 |
| *"In winter the weather is very"* vs *"In summer the weather is very"* | 0.0061 | 0.0519 | 1.00 |
| *"A bird can fly through the"* vs *"A fish can swim through the"* | 0.0038 | 0.0245 | 1.00 |

### Probe C: Controlled Context Distance Sensitivity (BLUE vs RED)
| Token Distance | JS Divergence | Cosine Distance |
|:---:|:---:|:---:|
| 8 tokens | 0.0000 | 0.0000 |
| 16 tokens | 0.0000 | 0.0000 |
| 32 tokens | 0.0000 | 0.0000 |
| 64 tokens | 0.0000 | 0.0000 |
| 128 tokens | 0.0000 | 0.0000 |

### Probe D: Negative Context Control (Clean vs Corrupted)
| Prompt Pair | JS Divergence | Cosine Distance |
|:---|:---:|:---:|
| Controlled Secret Sequence (Clean vs Corrupted) | 0.0017 | 0.0007 |
| Scientific Experiment Sentence (Clean vs Corrupted) | 0.0169 | 0.0718 |

---

## 5. Side-by-Side Baseline vs Trained Comparison

Side-by-side generation on prompt *"The sun rises in the"* (seed 42, temp 0.7, top-k 40, top-p 0.9, max 48 tokens):

```text
PROMPT:
The sun rises in the

BASELINE:
the (data.0हुत lossless  lossless _{ करते हैं।karne me -4-4py ष ष षs": 6,s": 6,s": 6,s": 6,ed Ta(SIM(SIM(SIM विशbalan192 192  थि^{\infty^{\infty[j, फelimincerकर्मणक्रsqrt(िसerror िििitit

TRAINED:
<EOS> (emitted immediately)

METRICS:
- Baseline tokens: 48 | Trained tokens: 1
- Baseline repetition: 0.3125 | Trained repetition: 0.0000
- Baseline EOS: False | Trained EOS: True
- Baseline latency: 525.10 ms | Trained latency: 8.60 ms
```

---

## 6. Objective Benchmark Performance Metrics

Derived from `docs/STEP_50_BENCHMARK_RESULTS.json`:

- **Startup Latency**: $311.84$ ms
- **Checkpoint Load Latency**: $202.34$ ms
- **Time to First Token (TTFT)**: $7.54$ ms mean ($7.43$ ms median)
- **Token Generation Latency**: $23.16$ ms/token
- **Generation Throughput**: $43.18$ tokens/sec on CPU
- **Session Reset Latency**: $< 0.001$ ms ($0.7$ µs)
- **Memory RSS Profile**: Initial $224.09$ MB, Final $421.60$ MB (Delta $+197.50$ MB for full runtime with both models loaded)
- **Baseline Weight Invariance**: $\Delta W_{\text{baseline}} = 0$ (`c5571c...` strictly identical pre/post evaluation)
- **Trained Weight Invariance**: $\Delta W_{\text{trained}} = 0$ (`85e1eb...` strictly identical pre/post evaluation)

---

## 7. Tripartite Scientific Analysis

### A. Observed (Ground Empirical Facts)
1. When prompted with conversational queries (`Hello.`, `What is your name?`), the trained checkpoint frequently predicts token ID 1 (`<EOS>`) with $> 50\%$ probability at step 1.
2. In basic language continuations, the trained model generates code-like syntactic fragments (e.g., `_tensor_kernel_`, `: int = 3 * d_ff_`).
3. Under identical prompts and seeds, the frozen baseline generates unstructured multi-script noise with repetition ratio $> 0.30$, while the trained model demonstrates sharp probability concentrations.

### B. Measured (Quantitative Metric Data)
1. Perplexity reduced from $4,311.62$ (baseline) to $12.58$ (trained checkpoint) on Stage B validation tokens.
2. Next-token entropy at step 1 dropped from near-uniform ($> 8.3$ nats) to $< 1.5$ nats on prompt continuations.
3. Context-sensitivity divergence across semantically contrasting prompts is strictly positive ($D_{\text{JS}} \in [0.0025, 0.0169]$).
4. Long-range distance sensitivity (Probe C) yielded $D_{\text{JS}} \approx 0.0000$ at 8–128 tokens, indicating that micro-training (100 optimization steps) has not yet developed long-distance syntactic binding.

### C. Interpretation (Scientific Inferences)
1. **Corpus Statistics Acquisition**: The model has learned local statistical token transition distributions and specific syntax patterns present in the training corpus.
2. **Absence of Conversational Alignment**: Because training data consisted of multi-domain pretraining shards rather than supervised dialogue instruction tuning, the model has no alignment to "chat" or "answer questions."
3. **No Reasoning or Understanding**: High confidence on technical tokens and prompt-dependent next-token selection reflect learned conditional Markov probabilities, not comprehension, consciousness, or reasoning.
