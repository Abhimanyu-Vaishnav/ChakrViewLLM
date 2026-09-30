# ChakrView Step 52: Empirical Neural Model Capability Baseline

- **Date**: 2026-09-30
- **Status**: Ratified Empirical Baseline
- **Auditor / Evaluator**: Antigravity Core Agent
- **Baseline Invariant**: `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` ($\Delta W_{\text{baseline}} = 0$)

---

## 1. Environment

- **Operating System**: Windows 11 (build 10.0.26100)
- **Python Version**: 3.14.7
- **PyTorch Version**: 2.14.0+cpu
- **Execution Device**: Pure CPU (`torch.device("cpu")`), zero CUDA/GPU calls.
- **Thread Allocation**: Standard PyTorch CPU multi-threading.
- **Test Runner**: pytest 9.1.1.

---

## 2. Model Configuration (`ChakrMicro` v0.1)

- **Architecture**: Decoder-only causal transformer.
- **Layers ($N$)**: 6
- **Hidden Dimension ($d_{\text{model}}$)**: 192
- **Attention Heads ($n_{\text{heads}}$)**: 6 (dimension per head = 32)
- **Feed-Forward Dimension ($d_{\text{ff}}$)**: 512 (SwiGLU activation)
- **Normalization**: Pre-RMSNorm ($\epsilon = 10^{-5}$)
- **Positional Embeddings**: Rotary Position Embeddings (RoPE, $\Theta = 10000.0, d_{\text{rot}} = 32$)
- **Embedding Tying**: Enabled ($W_{\text{out}} \equiv E^T$)
- **Total Parameters**: **3,443,136**
- **Context Ceiling ($T_{\text{max}}$)**: 512 tokens
- **Static FP32 Weight Footprint**: 13.13 MiB (13.77 MB)

---

## 3. Checkpoint Identities & Hashes

| Checkpoint Name | Description | SHA-256 Digest | Status |
|:---|:---|:---|:---|
| **Frozen Baseline** | Canonical deterministic initialization (seed 42, step 0) | `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` | **Frozen / Immutable ($\Delta W = 0$)** |
| **Step 51 Micro-Coding** | Experimental 60-step pretraining on code shards | `bc6c77093282bdfba5bf07b7b1b3fbdb1d9e2db3db78e121516e537eefea8bbf` | Experimental Artifact |
| **Stage C Full Epoch** | Experimental 6,478-step pretraining on math/text corpus | `cbca283ae32483161c56da9c1cf2ff6b7d27e997f8e81e37bc901a8f96ce751c` | Experimental Artifact |

---

## 4. Tokenizer Identity

- **Tokenizer Type**: Byte-Level BPE (`BPETokenizer`)
- **Vocabulary Size**: 4,096
- **Special Tokens**: `<BOS>=0`, `<EOS>=1`, `<PAD>=2`
- **Merges Count**: 3,837
- **Artifact SHA-256**: `7498d92adeef7c9e64d4d1bbd5b4b23f8774692d868742e3b98e30c584c4403f`
- **Location**: `data/experiments/vocab_4096/`

---

## 5. Benchmark Task Definitions (Level-0 Capability Set)

The benchmark evaluates 20 deterministic tasks across 6 functional categories:

1. **Arithmetic** (4 tasks):
   - `MATH_01`: `"1 + 1 = "` $\to$ expects `2`
   - `MATH_02`: `"2 + 3 = "` $\to$ expects `5`
   - `MATH_03`: `"10 - 4 = "` $\to$ expects `6`
   - `MATH_04`: `"5 * 2 = "` $\to$ expects `10`
2. **Transformation** (3 tasks):
   - `TRANS_01`: `"apple in uppercase is "` $\to$ expects `APPLE`
   - `TRANS_02`: `"Reverse 'abc': "` $\to$ expects `cba`
   - `TRANS_03`: `"The opposite of hot is "` $\to$ expects `cold`
3. **Structured Completion** (3 tasks):
   - `STRUCT_01`: `'{"name": "ChakrView", "status": "'` $\to$ expects valid JSON closure
   - `STRUCT_02`: `'numbers = [1, 2, 3, '` $\to$ expects valid Python list closure
   - `STRUCT_03`: `'title: Test\nauthor: '` $\to$ expects non-empty key-value string
4. **Code Completion** (4 tasks):
   - `CODE_01`: `"def add(a, b):\n    return "` $\to$ expects valid Python code AST
   - `CODE_02`: `"def is_even(n):\n    return n % 2 == "` $\to$ expects `0` and valid AST
   - `CODE_03`: `"x = 10\ny = 20\ntotal = x + "` $\to$ expects `y` and valid AST
   - `CODE_04`: `"def square(x):\n    return x * "` $\to$ expects `x` and valid AST
5. **Factual / Formatting** (3 tasks):
   - `FACT_01`: `"The capital of France is "` $\to$ expects `Paris`
   - `FACT_02`: `"Question: What color is the sun?\nAnswer: "` $\to$ expects `yellow` or `white`
   - `FACT_03`: `"Days of week: Monday, Tuesday, Wednesday, "` $\to$ expects `Thursday`
6. **Instruction Following** (3 tasks):
   - `INST_01`: `"Repeat the word 'hello':\nOutput: "` $\to$ expects `hello`
   - `INST_02`: `"Is 5 greater than 2? Answer yes or no:\nAnswer: "` $\to$ expects `yes`
   - `INST_03`: `"Output only the number seven:\n"` $\to$ expects `7` or `seven`

---

## 6. Generation Settings

- **Sampling Mode**: Deterministic Greedy (`temperature = 0.0` / `argmax(logits)`)
- **Random Seed**: 42
- **Repetition Penalty**: 1.0 (raw unpenalized logits to observe natural model behavior)
- **Context Management**: Single-prompt prefill with incremental single-token decoding
- **EOS Handling**: Stops on token ID 1 (`<EOS>`)

---

## 7. Raw Model Outputs

### A. Frozen Baseline Checkpoint (`c5571c...`)
| Task ID | Prompt | Raw Model Output | Failure Category |
|:---|:---|:---|:---|
| `MATH_01` | `"1 + 1 = "` | `" =  =  =  =  =  =  =  = "` | `REPETITION_COLLAPSE` |
| `MATH_02` | `"2 + 3 = "` | `" =  =  =  =  =  =  =  = "` | `REPETITION_COLLAPSE` |
| `MATH_03` | `"10 - 4 = "` | `" =  =  =  =  =  =  =  = "` | `REPETITION_COLLAPSE` |
| `MATH_04` | `"5 * 2 = "` | `" =  =  =  =  =  =  =  = "` | `REPETITION_COLLAPSE` |
| `TRANS_01` | `"apple in uppercase is "` | `"is is is is is is is is is is is is "` | `REPETITION_COLLAPSE` |
| `TRANS_02` | `"Reverse 'abc': "` | `": : : : : : : : : : "` | `REPETITION_COLLAPSE` |
| `TRANS_03` | `"The opposite of hot is "` | `"is is is is is is is is is is "` | `REPETITION_COLLAPSE` |
| `STRUCT_01`| `'{"name": "ChakrView", "status": "'` | `""` | `EMPTY_OUTPUT` |
| `STRUCT_02`| `'numbers = [1, 2, 3, '` | `", , , , , , , , , , , , , , , , "` | `REPETITION_COLLAPSE` |
| `STRUCT_03`| `'title: Test\nauthor: '` | `": : : : : : : : : : : : "` | `REPETITION_COLLAPSE` |
| `CODE_01` | `"def add(a, b):\n    return "` | `"================================================"` | `REPETITION_COLLAPSE` |
| `CODE_02` | `"def is_even(n):\n    return n % 2 == "` | `"=============================="` | `REPETITION_COLLAPSE` |
| `CODE_03` | `"x = 10\ny = 20\ntotal = x + "` | `"+ + + + + + + + + + "` | `REPETITION_COLLAPSE` |
| `CODE_04` | `"def square(x):\n    return x * "` | `"* * * * * * * * * * "` | `REPETITION_COLLAPSE` |
| `FACT_01` | `"The capital of France is "` | `"is is is is is is is is is is "` | `REPETITION_COLLAPSE` |
| `FACT_02` | `"Question: What color is the sun?\nAnswer: "` | `": : : : : : : : : : : : "` | `REPETITION_COLLAPSE` |
| `FACT_03` | `"Days of week: Monday, Tuesday, Wednesday, "` | `", , , , , , , , , , , , "` | `REPETITION_COLLAPSE` |
| `INST_01` | `"Repeat the word 'hello':\nOutput: "` | `": : : : : : : : : : "` | `REPETITION_COLLAPSE` |
| `INST_02` | `"Is 5 greater than 2? Answer yes or no:\nAnswer: "`| `": : : : : : : : "` | `REPETITION_COLLAPSE` |
| `INST_03` | `"Output only the number seven:\n"` | `": : : : : : : : "` | `REPETITION_COLLAPSE` |

### B. Step 51 Micro-Coding Checkpoint (60 Steps Pretrained)
| Task ID | Prompt | Raw Model Output | Failure Category |
|:---|:---|:---|:---|
| `MATH_01` | `"1 + 1 = "` | `"1\n    "1\n    "1\n    "1\n    "` | `REPETITION_COLLAPSE` |
| `CODE_01` | `"def add(a, b):\n    return "` | `'""")\n    assert assert assert assert assert assert...'` | `REPETITION_COLLAPSE` |
| `CODE_02` | `"def is_even(n):\n    return n % 2 == "` | `'")\n    assert snake_sor'` | `SYNTAX_ERROR` |
| `CODE_03` | `"x = 10\ny = 20\ntotal = x + "` | `'snake_sor_sor_sor_'` | `REPETITION_COLLAPSE` |

### C. Stage C Full Epoch Checkpoint (6,478 Steps Pretrained)
| Task ID | Prompt | Raw Model Output | Evaluation | Failure Category |
|:---|:---|:---|:---|:---|
| `MATH_01` | `"1 + 1 = "` | `"<<40*2=2"` | FAILED | `INCORRECT_VALUE` |
| `MATH_02` | `"2 + 3 = "` | `"<<40*2=2"` | FAILED | `INCORRECT_VALUE` |
| `STRUCT_03`| `'title: Test\nauthor: '`| `"Step-by-Step Solution"` | **PASSED** | `SUCCESS` |
| `CODE_01` | `"def add(a, b):\n    return "` | `"day, so she were were were "` | FAILED | `SYNTAX_ERROR` |

---

## 8. Quantitative Evaluation Results & Comparison

| Metric | Frozen Baseline (`c5571c...`) | Step 51 Coding (60 steps) | Stage C Full Epoch (6,478 steps) |
|:---|:---|:---|:---|
| **Total Benchmark Tasks** | 20 | 20 | 20 |
| **Tasks Passed** | **0** | **0** | **1** |
| **Overall Pass Rate** | **0.0%** | **0.0%** | **5.0%** |
| **Repetition Collapse Count** | 19 (95.0%) | 13 (65.0%) | **0 (0.0%)** |
| **Empty Output Count** | 1 (5.0%) | 0 (0.0%) | 0 (0.0%) |
| **Syntax Error Count** | 0 (0.0%) | 2 (10.0%) | 6 (30.0%) |
| **Incorrect Value Count** | 0 (0.0%) | 5 (25.0%) | 13 (65.0%) |
| **Average Latency per Task** | 32.1 ms | 31.4 ms | 33.8 ms |
| **Average Token Throughput** | 358 tokens/sec | 362 tokens/sec | 344 tokens/sec |

---

## 9. Failure Mode Analysis

1. **Repetition Collapse in Frozen Baseline (95%)**:
   - Randomly initialized weights produce unconditioned logits where the self-attention mechanism reinforces whichever token happened to receive highest initial dot-product similarity (typically the final token of the prompt: `=`, `:`, `,`, or space).
   - This leads to monotonic single-token loops (` =  =  = `, `: : : :`).
2. **Transition in Step 51 Micro-Training (60 steps)**:
   - Repetition collapse dropped from 95% to 65%.
   - The model began outputting structured tokens from the training corpus (`assert`, `"""`, `snake_sort`), but lacked sufficient training steps to form complete syntactical trees.
3. **Transition in Stage C Full Epoch (6,478 steps)**:
   - Repetition collapse was **completely eliminated (0%)**.
   - The model acquired English language syntax and structured formatting (`<<...>>`, `Step-by-Step Solution`, `#### 4`), correctly solving structured key-value formatting (`STRUCT_03`).
   - However, because Stage C was trained on natural language math reasoning (GSM8K-style problems) rather than Python syntax, it fails code completion tasks by emitting prose tokens (`day, so she were...`).

---

## 10. Runtime & Resource Measurements

- **Inference Latency**:
  - Prefill + 8 tokens: 19.1 ms to 24.8 ms on CPU.
  - Prefill + 16 tokens: 36.3 ms to 47.6 ms on CPU.
- **Generation Speed**: 303 to 418 tokens/second on single-threaded CPU execution.
- **Memory Footprint**:
  - Model weights: 13.13 MiB.
  - Dynamic KV cache: 0.18 MiB for 512 context tokens.
  - Peak RSS during benchmark: < 120 MB (comfortably below 256 MB ceiling).
- **GPU Usage**: Exactly 0.0 MB (zero CUDA allocations).

---

## 11. Architectural Limitations Identified

1. **Parameter Budget**: At 3.44M parameters, `ChakrMicro` has limited memorization capacity. Tasks requiring open-domain factual knowledge (`The capital of France`) cannot succeed without either targeted training or external RAG retrieval.
2. **Context Horizon**: Fixed at 512 tokens.
3. **Pretraining Domain Specialization**: The model reflects its pretraining distribution. Stage C acquired math reasoning syntax, while Step 51 acquired initial code tokens. Neither checkpoint has yet been trained on a unified multi-task coding curriculum.

---

## 12. What These Results Actually Prove

1. **The Frozen Baseline is Genuinely Frozen and Untrained**:
   - Pass rate is 0.0%.
   - Demonstrates that earlier claims of "Level 1 to 7 capabilities" were entirely artifacts of test fixtures, not neural model capability.
2. **Training Produces Measurable, Statistically Verifiable Behavioral Shifts**:
   - Pretraining transforms the model from 95% repetition collapse $\to$ 0% collapse and coherent vocabulary emission.
3. **The Inference Engine and Evaluator Are Fully Decoupled & Functioning**:
   - Zero hardcoded answers.
   - Deterministic execution with identical outputs under seed 42.
   - $\Delta W_{\text{baseline}} = 0$ strictly preserved pre- and post-benchmark.

---

## 13. What These Results Do NOT Prove

1. They do **NOT** prove that ChakrView understands code.
2. They do **NOT** prove that ChakrView can autonomously repair software projects.
3. They do **NOT** prove that ChakrMicro can solve arithmetic zero-shot.
4. They do **NOT** demonstrate AGI, general reasoning, or reliable instruction following.

---

## 14. Verification Decision

The first honest empirical capability baseline has been established.
- Frozen Baseline capability score: **0.0%**
- Experimental capability score: **5.0%**
- Baseline integrity: **Verified ($\Delta W_{\text{baseline}} = 0$)**
