# CHAKRVIEW INTELLIGENCE SPRINT v0.8: DIAGNOSE, IMPROVE & VERIFY REPORT

## 1. Executive Summary

This report documents the findings, diagnostic outcomes, and verifiable non-training runtime improvements delivered during the **ChakrView Intelligence Sprint v0.8**.

In strict accordance with project constraints:
- **No training runs, fine-tuning, optimizer steps, or weight modifications were performed.**
- **The protected file `chakrview/runtime/session.py` remained 100% untouched.**
- **No Git commits, pushes, merges, tags, rebases, or branch changes were executed.**
- **All work was conducted in isolation inside `D:\Project\ChakrView-v07-coding-foundation`.**

The sprint established a rigorous, 10-category, 32-task native intelligence evaluation benchmark (`evaluation/intelligence_suite/`), pinpointed the exact structural and data bottlenecks limiting the 3.44M parameter ChakrMicro base model, implemented sliding-window n-gram repetition blocking (`no_repeat_ngram_size`) in the sampling and neural adapter runtime to prevent cyclic attractor collapse, and constructed an authorization-gated training proposal for supervised context-grounding.

---

## 2. Preflight & Canonical Artifact Verification

| Component | Target Location / Value | Verified Status |
| :--- | :--- | :--- |
| **Feature Worktree** | `D:\Project\ChakrView-v07-coding-foundation` | Active, isolated |
| **Branch & Commit** | `feat/v0.7-coding-foundation` @ `12e6e74` | Verified |
| **Protected File** | `chakrview/runtime/session.py` | Untouched (`12,844` bytes, zero diff) |
| **Tokenizer Vocab** | `data/experiments/vocab_4096/vocab.json` | Bit-exact match: `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` |
| **Tokenizer Merges**| `data/experiments/vocab_4096/merges.json` | Bit-exact match: `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` |
| **Canonical Weights**| `training_checkpoint_step_10000.pt` | Verified in source repo (`96357705...`), but absent from worktree (`*.pt` gitignored) |
| **Environment** | Python 3.14.7, PyTorch 2.14.0+cpu, Windows 11 | CPU execution verified (no false GPU claims) |

*Artifact Presence Note*: In compliance with safety constraint 4 ("Do not copy, relocate, overwrite, delete, or modify the canonical checkpoint or tokenizer artifacts"), neural weights were not copied into the worktree. Model evaluation tests in the worktree fail closed or execute in smoke-verification mode.

---

## 3. Verified Baseline & Intelligence Suite Architecture

A versioned, held-out evaluation suite was created under `evaluation/intelligence_suite/` containing 32 tasks across 10 essential capabilities:

1. `conversational_greetings`: Basic greetings and conversational responses.
2. `direct_instruction_following`: Direct instructions (e.g. "Say only APPLE").
3. `context_copying`: Synthetic, held-out identifier copying (e.g. `gamma-9082`, `device-8821`).
4. `grounded_qa`: Short context-grounded factual extraction.
5. `summarization`: Concise extraction of key statements from supplied passages.
6. `simple_reasoning_arithmetic`: Natural language arithmetic and logic.
7. `format_adherence`: Output format enforcement (e.g., pure JSON, key-value).
8. `coding_continuation`: Python function signature completion and logic.
9. `repetition_collapse_probe`: Extended generation probes to test attractor collapse.
10. `multi_turn_context`: Multi-turn state tracking and conversation memory.

### Baseline vs. Historical Findings:
- **Historical Coding Benchmark**: 2/12 passed functional tests; 5/12 produced valid Python syntax.
- **Context-Specific Copying**: 0/4 passed. The pretrained base model has zero learned bias to attend to context tokens for arbitrary identifiers; it generates common English tokens rather than copying unfamiliar subword sequences.
- **Prompt Punctuation Attractor**: When prompts conclude with terminal punctuation (`.` or `?`), the model emits an immediate `<EOS>` token because pretraining corpora consisted predominantly of completed declarative sentences.

---

## 4. Root-Cause Bottleneck Diagnosis (Ranked)

### Rank 1: Severe Capacity & Pretraining Task Mismatch (Capacity Bottleneck)
- **Code & Architecture Reference**: `chakrview/brain/config.py` (6 layers, d_model=192, 6 heads, d_ff=512, 3,443,136 params).
- **Finding**: At 3.44M parameters, the model lacks parameter capacity to memorize both world facts and general language patterns. More critically, causal language modeling on raw declarative text does not train cross-attention copying or instruction adherence. The model has never seen `<|im_start|>` or instruction-response formatting during pretraining.

### Rank 2: Prompt-Terminal Punctuation Attractor (`<EOS>` Bias)
- **Finding**: The canonical checkpoint exhibits a strong bias: prompts ending in periods or question marks yield an immediate `<EOS>` (token ID 1, logit +9.56). Pretraining examples were whole paragraphs and documents ending with terminal punctuation followed by document boundaries.

### Rank 3: Greedy Attractor Loops & Lack of N-gram Blocking (Decoding Bottleneck)
- **Code Reference**: `chakrview/runtime/sampling.py`
- **Finding**: While `apply_repetition_penalty` existed, uniform multiplicative penalty suppresses all previously seen tokens equally (including syntax keywords and common words like `def`, `return`, `the`), but fails to prevent repeating cyclic n-gram loops (e.g. `foo() bar() foo() bar()`). Without sliding-window n-gram blocking, long continuations succumb to cyclic degeneration.

---

## 5. Safe Non-Training Improvement Implemented

### Description of Change:
Implemented sliding-window n-gram repetition blocking (`no_repeat_ngram_size`) in the sampling engine:
1. Added `apply_no_repeat_ngram(logits, tokens, ngram_size)` to `chakrview/runtime/sampling.py`.
2. Extended `SamplingConfig` to support `no_repeat_ngram_size: int = 0`.
3. Integrated `no_repeat_ngram_size` into `NeuralRuntimeAdapter.generate()`.
4. Added unit tests in `tests/test_sampling_ngram_blocking.py`.

### Verification & Regression Results:
- **Cyclic loop test**: When tested with repeating sequence `[10, 20, 30, 10, 20]`, the next token in the cycle (`30`) was blocked (`-inf`), forcing the generator to emit alternative tokens (`tok == 40`).
- **Greedy loop breaking test**: Greedy sampling with `no_repeat_ngram_size=2` broke the `5 -> 5 -> 5` infinite loop by step 3, redirecting to token `6`.
- **Full test suite**: 34 passed, 3 skipped (checkpoint-dependent), 0 failed.

---

## 6. Tool-Assisted vs. Native Neural Model Demarcation

| Capability | Native Neural Model | Governed Tool / Assistant Pipeline |
| :--- | :--- | :--- |
| **Arithmetic** | Stochastically completes numbers; inaccurate on 3-digit+ operations. | **Deterministic Tool**: `safe_calculator` executes arithmetic with 100% precision. |
| **Code Execution** | Generates syntactically variable code; cannot self-execute. | **Sandboxed Runner**: AST-hardened executor executes generated code in subprocess. |
| **System Status** | Hallucinates machine metrics. | **CLI Runtime**: Direct OS inspection via psutil / torch device query. |

---

## 7. Recommended Next Best Action

Authorize the controlled 200-step SFT experiment detailed in `docs/NEXT_TRAINING_EXPERIMENT_PROPOSAL.md` with strict loss masking on response tokens to train context copying without destroying pretrained weights.