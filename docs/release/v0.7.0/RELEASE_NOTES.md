# ChakrView v0.7.0 Release Notes — Coding Foundation & Runtime Hardening

## Overview
ChakrView v0.7.0 delivers the foundational architecture for coding-specialized workflows, secure subprocess execution, robust benchmark evaluation, and output-level repetition controls.

In accordance with ChakrView's Project North Star, this release emphasizes **honest capability measurement, process isolation, and clean separation between native neural generation and deterministic tools**.

---

## What is New in v0.7.0

### 1. Hardened Subprocess Test Runner (`evaluation/coding_suite/isolated_executor.py`)
- Executes generated code and test assertions in an isolated subprocess.
- Strict AST pre-auditing blocks dangerous builtins (`eval`, `exec`, `open`, `__import__`) and introspective attributes (`__subclasses__`, `__globals__`).
- Wall-clock timeouts prevent CPU-hanging infinite loops.
- Provides defense-in-depth for local benchmark evaluation. *(Notice: This is a restricted local test runner, not an operating-system-level hypervisor jail).*

### 2. Coding Assistant Subsystem & CLI (`chakrview/runtime/coding_assistant.py`)
- Multi-action coding interface supporting:
  - `draft`: Generate code continuations from function signatures or natural language prompts.
  - `explain`: Analyze and summarize supplied code snippets.
  - `fix`: Suggest repairs for broken code and error traces.
  - `test`: Synthesize candidate unit test cases.
  - `sandbox`: Execute code with assertion fixtures inside the isolated runner.
- Integrated into `assistant_cli.py` via `python -m chakrview.runtime.assistant_cli code <action>`.

### 3. Sliding-Window N-Gram Repetition Blocking (`chakrview/runtime/sampling.py`)
- Implemented `apply_no_repeat_ngram(logits, tokens, ngram_size)` and `no_repeat_ngram_size` in the sampling engine.
- Banned cyclic n-gram repetitions during greedy and stochastic generation.
- Empirically verified to reduce long-sequence attractor collapse loops from **53.1% to 0.0%** (`no_repeat_ngram_size=3`).
- *Honest Demarcation*: N-gram blocking is an output presentation technique; it does not alter weights or create missing model intelligence.

### 4. 10-Category Native Intelligence Suite (`evaluation/intelligence_suite/`)
- 32 held-out evaluation tasks across 10 core capability dimensions: greetings, direct instructions, context copying, grounded QA, summarization, arithmetic reasoning, format adherence, coding continuation, attractor probes, and multi-turn context.
- Robust scoring evaluator (`scoring_evaluator.py`) prevents false-positive substring matches (e.g. number isolation, word boundaries).
- Real isolated test execution for coding tasks.

### 5. Leakage-Proof Coding Data Pipeline (`chakrview/data/coding_data_pipeline.py`)
- Ingestion and normalization pipeline for coding datasets.
- Strict deduplication and cross-split leakage detection using normalized token hashing.

---

## Verified Baseline Model Capabilities (Step 10,000 Checkpoint)

All measurements performed against the canonical 3.44M parameter ChakrMicro checkpoint (Step 10,000, SHA-256 `96357705...`):
- **12-Task Coding Benchmark**:
  - Functional pass rate: **2/12 (16.7%)**
  - Syntax valid rate: **5/12 (41.7%)**
- **32-Task Native Intelligence Suite**:
  - Baseline pass rate: **1/32 (3.1%)**
  - Repetition attractor loops: **17/32 (53.1%)** under default greedy; reduced to **0/32 (0.0%)** with `no_repeat_ngram_size=3`.
  - Context-Specific Copying: **0/4 (0.0%)** (Model lacks learned copy priors for unfamiliar synthetic identifiers).

---

## Compatibility & Integrity
- **Protected File**: `chakrview/runtime/session.py` is completely untouched.
- **Base Checkpoint**: Canonical weights and BPE tokenizer artifacts remain intact.
- **Delta-W = 0**: Zero model weights were modified in this release.