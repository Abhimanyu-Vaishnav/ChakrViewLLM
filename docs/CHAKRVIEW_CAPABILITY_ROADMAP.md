# CHAKRVIEW CAPABILITY ROADMAP & VERSION CONTRACT

## 1. Principles and Versioning Contract

ChakrView is an experimental from-scratch neural language model and adaptive AI architecture.
Its version milestones represent **demonstrated, tested, and validated progress** rather than theoretical roadmaps or wrapper code.

### Status Taxonomy:
- `IMPLEMENTED`: Built, integrated, and verified against functional regression tests.
- `TESTED`: Thoroughly benchmarked with frozen held-out suites and reproducible artifacts.
- `EXPERIMENTAL`: Prototype implementation present; behavior actively researched or partially validated.
- `PLANNED`: Documented design and specification; implementation not yet started.
- `BLOCKED`: Dependency, hardware, or theoretical constraint prevents execution.

---

## 2. Version Contracts

### v0.6.0 — Research Preview & Reproducible Baseline
- **Release Commit**: `12e6e74bb3e08d0c595974a24d661ce19bae1e45`
- **Scope**: Canonical ChakrMicro 3.44M architecture, deterministic BPE tokenizer (vocab 4096), initial 10,000-step pretraining checkpoint, foundational runtime session, and initial smoke harness.
- **Status**: `TESTED` & Published.

### v0.7.0 — Coding Foundation, Subprocess Isolation & Release Gate
- **Scope**:
  - Secure subprocess-isolated Python test runner with AST safety pre-auditing (`IMPLEMENTED`, `TESTED`).
  - Coding task benchmark (12 tasks) with strict syntax vs. functional test separation (`TESTED`).
  - Sampling subsystem with deterministic greedy/stochastic modes and sliding-window n-gram repetition blocking (`IMPLEMENTED`, `TESTED`).
  - Hardened inference runtime adapter enforcing CPU execution and Delta-W = 0 verification (`IMPLEMENTED`, `TESTED`).
  - Clean separation of native neural generation from deterministic tools (`IMPLEMENTED`).
- **Release Status**: **APPROVED FOR MERGE / RELEASE CANDIDATE**.

### v0.8.0 — Native Context Grounding & SFT Specialization
- **Scope**:
  - 32-task 10-category versioned Native Intelligence Benchmark (`IMPLEMENTED`, `TESTED`).
  - Controlled 200-step Supervised Fine-Tuning (SFT) on response-masked context copying (`PLANNED — PENDING USER AUTHORIZATION`).
  - Elimination of synthetic memorization; generalization to unseen IDs and entities (`PLANNED`).
  - Attractor loop suppression across long-sequence completions (`TESTED` via decoding n-gram blocking; `PLANNED` via loss regularization).

### v0.9.0 — Extended Context, Reasoning & Safe Tools
- **Scope**:
  - Context horizon expansion from 512 to 1024 / 2048 tokens via RoPE frequency scaling (`PLANNED`).
  - Multi-step structured chain-of-thought routing (`EXPERIMENTAL`).
  - Governed tool orchestration with AST safety and rate limiting (`IMPLEMENTED` for local calculator and sandbox; `PLANNED` for web search).
  - Multi-turn conversation state persistence (`EXPERIMENTAL`).

### v1.0.0 — Production-Ready Coding Model & Usable Brain
- **Scope**:
  - Fully specialized coding model achieving >= 80% pass rate on core function synthesis (`PLANNED`).
  - Comprehensive memory store with semantic retrieval and attribution provenance (`EXPERIMENTAL`).
  - Safe error self-correction and automated execution repair loops (`EXPERIMENTAL`).
  - Multi-backend hardware engine (graceful CPU, CUDA, MPS acceleration) (`EXPERIMENTAL`).

---

## 3. Comprehensive Capability Tracking Matrix

| Capability Dimension | Current Status | Demonstrated Evidence / Blocker |
| :--- | :--- | :--- |
| **Neural Core & Tokenizer** | `TESTED` | ChakrMicro 3.44M params, 4096 BPE tokenizer bit-exact verification. |
| **Hardware Adaptation (CPU/GPU)**| `IMPLEMENTED` | CPU execution strictly verified; graceful detection of hardware topology without false GPU claims. |
| **Subprocess Code Isolation** | `TESTED` | AST policy validation, resource limits, process isolation (`tests/test_security_hardened_executor.py`). |
| **Deterministic Sampling Engine** | `TESTED` | Multiplicative repetition penalty + n-gram blocking breaking greedy loops (`tests/test_sampling_ngram_blocking.py`). |
| **Native Context-Grounding** | `BLOCKED` | Base 3.44M pretrained weights lack copy-mechanism priors (0/4 on synthetic IDs). Requires SFT. |
| **Prompt Instruction Adherence** | `EXPERIMENTAL` | Base model responds with math/declarative continuations; requires instruction tuning. |
| **Tool vs. Neural Separation** | `TESTED` | CLI and test harnesses clearly differentiate native output from deterministic calculator/sandbox tools. |
| **Memory & Multi-turn Retention**| `EXPERIMENTAL` | In-memory conversation state implemented; neural model does not natively condition on past turns. |
| **Retrieval with Provenance** | `EXPERIMENTAL` | Semantic retrieval engine exists in runtime; citation provenance not yet bound to model generation. |
| **Safe Self-Improvement / Loops**| `EXPERIMENTAL` | `improvement.py` and `answer_verifier.py` present; weight-modifying self-updates forbidden. |
| **Humour & Emotional Cues** | `PLANNED` | Unimplemented in 3.44M core; deferred to v0.9+ fine-tuning. |
| **Distributed Computing Engine** | `PLANNED` | Architecture supports RPC abstractions; multi-node training blocked on infrastructure. |

---

## 4. Capability Metrics & Regression Gates

All future model iterations must run against the two frozen, reproducible benchmark suites:
1. **12-Task Coding Benchmark**: Must exceed 2/12 functional pass rate without regressions in syntax validity (>= 5/12).
2. **32-Task Native Intelligence Suite**:
   - Zero attractor collapse loops (must maintain <= 3% collapse rate).
   - Context copying pass rate must exceed 50% on unseen synthetic identifiers.
   - Exact-match instruction following must exceed 25%.