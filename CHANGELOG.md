# Changelog

## [v0.7.0] - 2026-10-20

### Added
- **Subprocess-Isolated Test Runner** (`evaluation/coding_suite/isolated_executor.py`): Restricted Python execution environment with AST safety auditing, timeout enforcement, and memory bounding for coding test assertions.
- **Coding Assistant Subsystem & CLI** (`chakrview/runtime/coding_assistant.py`): Supports `draft`, `explain`, `fix`, `test`, and `sandbox` actions with AST-governed safety.
- **Sliding-Window N-Gram Repetition Blocking** (`chakrview/runtime/sampling.py`): Added `apply_no_repeat_ngram` and `no_repeat_ngram_size` to sampling configurations, eliminating cyclic repetition attractor loops (dropping collapse from 53.1% to 0.0% on greedy decoding).
- **10-Category 32-Task Native Intelligence Suite** (`evaluation/intelligence_suite/`): Versioned native benchmark covering greetings, instructions, context copying, grounded QA, summarization, arithmetic, formatting, coding continuation, repetition probes, and multi-turn context.
- **Robust Scoring Evaluator** (`evaluation/intelligence_suite/scoring_evaluator.py`): Replaces naive substring matches with word boundaries, number isolation, and isolated subprocess execution for coding tasks.
- **Coding Data Pipeline** (`chakrview/data/coding_data_pipeline.py`): Dataset normalization with exact deduplication and cross-split leakage prevention.
- **Project North Star & Living Roadmap** (`docs/PROJECT_NORTH_STAR.md`, `docs/CHAKRVIEW_CAPABILITY_ROADMAP.md`).
- **Release Documentation Package** (`docs/release/v0.7.0/`): Release checklist, release notes, reproducibility guide, and final execution report.

### Changed
- `chakrview/runtime/neural_runtime_adapter.py`: Integrated `no_repeat_ngram_size` parameter and dynamic external checkpoint resolution fallback with SHA-256 integrity validation.
- `chakrview/runtime/assistant_cli.py`: Added `code` action subgroup and integration with coding assistant.

### Verified Baseline Performance
- 12-Task Coding Benchmark: 2/12 (16.7%) functional pass rate, 5/12 (41.7%) syntax valid.
- 32-Task Intelligence Suite: 1/32 (3.1%) pass rate; attractor collapse reduced from 53.1% to 0.0% with `no_repeat_ngram_size=3`.

---


All notable changes to the ChakrView project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2026-10-10

### Added
- **Hardware-Adaptive Execution Engine (`HardwareAdaptiveGovernor`)**:
  - Dynamic discovery of host CPU, RAM, GPU, and VRAM with profile selection (`CPU_ONLY`, `BALANCED`, `CPU_HIGH_MEMORY`, `LOW_RESOURCE`).
  - Safe GPU inference device placement reconciliation across model parameters, input tokens, and KV cache with automatic fallback to CPU.
  - Chunked adaptive execution with user-activity cooperation and memory boundaries.
- **Durable Task Recovery (`DurableExecutionJournal`, `DurableChunkedExecutor`)**:
  - Atomic disk-backed execution journaling, SHA-256 tamper-evident integrity checks, and process restart resumption.
- **Relational Semantic Grounding & Compositional Language Understanding**:
  - Verbatim surface normalization (`TextNormalizer`) preserving proper nouns across English and common conversational Hinglish patterns.
  - Compositional sentence parsing (`CompositionalSentenceParser`) extracting canonical relational triples `(Subject, Relation, Object)`.
  - Multi-hop path transitivity evaluation over relational graphs without lookup tables.
- **Semantic Cognitive Coupling & Epistemic Abstention**:
  - Integration of relational grounding into cognitive decision frames (`ANSWER`, `ABSTAIN`, `REPORT_CONFLICT`).
  - Precedence of verified external evidence over stale internal beliefs.
- **Auditable QA Review & Founder Acceptance Tooling**:
  - Tracked founder review workflow CLI (`tools/qa_review/founder_review_workflow.py`) with fail-closed gates.
  - Automated 15-test founder acceptance suite (`scripts/founder_manual_test.py --suite`).
  - Strict 14-point release candidate verification script (`scripts/verify_v050_release.py`).

### Fixed
- **CUDA Inference Device Placement**:
  - Resolved tensor device mismatch between CPU model parameters and CUDA input tokens.
  - Reconciled physical device comparison in KV-cache (`cuda` vs `cuda:0`).
  - Added safe CPU fallback on unsupported or invalid GPU device requests.

### Verified Invariants
- **Canonical Model Immutability**:
  - Bit-exact 3,443,136 parameters, SHA-256 `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` ($\Delta W = 0$).
- **Protected Checkpoints & Manifests**:
  - Stage C baseline (`3e03ae0a...`), Stage D pilot (`8ab1ec36...`), and QA pilot manifest (`b7ebf80c...`) verified bit-exact.
- **Founder Authority & QA Queue**:
  - All 30 candidate records in `qa_review_queue_v0.3.0.jsonl` strictly maintained as `UNREVIEWED`.

### Limitations & Research Scope
- **Experimental Research Preview**: Designated strictly for research exploration; not approved for safety-critical, legal, medical, or unsupervised decisions.
- **Neural Reasoning**: The microscopic 3.44M-parameter core generates repetitive language under greedy decoding and does not possess generalized zero-shot reasoning. Factual answers originate from governed cognitive verification, not unconstrained neural generation.

## [0.2.0] - 2026-10-08

### Added
- **Unified Cognitive Runtime (`UnifiedCognitiveRuntime`)**:
  - Full end-to-end integration of language baseline, neural routing gateway, recursive reasoning core, and meta-governance.
  - Sub-second execution on commodity CPUs with peak RSS memory < 185 MB.
- **Learned Neural Cognitive Routing Gateway**:
  - 74,016-parameter neural router providing dynamic routing between language representation and cognitive reasoning.
- **I5 Recursive Multi-Hop Compositional Reasoning**:
  - Validated multi-step reasoning: 1-Hop (100.0%), 2-Hop (95.0%), 3-Hop (92.5%), 4-Hop (86.0%).
- **Governed Relational Iterative Learning (RIL)**:
  - Episodic experience buffer capturing structured trajectories.
  - Governed candidate updates yielding 86.0% accuracy on previously unseen relational tasks (+12.5% improvement over naive baseline).
- **Meta-Cognitive Governance & Peering**:
  - Discrimination margin 0.55 between correct and incorrect paths.
  - Dynamic strategy selection (92.0% accuracy) with self-correction recovery (80.0%).
  - Cryptographic peer request/verification with Byzantine fault recovery and 0% task failure under peer drops.
- **Release Hardening & Auditing (Waves 512–519)**:
  - Clean isolated subprocess verification.
  - Deterministic release artifact inventory and SHA-256 validation.
  - Zero security credential leaks across 1,737 tracked files.
  - Deterministic release bundle generator and single-command founder verification script (`scripts/verify_v020.py`).
  - Apache-2.0 License adoption.

### Verified Invariants
- **Canonical Baseline**: Bit-exact 3,443,136 parameters, SHA-256 `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` ($\Delta W = 0$).
- **Isolation & Provenance**: Zero parameter contamination across training/eval splits and explicit disambiguation of historical vs v0.2.0 benchmarks.

## [0.1.0] - 2026-10-07

### Added
- **Verified Cognitive Core (I4)**:
  - Neural Relational Acquisition Module (`chakrview.cognition.neural_relational_acquisition`).
  - Two-hop compositional reasoning over disjoint unseen token pools (G4 mean = 77.78%).
  - Hop-1 key routing = 100%, Hop-2 key routing = 88.89%.
- **Model Registry & Governance**:
  - Formal candidate manifest infrastructure and state machine lifecycle.
  - Capability Contract v0.1.0 and release benchmark.
- **CPU-First Runtime**:
  - Standalone local runtime and BPE tokenizer (vocab size 4,096).
