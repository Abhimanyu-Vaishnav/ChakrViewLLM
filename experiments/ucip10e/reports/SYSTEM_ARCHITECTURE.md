# UCIP-10E System Architecture & Infrastructure Design

**Date:** 2026-11-18  
**Mission:** UCIP-10E ChakrView Intelligence Evaluation, Measurement & Regression Infrastructure  
**Repository:** `D:\Project\ChakrView`  
**Package / Path:** `experiments/ucip10e/`

---

## 1. Architectural Philosophy & Isolation

The UCIP-10E evaluation infrastructure provides a permanent, reusable, and scientifically rigorous intelligence measurement framework for ChakrView. It operates under strict project principles:
1. **Zero External AI / LLM APIs:** Evaluation grading is performed exclusively by deterministic mathematical oracles, symbolic parsers, and sandboxed execution environments.
2. **Unified Model Adapter:** Checkpoint loading is completely decoupled from file paths or hardcoded architectures. The `ModelAdapter` inspects internal configuration, verifies parameter shapes, and runs deterministic autoregressive decoding with repetition and n-gram penalty controls.
3. **Strict Scoring Over Heuristics:** Permissive length-only and keyword-only heuristics from prior iterations are eradicated. Every probe specifies an explicit semantic rubric, exact numerical target, or sandboxed unit test.

---

## 2. Directory Layout & Module Structure

```text
experiments/ucip10e/
├── config/              # Run configurations and hyperparameters
├── manifests/           # Versioned registry and SHA-256 manifests
│   └── manifest_registry.json
├── datasets/            # Versioned benchmark probe suites
│   ├── development/     # benchmark_dev.json (840 probes)
│   └── sealed/          # benchmark_sealed.json (840 probes) & benchmark_sealed_core100.json
├── evaluators/          # Strict domain-specific evaluators
│   └── strict_evaluators.py
├── runners/             # Model adapters and execution engine
│   ├── model_adapter.py
│   └── eval_engine.py
├── validators/          # Manifest and probe schema validators
├── schemas/             # JSON schemas for probes and results
│   └── probe_schema.json
├── reports/             # Comprehensive mission documentation reports
├── results/             # Raw model generations and evaluation logs
├── scripts/             # Manifest generation and data utilities
│   ├── build_benchmark_suites.py
│   └── build_core100_suite.py
└── tests/               # Automated unit tests for infrastructure and evaluators
    ├── test_evaluator_integrity.py
    └── test_ucip10e_regression.py
```

---

## 3. CLI Interfaces & Tooling

The evaluation infrastructure is accessible via a command-line interface:
- `python experiments/ucip10e/runners/eval_engine.py validate`: Validates manifest integrity and hashes.
- `python experiments/ucip10e/runners/eval_engine.py list`: Lists available benchmark suites and domains.
- `python experiments/ucip10e/runners/eval_engine.py evaluate --checkpoint <path> --suite <name>`: Runs deterministic evaluation and records raw outputs into `experiments/ucip10e/results/`.
- `python experiments/ucip10e/runners/eval_engine.py compare --baseline <path> --candidate <path>`: Computes category-by-category delta and flags regressions.
