# Changelog

All notable changes to the ChakrView project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-07 (Release Candidate)

### Added
- **Verified Cognitive Core (I4)**:
  - Neural Relational Acquisition Module (`chakrview.cognition.neural_relational_acquisition`).
  - Two-hop compositional reasoning over disjoint unseen token pools.
  - Multi-seed G4 generalization mean = 77.78% (min seed 66.67%).
  - Hop-1 key routing = 100%, Hop-2 key routing = 88.89%.
  - Zero contamination and adversarial anti-shortcut resilience.
- **Model Registry & Governance**:
  - Formal candidate manifest infrastructure (`chakrview.cognition.i4_candidate_manifest`).
  - Immutable state machine lifecycle: `EXPERIMENTAL` -> `FROZEN` -> `RELEASED` (`chakrview.cognition.model_registry`).
  - Capability Contract v0.1.0 (`chakrview.cognition.capability_contract`).
  - Release Artifact Verifier (`chakrview.cognition.release_manifest_verifier`).
  - Unified Release Benchmark entry point (`chakrview.cognition.release_benchmark`).
- **CPU-First Runtime**:
  - Deterministic standalone local runtime (`chakrview.runtime.local_runtime`).
  - BPE Tokenizer with vocabulary size 4,096 (`chakrview.tokenizer`).
  - Strict resource budget guarantees (<512MB RAM, CPU-first, no GPU required).

### Verified Invariants
- **Canonical Baseline**: Bit-exact ChakrMicro model (`3,443,136` parameters, SHA-256: `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`).
- **Isolation**: Candidate relational module (69,809 parameters) operates without modifying baseline weights.
