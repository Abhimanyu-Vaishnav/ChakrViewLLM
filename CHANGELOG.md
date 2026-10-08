# Changelog

All notable changes to the ChakrView project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
