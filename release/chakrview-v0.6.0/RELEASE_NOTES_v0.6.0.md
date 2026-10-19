# ChakrView v0.6.0 Release Notes

**Version**: `v0.6.0-rc1`  
**Classification**: **Experimental Research Preview**  
**Target Release Date**: 2026-11-02  
**Status**: `RELEASE_CANDIDATE_READY` (Local Candidate)

---

## 1. Release Overview

ChakrView v0.6.0 is an accelerated research-preview release focusing on **working software, validated inference, reproducible packaging, and truthful capability boundaries**. 

This release stabilizes the indigenous neural runtime stack on local CPU hardware, providing an easy-to-use, deterministic programmatic and CLI interface for autoregressive inference using the canonical pretraining checkpoint (**ChakrMicro Step 10,000**).

---

## 2. Key Highlights & Deliverables

1. **Standardized Neural Runtime Adapter (`NeuralRuntimeAdapter`)**:
   - Single-entry point for checkpoint loading, BPE tokenization, greedy/sampling generation, and Delta-W immutability verification.
   - Lazy model initialization with fast cold start (~444 ms on standard multi-core CPUs).
   - CPU-native autoregressive inference operating at ~136 tokens/second with under 275 MB process memory footprint.
2. **Unified CLI Runtime (`chakrview.runtime.assistant_cli`)**:
   - System hardware profiling (`status`), document ingestion (`ingest`), hybrid retrieval QA (`ask`), and interactive multi-turn exploration (`interactive`).
3. **Deterministic Verification & Benchmarking Suite**:
   - 7 automated release smoke tests verifying bit-exact SHA-256 artifact hashes, parameter shapes, context bounds, and deterministic greedy generation.
   - Standardized local benchmark script (`benchmark_v060.py`) reporting cold-start time, TTFT, throughput, and memory.
4. **Actionable Structured Extraction Roadmap**:
   - Formal proposal for the next controlled training experiment (Single-Field Entity Copy Pilot, 250–500 steps) based on rigorous empirical failure analysis.

---

## 3. Canonical Artifact Provenance & Checksums

The canonical neural core and tokenizer files are frozen and cryptographically verified:

| Artifact | Local File Path | Verified SHA-256 Checksum |
| :--- | :--- | :--- |
| **ChakrMicro Step-10,000 Checkpoint** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` |
| **Canonical BPE Vocabulary** | `data/experiments/vocab_4096/vocab.json` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` |
| **Canonical BPE Merges** | `data/experiments/vocab_4096/merges.json` | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` |

---

## 4. What Changed Between v0.5.0 and v0.6.0

- **Neural Architecture**: Preserved invariant ($\Delta W = 0$). Canonical parameter count is 3,443,136 weights (4,229,568 total tensor elements across 57 state dict keys with tied embeddings).
- **Runtime Stabilization**: Replaced fragmented inference scripts with `NeuralRuntimeAdapter`, enforcing fail-closed artifact allowlisting and structured metadata output.
- **Diagnostics Completed**: Integrated empirical insights from SFT V3 pilot analysis and copy capability gates. Documented why prompt framing eliminates doctest attractors but requires specialized single-field copying curricula to establish true neural entity extraction.
- **Safety Enforcement**: Protected runtime files (notably `chakrview/runtime/session.py`) and held-out test splits were strictly safeguarded.
