# Initial Audit: Batches 211–220 Opening State & Diagnostics
**Milestone:** Batches 211–220: Relational Knowledge Learning, Exact-Value Binding, and Generalization Validation  
**Date:** 2026-10-13  
**Branch:** `dev/v0.6-v0.7-validation`  

---

## 1. Git Status & Remote Tracking State
- **Branch:** `dev/v0.6-v0.7-validation`
- **Latest Commit:** `720e2df` (`feat(batch201-210): integrate grounded interactive runtime`)
- **Remote:** `origin/dev/v0.6-v0.7-validation` (up to date with `720e2df`).
- **Working Tree:**
  - `chakrview/runtime/session.py` contains pre-existing user modifications (strictly preserved).
  - Unrelated untracked files in `docs/sprints/` and `experiments/` remain untouched.

---

## 2. Canonical Checkpoint & Architecture Verification
- **Checkpoint File:** `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
- **SHA-256 Digest:** `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (Exact Match).
- **Total Unique Model Parameters:** `3,443,136` (6 layers, `d_model=192`, SwiGLU, RMSNorm, RoPE, max sequence length 512).
- **Integrity Status:** Intact. Zero parameter modification.

---

## 3. Environment & Hardware Capabilities
- **Platform:** Windows-11-10.0.26300-SP0, x86_64 AMD64
- **Python Version:** 3.14.7
- **PyTorch Version:** 2.6+
- **CUDA Available:** `False` (Device count: 0)
- **Available System Memory:** ~19.5 GB available out of 31.64 GB total RAM.
- **Hardware Policy:** CPU-first execution with bounded sequence lengths and memory constraints.

---

## 4. Prior Results Reproduction & Ground Truth Audit
- **Re-run Test Suite:** `tests/test_batch201_210_runtime_integration.py` -> 4/4 passed (1.39s).
- **Audit of Prior Integrated Score (8/8 in Batch 207):**
  - Investigated `experiments/batch201_210_runtime_integration/reports/interactive_runtime_benchmark.json`:
    - Turn 1 (Circle area formula): `CONSTRAINED_TRIE` (`2 * pi`).
    - Turn 2 (Pronoun circumference): `NEURAL_CORE` dialogue continuation.
    - Turn 3 (QuickSort complexity): `CONSTRAINED_TRIE` (`O(n^2)`).
    - Turn 4 (Carbon atomic number): `DETERMINISTIC_EXTRACTOR` (`6`).
    - Turn 5 (Hindi Gold melting point): `NEURAL_CORE` dialogue continuation.
    - Turn 6 (Boiling point conflict): `DETERMINISTIC_EXTRACTOR` (detected `-196 C` vs `-190 C`).
    - Turn 7 (Mars capital ungrounded): `DETERMINISTIC_EXTRACTOR` abstention (*"Not mentioned in context"*).
    - Turn 8 (Tenant isolation): `persistent_memory` zero history isolation.
  - **Key Finding:** The 8/8 integrated result relied predominantly on deterministic extraction, candidate trie selection, and memory isolation. The underlying pure neural core scored 0/4 on held-out relational extraction.
- **Goal for Batches 211–220:** Synthesize a scaled relational corpus (10,000+ records) across Mathematics, Coding, and Science, train an isolated relational pointer head with leakage-resistant evaluation slices, and rigorously test exact-value binding against distractors.
