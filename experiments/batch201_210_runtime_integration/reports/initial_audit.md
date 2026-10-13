# Initial Audit: Batches 201–210 Opening State & Diagnostics
**Milestone:** Batches 201–210: Interactive Runtime Integration, Multilingual Relation Binding, and Honest Capability Validation  
**Date:** 2026-10-13  
**Branch:** `dev/v0.6-v0.7-validation`  

---

## 1. Git Status & Remote Tracking State
- **Branch:** `dev/v0.6-v0.7-validation`
- **Latest Commit:** `e199c45` (`feat(batch191-200): improve query-aware pointer and grounded conversation`)
- **Remote:** `origin/dev/v0.6-v0.7-validation` (up to date with `e199c45`).
- **Working Tree:**
  - `chakrview/runtime/session.py` contains pre-existing user modifications (preserved untouched).
  - Unrelated untracked files in `docs/sprints/` and `experiments/` remain unmodified and preserved.

---

## 2. Canonical Checkpoint & Architecture Verification
- **Checkpoint File:** `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
- **SHA-256 Digest:** `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (Exact Match).
- **Total Unique Model Parameters:** `3,443,136` (Tied token embeddings: 4096 x 192).
- **Architecture Invariants:** 6 transformer layers, `d_model=192`, SwiGLU, RMSNorm, RoPE, max sequence length 512.
- **Integrity Status:** Strictly intact. Zero parameter modification.

---

## 3. Environment & Hardware Capabilities
- **Platform:** Windows-11-10.0.26300-SP0, x86_64 AMD64
- **Python Version:** 3.14.7
- **PyTorch Version:** 2.6+
- **CUDA Available:** `False` (Device count: 0)
- **Available System Memory:** 19.53 GB available out of 31.64 GB total RAM.
- **Hardware Policy:** Strictly adaptive CPU-first inference with bounded tensors. No CUDA execution assumed.

---

## 4. Reproduction of Prior Results & Regression Status
- **Prior Tests:** Ran `tests/test_batch191_200_grounded_conversation.py` -> 4/4 passed (1.43s).
- **Prior Regression Suites:** Ran batches 147 to 190 suites -> 25/25 passed.
- **Total Regression:** 29/29 tests passed.
- **Prior Benchmark Claims Review:**
  - Query-aware pointer greedy: 33.3% EM on 6 test cases (2/6 correct).
  - Standard pointer greedy: 16.7% EM (1/6 correct).
  - TokenTrie candidate decoding: 66.7% EM on candidate spans (4/6 correct).
  - Deterministic extractor oracle: 100% on specific template extraction.
  - Denominator sizes are small (6 test cases). Truthful attribution is required: deterministic and constrained Trie selection must never be reported as proof of open-ended neural intelligence.

---

## 5. Objectives for Batches 201–210
1. Integrate the query-aware pointer and constrained Trie decoder into the interactive REPL and CLI runtime (`chakrview/runtime/session_adapter.py` and `chakrview/runtime/interactive.py`).
2. Implement explicit component provenance metadata distinguishing neural generation, pointer copying, constrained Trie selection, deterministic extraction, and abstention.
3. Construct a multi-domain relation binding dataset (Mathematics, Coding, Science) in English and Hindi with explicit leakage protection.
4. Benchmark unconstrained vs constrained decoding under memory and latency bounds.
5. Validate full end-to-end interactive conversation and deliver verified commit to `dev/v0.6-v0.7-validation`.
