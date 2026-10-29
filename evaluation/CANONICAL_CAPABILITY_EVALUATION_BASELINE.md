# CHAKRVIEW CANONICAL CAPABILITY EVALUATION BASELINE REPORT

**Evaluation Date:** 2026-10-29T13:43:00+05:30
**Evaluator:** ML Infrastructure Engineer & Repository Maintainer
**Git HEAD Commit:** `a3951692b24f69ebfeed5b3fe42a021964728fb0` (`origin/main`)
**Canonical Checkpoint:** `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
**Checkpoint SHA-256:** `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`

---

## 1. Executive Summary & Verification Findings

This baseline formally evaluates the canonical ChakrMicro pre-trained neural checkpoint (Step 10,000, 3.44M parameters) across the 7 required capability dimensions without retraining, fine-tuning, or altering canonical artifacts:

1. **Artifact Cryptographic Integrity:** All 5 governed canonical files match repository governance SHA-256 checksums bit-for-bit.
2. **Deterministic Contracts Passed:** Model loading, weight tying, tokenizer roundtrip encode/decode, and greedy generation determinism verified 100% compliant.
3. **Execution Latency & Throughput:** CPU inference runs with an average prompt response latency of **202.3 ms** (~**158.2 tokens/second**).
4. **Native Language & Instruction Capability:** **0/32 passed (0.0%)**, with an attractor collapse rate of **53.1% (17/32)** under greedy decoding (corrected from 1/32 after eliminating the punctuation-only substring false positive on PROBE-03).
5. **Coding Capability:** **0/12 functional pass (0.0%)**, with Python syntax validity on generated code of **25.0% (3/12)** (corrected from 2/12 after preventing prompt scaffolding execution and requiring genuine executable completions).
6. **Robustness & Edge-Cases:** Bounded outputs and graceful termination on empty strings, single chars, long contexts, and ASCII punctuation. UTF-8 surrogate pairing inputs were safely rejected.

---

## 2. Canonical Artifact Integrity Audit

| Canonical Component | Repository File Path | Expected SHA-256 | Actual Observed SHA-256 | Verification Status |
|---|---|---|---|:---:|
| **Canonical Checkpoint** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **BIT-EXACT MATCH** |
| **BPE Vocabulary** | `data/experiments/vocab_4096/vocab.json` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | **BIT-EXACT MATCH** |
| **BPE Merges** | `data/experiments/vocab_4096/merges.json` | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | **BIT-EXACT MATCH** |
| **Model Core Architecture** | `chakrview/brain/model.py` | `1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361` | `1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361` | **BIT-EXACT MATCH** |
| **Runtime Session** | `chakrview/runtime/session.py` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **BIT-EXACT MATCH** |

---

## 3. Capability Dimensions Evaluation

### 3.1 Loadability & Model Spec
- **Total Parameters:** `3,443,136`
- **Embedding / LM Head Tying:** Confirmed bit-exact shared memory pointer (`embedding.weight.data_ptr() == lm_head.weight.data_ptr()`).
- **Configuration:** `vocab_size=4096`, `d_model=256`, `n_layers=4`, `n_heads=4`, `hidden_dim=1024`, `max_seq_len=256`.
- **Initialization Latency:** 146 ms on CPU.

### 3.2 Tokenizer Correctness
- **Roundtrip Identity:** `sample_text == decode(encode(sample_text))` passed.
- **Special Tokens Contract:** `bos_id=0`, `eos_id=1`, `pad_id=2`.

### 3.3 Generation Determinism
- **Greedy Decoding:** Repeated runs on `"Ram lives in"` produce identical outputs across calls (`"clude the world of the world of the worl"`).

### 3.4 Robustness & Resource Performance
| Case ID | Input Preview | Latency | Tokens Out | Termination / Status |
|---|---|:---:|:---:|---|
| `empty_string` | `""` | 117.7 ms | 32 | `MAX_TOKENS` (Safe) |
| `single_char` | `"a"` | 111.4 ms | 32 | `MAX_TOKENS` (Safe) |
| `whitespace_only` | `"   "` | 121.8 ms | 32 | `MAX_TOKENS` (Safe) |
| `ascii_punctuation` | `"!@#$%^&*()_+-..."` | 145.1 ms | 32 | `MAX_TOKENS` (Safe) |
| `unicode_mix` | `"Hello 😀 world "` | 0.0 ms | 0 | `REJECTED_INPUT` (Handled) |
| `long_repeated_context` | 30 repetitions | 458.1 ms | 32 | `MAX_TOKENS` (Safe) |
| `boundary_numeric` | `"100000000000..."` | 139.8 ms | 32 | `MAX_TOKENS` (Safe) |
| `factual_simple` | `"The capital of France is"` | 204.0 ms | 32 | `MAX_TOKENS` (Safe) |
| `arithmetic_simple` | `"Compute: 7 + 5 ="` | 260.5 ms | 32 | `MAX_TOKENS` (Safe) |
| `code_prefix` | `"def fibonacci(n):"` | 262.2 ms | 32 | `MAX_TOKENS` (Safe) |

- **Average Latency:** 202.3 ms
- **Throughput:** 158.19 tokens/sec

### 3.5 Native Intelligence Benchmark (32 Tasks)
- **Overall Pass Rate:** `0 / 32` (0.0%) [v1.1.0 corrected]
- **Attractor Collapse Rate:** `17 / 32` (53.1%)
- **Analysis:** Demonstrates expected pretraining stage characteristics: strong syntactic phrase generation, but lacking post-training SFT alignment to follow instructional commands without collapsing into dataset high-frequency n-gram attractors (such as `"Step-by-Step Solution:\nThe total..."`). Trivial punctuation-only false positives (`PROBE-03`) have been eliminated with strict three-word sentence grammar criteria.

### 3.6 Coding Capability Benchmark (12 Tasks)
- **Reported Executable Test Pass Rate:** `0 / 12` (0.0%) [v1.3.0 corrected]
- **Syntax Validity Rate:** `3 / 12` (25.0%)
- **Epistemic Clarification on Scoring:** In earlier baseline iterations, `CODE-TEST-01` and `CODE-TEST-02` falsely passed because the prompt provided the function definition and the model produced a comment. Under the corrected test harness, prompt scaffolding is decoupled from generated code, and completions must contain genuine non-comment executable code to be evaluated functionally. Consequently, functional execution pass rate is strictly 0/12, with syntax generation validity at 25.0% (3/12).

---

## 4. Exact Reproduction Commands

```powershell
# 1. Run deterministic test contracts
.\.venv\Scripts\python.exe -m pytest -q tests/test_evaluation_contracts.py

# 2. Run full 7-dimension canonical baseline runner
.\.venv\Scripts\python.exe evaluation/canonical_baseline_runner.py

# 3. Run release smoke validation
.\.venv\Scripts\python.exe scripts/smoke_test_fresh_user.py
```
