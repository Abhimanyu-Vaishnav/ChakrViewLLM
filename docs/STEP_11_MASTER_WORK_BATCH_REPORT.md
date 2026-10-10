# Master Work Batch 11 — Benchmark Integrity, Checkpoint Identity Audit & Quality Evaluation v2 Report

**Repository:** `D:\Project\ChakrView`  
**Execution Context:** Step 11 Neural Capability Evaluation & Metric Hardening Milestone  
**Baseline Commit:** `ea12cb24c7f0db4c6a6f11f4228c2e6467382d51`  
**Branch:** `main` (Ahead of origin/main: 34)  
**Safety & Governance Status:** `TRAINING READINESS: NOT READY FOR TRAINING` ($\Delta W = 0$)

---

## 1. Executive Summary

Master Work Batch 11 built directly upon the committed Batch 10 neural benchmark framework to resolve a critical checkpoint identity ambiguity, audit metric calculation correctness, expand the benchmark suite with objective tasks, and execute rigorous comparative capability experiments across frozen model states.

### Key Deliverables & Findings:
1. **Resolution of the Stage D Weight Hash Discrepancy:**
   - Recomputed the loaded model weight hash for the frozen Stage D 500-step pilot checkpoint directly from its state dictionary using the canonical byte-digesting algorithm (`InferenceEngine.compute_weight_hash`).
   - The full, true loaded weight hash is **`5f3712d80940fb9345d01b5434e8bb64936d01e4d3aaff096ab5268c282d506c`**, which exactly matches the internal `model_hash` recorded in the checkpoint dictionary (`data['model_hash']`).
   - The discrepancy noted in Batch 10's Markdown summary (`38b4ca...55db1`) was a documentation-level truncation error caused by accidental transposition from another artifact; the actual JSON result file and frozen weights always matched `5f3712d8...506c`.
   - The distinction between **checkpoint file SHA-256** (`8ab1ec36...cc1a449`) and **in-memory parameter byte hash** (`5f3712d8...506c`) has been formally codified and validated with automated regression tests.
2. **Metric Correctness & Benchmark Integrity Audit:**
   - Confirmed that token counting operates strictly on emitted tokens (`tokens_emitted_ids`), preventing prompt prefix tokens from polluting generated sequence lengths.
   - Diversity and repetition metrics (unique-token ratio, 2-gram, 3-gram, 4-gram repetition ratios) operate exclusively on generated suffix tokens.
   - Identified and fixed metric reporting subtleties around EOS termination and empty-string outputs (e.g. single-token EOS generations like token `1` decoding to empty string `""` but consuming 1 actual emitted token).
3. **Capability Suite Expansion (Benchmark v2.0):**
   - Designed and published `evaluation/benchmark_prompts_v2.0.json`, containing 30 prompts across 6 distinct domains with verified provenance, expected answers, deterministic evaluation keys, and human-review rubrics.
   - Added objective deterministic evaluation to the harness for arithmetic, factual extraction, logical deductions, and code syntax.
4. **Controlled Inference Experiments (Short 32-Token vs. Long 96-Token):**
   - Conducted 6 controlled benchmark runs across 3 frozen states (Random Baseline Seed 42, Stage C 9k Checkpoint, and Stage D 500 Checkpoint).
   - Revealed clear empirical behavioral profiles:
     - **Random Baseline:** Immediate degenerate collapse into punctuation and character loops (0.010 unique token ratio, 0.989 4-gram repetition).
     - **Stage C (Unsupervised 9k):** Fluent language generation and structural diversity, but suffers from open-loop topical repetition and failure to terminate naturally on question answering.
     - **Stage D (QA 500-Step Pilot):** Severe mode collapse and dataset overfitting; memorized specific arithmetic scratchpad strings (`The answer is 200.`, `So he has 200*2=<<2*2=200>>12`), generating this pattern regardless of prompt domain or semantics. Achieved 0.0% accuracy on objective tasks.
5. **Protected Artifact Verification & Safety:**
   - All protected file hashes (Canonical model weights, Stage C checkpoint file, Stage D checkpoint file, QA pilot manifest) verified bit-exact ($\Delta W = 0$).

---

## 2. Checkpoint Identity Audit & Discrepancy Resolution

### 2.1 Hashing Methodology Distinction
In ChakrView, two distinct SHA-256 digests exist for neural checkpoints:
1. **Checkpoint File SHA-256:**
   - Computed by streaming the binary bytes of the saved `.pt` file on disk:
     $$\text{SHA-256}(\text{bytes of } \texttt{checkpoint\_0000500.pt}) = \texttt{8ab1ec36b5c5a09929b8510332f18fecc0183eb0e9332ab99375995a8cc1a449}$$
2. **Loaded Model Weight Hash (Parameter Byte Digest):**
   - Computed by `InferenceEngine.compute_weight_hash()`. It iterates over `sorted(model.named_parameters())`, feeding parameter name strings and contiguous float32 parameter numpy byte buffers into a SHA-256 hasher:
     $$\text{SHA-256}\left(\sum_{p \in \text{sorted}(P)} \text{encode}(name_p) + \text{tobytes}(p)\right) = \texttt{5f3712d80940fb9345d01b5434e8bb64936d01e4d3aaff096ab5268c282d506c}$$

### 2.2 Recomputed Fingerprints & Identity Audit
We independently verified all model configurations and checkpoints:

| Model / Checkpoint | File Path | Checkpoint File SHA-256 | Loaded Model Weight Hash | Stored Metadata Hash | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Random Baseline (Seed 42)** | `None` (in-memory) | N/A | `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` | N/A | **VERIFIED CANONICAL** |
| **Stage C 9,000-Step** | `checkpoints/.../checkpoint_0009000.pt` | `3e03ae0a134da40439038d965278f4da2996b1ac2e8f959d6059859600b0bb23` | `36711d25831740840dc9d62d60087a898aa4a19746322ef3cb0e04974a2770e3` | `None` | **VERIFIED FROZEN** |
| **Stage D 500-Step Pilot** | `checkpoints/.../checkpoint_0000500.pt` | `8ab1ec36b5c5a09929b8510332f18fecc0183eb0e9332ab99375995a8cc1a449` | `5f3712d80940fb9345d01b5434e8bb64936d01e4d3aaff096ab5268c282d506c` | `5f3712d80940...506c` | **VERIFIED EXACT MATCH** |

### 2.3 Explanation of Historical Batch 10 Discrepancy
In Batch 10, the generated JSON file `evaluation/results/stage_d_500_benchmark.json` and Markdown report `stage_d_500_benchmark.md` correctly recorded:
`"model_weight_hash": "5f3712d80940fb9345d01b5434e8bb64936d01e4d3aaff096ab5268c282d506c"`
However, in `docs/STEP_10_MASTER_WORK_BATCH_REPORT.md` (Table line 56), the abbreviated entry was accidentally written as `38b4ca...55db1`. This string matches a substring from an external dataset manifest entry (`55db1` appears in `stage_c_manifest.json`), indicating a documentation transcription artifact during Markdown table compilation.
**Conclusion:** The Stage D checkpoint weights are intact, identical to their saved training state, and match `5f3712d8...506c` bit-for-bit. Identity is established with zero ambiguity.

---

## 3. Benchmark Metric Audit & Methodology Hardening

The benchmark harness `evaluation/neural_benchmark_harness.py` was inspected and upgraded to v2.0:

1. **Emitted Token Counting:**
   - The token loop measures chunks directly from `runtime.stream_generate()`. Every emitted token ID is logged in `tokens_emitted_ids`. Prompt tokens are not included.
2. **Timing Boundaries & TTFT:**
   - `start_time` is taken immediately before entering the autoregressive stream loop.
   - `ttft_time` records elapsed time upon arrival of the very first token chunk (`step_index = 0`), capturing prompt prefill latency plus initial token sampling.
   - Total latency captures wall-clock time from loop start to generator exhaustion.
   - TPS is computed as $\text{actual\_generated\_tokens} / \text{total\_duration\_seconds}$.
3. **Warm-up Run Isolation:**
   - 2 initial generation passes (`max_new_tokens=16`) are executed before benchmark recording to prime PyTorch thread scheduling and CPU cache structures. Outputs and latencies from warm-up passes are completely discarded.
4. **Diversity & Repetition Metrics:**
   - Unique token ratio: $|\text{set}(T_{\text{gen}})| / |T_{\text{gen}}|$.
   - $N$-gram repetition ratio: $\sum_{ng \in NG, \text{count}(ng) > 1} (\text{count}(ng) - 1) / |NG|$, strictly evaluated over $T_{\text{gen}}$.
5. **Deterministic Task Evaluation:**
   - Added `evaluate_deterministic_match()`, supporting `deterministic_substring` and `deterministic_syntax_check` against verified answer keys.
6. **Determinism Verification:**
   - Repeated runs of identical model configurations under greedy decoding ($T = 0.0$) yield identical token ID sequences across all prompts.

---

## 4. Benchmark Suite v2.0 Architecture

The suite `evaluation/benchmark_prompts_v2.0.json` expands coverage from 10 to 30 versioned prompts across 6 domains:

| Domain | Count | Provenance | Evaluation Type | Expected Capability |
| :--- | :---: | :--- | :--- | :--- |
| **Passage Factual Recall** | 5 | Indus Valley, Project Tiger, ISRO missions, Biology, Architecture | Deterministic substring | Extract explicit factual entities from provided text passages |
| **Passage Summarization / Explanation** | 5 | Operating systems, Compilers, Distributed systems, Caches | Human-review rubric | Explain structural trade-offs based on provided passages |
| **Arithmetic Logic** | 5 | Addition, subtraction, multiplication, division, word problem | Deterministic exact / substring | Objective mathematical evaluation ($12+15=27$, $50-18=32$, etc.) |
| **Algorithmic Reasoning** | 5 | Syllogisms, arithmetic sequences, boolean logic, LIFO, string reversal | Deterministic substring | Basic logical deduction and sequence completion |
| **Safe Coding Structure** | 5 | Python function signatures, C declarations, SQL queries | Deterministic syntax check | Basic programming syntax and idiomatic declarations |
| **Open Continuation** | 5 | Virtual memory, pipelining, distributed systems, sovereign AI | Human-review rubric | Coherent conceptual continuation without degeneration |

---

## 5. Controlled Experimental Results

Six comprehensive benchmarks were executed on CPU (2 threads, greedy sampling $T = 0.0$):
- **Short Generation:** `max_new_tokens = 32`
- **Long Generation:** `max_new_tokens = 96`

### 5.1 Comparative Metrics Summary Table

| Experiment Run | Total Tokens | Mean Latency (ms) | Mean TTFT (ms) | Throughput (tok/s) | Mean Unique Ratio | Mean 4-Gram Repetition | Terminations (EOS / MaxTok) | Deterministic Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Baseline Short (32 tok)** | 960 | 149.7 | 20.6 | 223.8 | 0.031 | 0.966 | 0 / 30 | 0/20 (0.0%) |
| **Stage C 9k Short (32 tok)** | 906 | 132.0 | 20.4 | 230.9 | 0.428 | 0.356 | 3 / 27 | 0/20 (0.0%) |
| **Stage D 500 Short (32 tok)** | 878 | 137.4 | 21.5 | 213.1 | 0.546 | 0.160 | 4 / 26 | 0/20 (0.0%) |
| **Random Baseline Long (96 tok)** | 2,880 | 377.8 | 20.2 | 257.4 | 0.010 | 0.989 | 0 / 30 | 0/20 (0.0%) |
| **Stage C 9k Long (96 tok)** | 2,634 | 439.7 | 24.0 | 196.8 | 0.229 | 0.628 | 3 / 27 | 0/20 (0.0%) |
| **Stage D 500 Long (96 tok)** | 2,358 | 340.1 | 21.7 | 231.0 | 0.345 | 0.463 | 9 / 21 | 0/20 (0.0%) |

---

## 6. Qualitative Findings & Semantic Analysis

### 6.1 Canonical Random Initialization Baseline
- **Behavior:** Complete collapse into ungrounded repetitive character loops (`????????????????`, `................`).
- **Diversity:** Unique token ratio drops to 0.010 on long horizon; 4-gram repetition exceeds 98.9%.
- **Objective Accuracy:** 0.0% (0/20).

### 6.2 Stage C 9k (Unsupervised Real-Corpus Pretraining)
- **Behavior:** Possesses real English lexical tokens and grammar, but lacks instruction awareness or passage grounding. When prompted with passage questions or instructions, it treats them as web text and generates Wikipedia-style headers (`\n\nReferences`, `\n\nThe Great of the Great...`).
- **Horizon Degradation:** On short horizon, unique token ratio is 0.428. Over longer horizons (96 tokens), it falls into repeating n-gram attractors (4-gram repetition increases from 0.356 to 0.628).
- **EOS Behavior:** Terminated naturally on 3/30 prompts in both short and long horizons (e.g. producing short section breaks followed by EOS).
- **Objective Accuracy:** 0.0% (0/20). Lacks factual extraction and numerical reasoning.

### 6.3 Stage D 500 (Fine-Tuning Pilot)
- **Behavior: Severe Mode Collapse into Training Templates:**
  - The pilot training run (trained on GSM8k-style arithmetic problem pairs) completely overfitted on specific scratchpad strings.
  - Across factual recall, summarization, logic, and coding prompts, it repeatedly emitted:
    - `" The answer is 200."`
    - `"So he has 200*2=<<2*2=200>>12 years old."`
    - `"There are 50*2=<<200*2=60>>400"`
- **Superficial Diversity Fallacy:**
  - Although Stage D shows higher unique token ratios (0.546 short, 0.345 long) and lower 4-gram repetition than Stage C, this is purely because the memorized scratchpad template (`<<2*2=200>>12 years old`) contains varied alphanumeric symbols. It does **not** indicate reasoning or comprehension.
- **EOS Triggering:**
  - Stage D terminates early (9 EOS terminations on long horizon) specifically when it completes the memorized string `" The answer is 200."` followed by token `1` (`<EOS>`).
- **Objective Accuracy:** 0.0% (0/20). Failed every arithmetic problem (outputting canned `200` regardless of inputs like `12 + 15` or `8 * 7`).

---

## 7. Protected Artifact Verification

All critical hashes were verified bit-exact via automated SHA-256 digests:

| Artifact Description | Location | Expected SHA-256 Digest | Actual SHA-256 Digest | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Canonical Model Weights** | In-Memory (Seed 42) | `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` | `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` | **BIT-EXACT MATCH** |
| **Stage C Checkpoint** | `checkpoint_0009000.pt` | `3e03ae0a134da40439038d965278f4da2996b1ac2e8f959d6059859600b0bb23` | `3e03ae0a134da40439038d965278f4da2996b1ac2e8f959d6059859600b0bb23` | **BIT-EXACT MATCH** |
| **Stage D Checkpoint** | `checkpoint_0000500.pt` | `8ab1ec36b5c5a09929b8510332f18fecc0183eb0e9332ab99375995a8cc1a449` | `8ab1ec36b5c5a09929b8510332f18fecc0183eb0e9332ab99375995a8cc1a449` | **BIT-EXACT MATCH** |
| **QA Pilot Manifest** | `qa_pilot_manifest.json` | `b7ebf80cd273e9cbbb9f60311ffeda34bda720a46d5f71ae8f8f6a067fb90604` | `b7ebf80cd273e9cbbb9f60311ffeda34bda720a46d5f71ae8f8f6a067fb90604` | **BIT-EXACT MATCH** |

Zero weights were mutated ($\Delta W = 0$). No training, re-training, or fine-tuning occurred.

---

## 8. Test Execution Evidence

All test suites were executed directly against the local environment:

1. **`tests/test_neural_benchmark_harness.py`**:
   - `test_01_synthetic_metric_calculations`: PASSED
   - `test_02_harness_fails_closed_on_missing_checkpoint`: PASSED
   - `test_03_random_baseline_benchmark_execution`: PASSED
   - `test_04_save_benchmark_results_serialization`: PASSED
   - `test_05_prompt_suite_v2_schema_and_domain_coverage`: PASSED
   - `test_06_checkpoint_identity_and_hash_resolution`: PASSED
   - `test_07_deterministic_match_evaluation`: PASSED
   - **Result:** `7 passed in 2.95s` (Exit Code: 0).
2. **`tests/test_qa_review_tooling.py` & `tests/test_step103_consolidation.py`**:
   - Clean checkout availability, unreviewed queue invariant, split isolation, candidate preservation, compile fail-closed: ALL PASSED.
   - **Result:** `12 passed in 2.96s` (Exit Code: 0).

---

## 9. Recommended Next Neural Research Experiment

### Identification of Primary Failure Mode:
The most consequential finding of Batch 11 is **catastrophic mode collapse resulting from narrow, unregularized fine-tuning**. Stage D was exposed to a small set of arithmetic examples without general language retention loss. As a result, the model lost Stage C's diverse grammatical representations and collapsed into emitting a single arithmetic template across all prompts.

### Recommendation:
**Experiment:** *Replay-Regularized Multi-Task Pilot with Stage C Retention.*
- **Why this experiment is superior to alternatives:**
  - *Alternative A (Hyperparameter tuning / Decoding configuration):* Changing temperature or top-p sampling on Stage D cannot fix fundamental weight collapse; it only scrambles the tokens of an already collapsed distribution.
  - *Alternative B (Larger dataset without replay):* Adding more QA pairs without anchor loss will simply overfit on a larger set of templates while still wiping out base pretraining representations.
  - *Recommended Approach:* Design an experimental training objective combining reviewed QA candidates (once founder-authorized) with a 20% replay mixture of Stage C general language text, using an explicit Kullback-Leibler (KL) divergence penalty against the frozen Stage C model weights. This directly addresses the observed catastrophic forgetting and prevents template degeneration.
- **Prerequisite Gate:**
  - Training readiness remains **NOT READY FOR TRAINING**. No training execution will occur until the Founder Review Gate approves the Stage D QA dataset.

---

## 10. Multi-Dimensional Readiness Assessment

1. **Neural Capability Assessment: LOW (Pre-Reasoning Baseline)**
   - The model has not developed genuine logical or numerical reasoning. Stage C demonstrates English lexical structure; Stage D demonstrates template overfitting.
2. **Runtime Performance Assessment: EXCELLENT**
   - Autoregressive forward passes achieve $210 - 257\text{ tok/s}$ sustained on 2 CPU threads with $\sim 20\text{ ms}$ TTFT and $\Delta W = 0$ verification.
3. **Engineering Reliability Assessment: EXCELLENT**
   - Checkpoint identity verified; metric definitions audited; prompt suite versioned and deterministic; regression tests comprehensive.
