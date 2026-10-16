# FINAL MISSION REPORT: NEURAL ROUTER FEASIBILITY GATE
## Batches 1661–1700 Comprehensive Engineering & Scientific Audit

### 1. Executive Summary

Batches 1661–1700 investigated the feasibility of using ChakrView's indigenous neural core (ChakrMicro) as a semantic intent router and structured argument extractor across six canonical tool intents: `CALCULATE`, `VERIFY_CLAIM`, `KNOWLEDGE_QUERY`, `CODE_TASK`, `CLARIFY`, and `OUT_OF_SCOPE`.

The mission addressed three core research questions:
1. **Can the 3.4M parameter model learn to output valid structured JSON under conversational paraphrases?**
   **Finding:** No. Without grammar-constrained decoding, the candidate model produces valid JSON on only **14.17%** of held-out inputs.
2. **Does the neural router beat simpler baselines?**
   **Finding:** No. A simple keyword heuristic achieved **57.50% intent accuracy**, while the candidate neural router achieved only **19.17%** (barely above random 16.67%), while being 4,000x slower (259 ms vs 0.06 ms).
3. **Can the neuro-symbolic hybrid maintain safe execution?**
   **Finding:** Yes. The governed neuro-symbolic hybrid achieved a **0.0% false execution rate**, safely catching all invalid or ambiguous neural outputs and delegating them to the fail-closed deterministic router.

---

### 2. Architecture & Checkpoint Provenance Verification

| Artifact Description | File Path | Verified SHA-256 Digest | Status |
| :--- | :--- | :--- | :---: |
| **Canonical Reference Checkpoint** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **Immutable / Preserved** |
| **Experimental Adapted Checkpoint** | `experiments/batch1381_1420/checkpoints/chakrmicro_domain_adapted_experimental.pt` | `15d48176efd6941415cc446b0f0602573a266a49a17da56d72d365d445091139` | **Immutable / Preserved** |
| **New Router Candidate Checkpoint** | `experiments/batch1661_1700/checkpoints/chakrmicro_router_candidate_step100.pt` | `e221d18057ad7b27a2173ddf9cd1c74b7a74ff44e23556195503615734d04a57` | **Newly Created** |
| **Frozen Held-Out Evaluation Dataset** | `experiments/batch1661_1700/data/test_held_out.json` | `6d56d955d953c2bcd343e88a1954f4eaaae63e4363313a8527c2289bd73c64b0` | **Frozen / Sealed** |
| **Controlled Training Corpus** | `experiments/batch1661_1700/data/train.jsonl` | `c34cb16f493b0811abca0df2d1b31d24dc8c06c31683aeb0eb8672b88b15b655` | **Frozen / Sealed** |
| **Validation Dataset** | `experiments/batch1661_1700/data/val.jsonl` | `54202cb6d8fd9145bb25fb3ac201569963fc3f81dd87e9d9fcf4cc7556c791c3` | **Frozen / Sealed** |

#### Resolution of Parameter Discrepancy:
- **Verified Unique In-Memory Parameters:** Exactly **3,443,136** parameters (`d_model=192, n_layers=6, n_heads=6, hidden_dim=512, vocab_size=4096`).
- **State-Dict Serialized Elements:** Exactly **4,229,568** elements due to weight-tying between `embedding.weight` and `lm_head.weight` ($3,443,136 + 786,432$).
- The 1,586,816 figure from Batch 1621 was an accidental table typographical citation; runtime code dynamically loaded the true 3,443,136 parameter model.

---

### 3. Empirical Results Across All Evaluated Systems

Evaluated on the exact same 120 frozen held-out tasks:

| Evaluation System | Intent Accuracy | Structured Argument Match | Macro-F1 (6 classes) | JSON Validity Rate | Latency (Avg CPU) | Unsafe Execution Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Governed Deterministic Router** | 23.33% | 2.50% | 0.1585 | 100.0% | **0.06 ms** | **0.0%** |
| **2. Keyword Heuristic Baseline** | **57.50%** | **50.00%** | **0.5365** | 100.0% | **<0.01 ms** | 30.8% |
| **3. Canonical Neural (Untrained)** | 16.67% | 16.67% | 0.0476 | 0.0% | 188.64 ms | **0.0%** |
| **4. Candidate Neural (Trained)** | 19.17% | 14.17% | 0.1001 | 14.17% | 259.08 ms | **0.0%** |
| **5. Governed Neuro-Symbolic Hybrid** | 23.33% | 2.50% | 0.1585 | 100.0% | 289.17 ms | **0.0%** |

---

### 4. Quality & Safety Gate Verdict

The pre-declared quality gates for promoting the neural router to production were:
- **Intent Accuracy $\ge 70.0\%$:** Actual 19.17% (**FAIL**)
- **JSON Validity Rate $\ge 90.0\%$:** Actual 14.17% (**FAIL**)
- **Unsafe Execution Rate $\le 1.0\%$:** Actual 0.0% (**PASS**)
- **Latency $\le 20.0$ ms:** Actual 259.08 ms (**FAIL**)

**Decision:** The candidate neural router is **NOT eligible for production deployment**. It remains strictly an offline research checkpoint. The governed deterministic router remains the default production routing layer.
