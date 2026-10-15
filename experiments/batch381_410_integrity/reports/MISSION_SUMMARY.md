# CHAKRVIEW MASTER MISSION — BATCHES 381–410
## Architecture Integrity, Neural Answering & Generalization Gate Final Report

**Project:** ChakrView  
**Working Branch:** `dev/v0.6-v0.7-validation`  
**Starting Commit:** `7097e6e`  
**Delivered Milestone:** Batches 381–410  
**Regression Suites Status:** 39 / 39 tests passing (100% green across 7 major suites)

---

## 1. Executive Summary & Core Results

Batches 381–410 addressed the central engineering bottlenecks identified in prior milestones:
1. **Resolved Parameter Count Discrepancy (3,443,136 vs 4,229,568):** Diagnosed and mathematically proved that the discrepancy is a weight-tying serialization artifact. The active model has **3,443,136 unique trainable parameters**. PyTorch serialization saves both `embedding.weight` and `lm_head.weight` as distinct dictionary keys, yielding $3,443,136 + (4096 \times 192) = 4,229,568$ serialized elements.
2. **Diagnosed 0% Exact-Match Generation Failure:** Discovered that prior training appended the literal string `"<|im_end|>"` (which decomposes into 8 ASCII/BPE tokens: `[63, 127, 1012, 98, 275, 103, 127, 65]`) with an off-by-one label shift, causing the model to emit corrupted character fragments (`'<imen|_d>'`) rather than true EOS token `1` (`<|endoftext|>`).
3. **Controlled Candidate with Corrected Loss Masking (`candidate_step150_corrected_eos`):** Trained a corrected candidate with exact next-token shift alignment and true EOS token `1`.
   - **Diagnostic Exact Match Rate jumped from 0.0% to 80.0%** across arithmetic, systems architecture, Python coding, and multilingual domains.
   - **True EOS Termination Rate reached 100.0%** (zero pseudo-character noise).
4. **Fresh Generalization & Paraphrase Audit:** Evaluated held-out fresh tasks:
   - Direct format match: 100.0%.
   - Paraphrase robustness: 33.3% (model shows sensitivity to surface prompt templates).
   - Compositional transfer: 50.0% (successfully answered multi-concept architecture queries).
5. **Replay Experiment Audit:** Confirmed that while 0% replay reaches lower immediate validation loss on small task sets, Stage C token rehearsal provides essential regularization against catastrophic forgetting on pretraining domains.

---

## 2. Parameter Count Accounting & Weight Tying Proof

| Component | Tensor Name / Submodule | Tensor Shape | Unique Elements | Serialized Elements in `state_dict` |
| :--- | :--- | :--- | :--- | :--- |
| **Token Embedding** | `embedding.weight` | `[4096, 192]` | 786,432 | 786,432 |
| **Attention Projections (x6)** | `layers.*.attn.{q,k,v,out}_proj.weight` | 24 tensors of `[192, 192]` | 884,736 | 884,736 |
| **FFN Projections (x6)** | `layers.*.ffn.{gate,up,down}_proj.weight`| 18 tensors | 1,769,472 | 1,769,472 |
| **Layer Normalizations** | `layers.*.norm_{1,2}.weight` + `final_norm.weight` | 13 tensors of `[192]` | 2,496 | 2,496 |
| **LM Head (Tied)** | `lm_head.weight` (shares `embedding.weight`) | `[4096, 192]` | **0** (Tied) | **786,432** (Serialized) |
| **TOTAL** | | | **3,443,136** | **4,229,568** |

**Conclusion:** Active trainable parameter count is **3,443,136**. The 4,229,568 figure is exclusively the element sum of the serialized state dictionary.

---

## 3. Checkpoint Lineage & Verified Checksums

| Checkpoint Name | File Path | Parameters | SHA-256 Checksum | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Canonical Step 10,000** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | 3,443,136 | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | Immutable Baseline |
| **Candidate Step 150 (Prior)** | `experiments/batch311_350_neural_generalization/checkpoints/candidate_step150.pt` | 3,443,136 | `b6725ce7a4095fe0c551e0b236256d18b55ebd055a865e2f5150e73036384759` | Batch 311-350 Reference |
| **Candidate Corrected EOS** | `experiments/batch381_410_integrity/checkpoints/candidate_step150_corrected_eos.pt` | 3,443,136 | `27812873dde718485e70825d7be101ede2d1f54d8dd60e3f6a169c51e91a8f9e` | **80% Exact Match / 100% EOS** |

---

## 4. Neural Answering Benchmark Results

Evaluated on `experiments/batch381_410_integrity/data/diagnostic_answer_set.jsonl`:

| Domain | Total Items | Canonical Baseline Exact Match | Candidate Corrected Exact Match | True EOS Termination Rate |
| :--- | :--- | :--- | :--- | :--- |
| **Mathematics** | 4 | 0% (0/4) | **75%** (3/4) | 100.0% |
| **Computer Architecture** | 3 | 0% (0/3) | **100%** (3/3) | 100.0% |
| **Python Coding** | 2 | 0% (0/2) | **100%** (2/2) | 100.0% |
| **Basic Science** | 2 | 0% (0/2) | **100%** (2/2) | 100.0% |
| **Multilingual (Hindi)** | 2 | 0% (0/2) | **100%** (2/2) | 100.0% |
| **Unseen Composition** | 2 | 0% (0/2) | **0%** (0/2) | 100.0% |
| **Overall** | **15** | **0.0%** | **80.0%** | **100.0%** |

### Sample Generated Outputs:
- *Query:* `What is 14 plus 28?` -> *Generated:* `42` (Clean termination on EOS token `1`).
- *Query:* `What does DMA stand for in computer systems?` -> *Generated:* `Direct Memory Access` (Clean EOS).
- *Query:* `भारत की राजधानी क्या है?` -> *Generated:* `नई दिल्ली` (Clean EOS).
- *Query:* `Write a Python function to compute the square of x.` -> *Generated:* `def square(x):\n    return x * x` (Clean EOS).

---

## 5. Fresh Generalization & Paraphrase Audit

Evaluated on held-out tasks in `experiments/batch381_410_integrity/reports/fresh_generalization_results.json`:
- **Direct Match Rate:** 100.0% (3/3).
- **Paraphrase Robustness:** 33.3% (1/3). When questions were paraphrased with complex clauses (e.g. *"Compute the sum of fourteen and twenty eight"* vs *"What is 14 plus 28?"*), the 3.44M parameter model reverted to frequent training phrases.
- **Compositional Transfer:** 50.0% (1/2). Correctly resolved *"What circuit in a CPU combines arithmetic operations with logic gates?"* -> `Arithmetic Logic Unit (ALU)`.

---

## 6. Capability Status Assessment

| Capability Item | Status | Verified Evidence |
| :--- | :--- | :--- |
| **Parameter-count discrepancy resolved** | **VERIFIED** | Weight-tying math proven in `PARAMETER_COUNT_AUDIT.md`. |
| **Checkpoint identification and loading** | **VERIFIED** | All checksums match and load cleanly without shape errors. |
| **Loss masking and EOS contract** | **VERIFIED** | True token `1` replaces pseudo `<|im_end|>` 8-token string. |
| **Neural-only answer generation** | **VERIFIED** | 80% exact match achieved without runtime tools or external LLMs. |
| **Paraphrase robustness** | **PARTIALLY VERIFIED** | 33.3% match; sensitive to unfamiliar prompt phrasing. |
| **Compositional reasoning** | **PARTIALLY VERIFIED** | 50.0% on 2-concept combination; multi-hop arithmetic requires tools. |
| **Zero hallucinations claim** | **REJECTED** | Truthful disclosure: 3.44M model still hallucinates on out-of-distribution queries. |

---

## 7. Full Regression Testing Results

All 39 tests across 7 major suites passed with zero failures:
- `tests/chakrverify/test_multihop_verification.py`: 5 passed
- `tests/chakrverify/test_verification.py`: 6 passed
- `tests/test_batch381_410_integrity.py`: 4 passed
- `tests/test_batch351_380_adaptive_intelligence.py`: 4 passed
- `tests/test_batch281_310_neural_capability.py`: 4 passed
- `tests/test_batch251_260_interactive_runtime.py`: 5 passed
- `tests/test_batch261_270_generalization.py`: 5 passed
- `tests/test_batch271_280_compound_reasoning.py`: 6 passed
- **Total:** **39 passed in ~8.5s (100% green)**

---

## 8. Recommendation for Next Milestone (Batches 411–440)

- **Do Not Promote Checkpoint to Production Release:** Keep `candidate_step150_corrected_eos.pt` as the active research candidate for continual pretraining.
- **Next High-Value Objective:**
  1. Build a synthetic paraphrase augmentation curriculum (generating 5–10 phrasing variants per concept) to lift paraphrase robustness from 33.3% toward >75%.
  2. Implement temperature-scaled sampling with token trie filtering for multi-step reasoning.
  3. Expand ChakrVerify Local to support layout-aware document chunking (tables and section headers).
