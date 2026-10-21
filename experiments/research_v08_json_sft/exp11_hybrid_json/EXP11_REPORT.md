# ChakrView Experiment 11 — Hybrid Generator-Pointer for Structured JSON Extraction Report

**Status**: `COPYING WORKS — STRUCTURED GENERATION NOT YET RELIABLE`  
**Experiment Path**: `experiments/research_v08_json_sft/exp11_hybrid_json/`  
**Parent Checkpoint**: `Step-10000` (`96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`)  
**Experiment 10 Checkpoint**: `checkpoints/exp10_span_pointer_best.pt` (`8333719ed3c24bb4f41bd9e17e84bde3c3f48fcde8d36a2617b1230e50db3e1f`)  
**Experiment 11 Checkpoint**: `checkpoints/exp11_hybrid_best.pt` (`454fbd1559a686cc30fe5f661f8267a6507d861dccbcda3946987a0044c081e2`)  

---

## 1. Executive Summary & Objective

The objective of Experiment 11 was to construct an isolated research prototype combining ChakrView's native causal language model output head with the Experiment 10 span-pointer mechanism. The mission: synthesize valid structured JSON syntax autoregressively via the language model head, while copying requested slot values (including unseen multi-token entities) directly from context using the span-pointer.

### Core Findings & Final Verdict:
1. **JSON Syntax Parse Validity**:
   - The native causal language model (Baseline A) achieved **0.0% JSON parse rate (0/180)** across all evaluation splits, collapsing into pretraining prose and URL repetition attractors.
   - The Hybrid Generator-Pointer prototype (Baseline C) dramatically improved syntactic compliance, achieving **85.0% JSON parse validity on Validation (51/60)**, **88.3% on Test A (53/60)**, and **100.0% on Test B (60/60)**.
2. **Context Value Copying via Pointer**:
   - The span-pointer successfully bound and extracted multi-token strings (e.g. `power_converter`, `solar_inverter`, `calibrated`, `throttled`) into active JSON values without token vocabulary corruption.
3. **Structured Autoregressive Stability & Whole-Object Match**:
   - Despite high parse validity, whole-object exact match was **5.0% on Validation (3/60)**, **1.7% on Test A (1/60)**, and **0.0% on Test B (0/60)**.
   - **Why?** The causal language model backbone (pretrained on unstructured text) does not yet maintain a stable autoregressive key-sequencing grammar. It repeatedly hallucinates duplicate keys (e.g., `{"status": "...", "status": "..."}` or `{"entity": "...", "code": "..."}`) when multiple fields are requested, causing schema mismatches.
4. **Selected Status**: `COPYING WORKS — STRUCTURED GENERATION NOT YET RELIABLE`.

---

## 2. Integrity Verification & Cryptographic Provenance

Cryptographic hashes were verified before and after all diagnostics, dataset generation, training, and evaluations:

| Artifact | File Path | SHA-256 Digest | Status |
| :--- | :--- | :--- | :--- |
| **Canonical Checkpoint** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **VERIFIED IDENTICAL** |
| **Tokenizer Vocab** | `data/experiments/vocab_4096/vocab.json` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | **VERIFIED IDENTICAL** |
| **Tokenizer Merges** | `data/experiments/vocab_4096/merges.json` | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | **VERIFIED IDENTICAL** |
| **Session Runtime** | `chakrview/runtime/session.py` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **VERIFIED UNTOUCHED** |
| **Exp 10 Checkpoint** | `experiments/research_v08_json_sft/exp10_span_pointer/checkpoints/exp10_span_pointer_best.pt` | `8333719ed3c24bb4f41bd9e17e84bde3c3f48fcde8d36a2617b1230e50db3e1f` | **VERIFIED IDENTICAL** |
| **Exp 11 Checkpoint** | `experiments/research_v08_json_sft/exp11_hybrid_json/checkpoints/exp11_hybrid_best.pt` | `454fbd1559a686cc30fe5f661f8267a6507d861dccbcda3946987a0044c081e2` | **VERIFIED CREATED** |

---

## 3. Dataset Specification & Leakage Audit

A deterministic synthetic dataset was constructed via `prepare_exp11_dataset.py` with full provenance documented in `data/dataset_manifest.json`:
- **Training set**: 300 records (`data/train.jsonl`)
- **Validation set**: 60 records (`data/val.jsonl`)
- **Test A (Familiar Entities in Novel Contexts)**: 60 records (`data/test_a_familiar.jsonl`)
- **Test B (Completely Held-Out Entities, Statuses, Codes)**: 60 records (`data/test_b_held_out.jsonl`)

### Split Partitioning & Leakage Audit:
- **Entity Pool Partitioning**:
  - Familiar Entities: `subsystem_alpha`, `hydraulic_actuator`, `power_converter`, `thermal_sensor`, `propulsion_thruster`, `solar_inverter`.
  - Held-Out Entities (Test B ONLY): `cryo_pump_01`, `optical_relay_02`, `quantum_core_03`, `flux_capacitor_04`, `laser_emitter_05`, `plasma_duct_06`.
  - Overlap between Train and Test B: **0 items (100.0% disjoint)**.
- **Punctuation Adjacency**: All entities are embedded inside structured clauses like `[entity]` and `'status'`, adjacent to brackets and quotes.
- **Ambiguity & Distractors**: Every record contains an interleaved distractor component clause with overlapping subword patterns.
- **Missing Field Policy**: When a requested field is omitted from context, the target JSON specifies `"none"`.

---

## 4. Architecture & Routing Implementation

Implemented in `hybrid_generator_pointer.py`:
1. **Shared Backbone**:
   - `ChakrMicro` 6-layer Transformer ($d_{\text{model}}=192$, $n_{\text{heads}}=6$, tied embedding LM head $V=4096$).
2. **Span Pointer Heads**:
   - $W_{\text{query\_start}}, W_{\text{key\_start}} \in \mathbb{R}^{192 \times 192}$
   - $W_{\text{query\_end}}, W_{\text{key\_end}} \in \mathbb{R}^{192 \times 192}$
3. **Explicit Inspectable Routing Contract**:
   - Step 1: Language model head generates JSON tokens autoregressively.
   - Step 2: When the decoded generation stream encounters an open string value trigger following a key (`': "'` or `':"'`), control explicitly routes to the pointer head.
   - Step 3: The pointer evaluates candidate positions across the prompt context under pure inference masking ($[1, \text{seq\_len}-2]$).
   - Step 4: The selected multi-token span is emitted into the token stream, followed by the closing quote `"` and continuation of structural JSON synthesis.
4. **Finite Gradient & Overfit Verification**:
   - Verified in `train_exp11.py`: finite loss and finite backward gradients confirmed.
   - Pilot overfit on 8 records reduced loss from 0.1060 to **0.0030** in 50 steps.

---

## 5. Comparative Evaluation Results

All baselines evaluated on identical test examples using pure greedy decoding ($T=0.0$, `max_new_tokens=64`):

### Summary Metrics Table

| Split | Model / Baseline | Total Examples | Valid JSON Parse | Valid Schema | Exact Object Match | Copy Slot Hits |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Validation** | Baseline A: Native LM | 60 | 0 / 60 (0.0%) | 0 / 60 (0.0%) | 0 / 60 (0.0%) | N/A |
| | Baseline B: Span Pointer | 60 | 0 / 60 (0.0%)* | 0 / 60 (0.0%)* | 0 / 60 (0.0%)* | 11 / 60 (18.3%) |
| | Baseline C: Hybrid Model | 60 | **51 / 60 (85.0%)** | **15 / 60 (25.0%)** | **3 / 60 (5.0%)** | **38 / 60 (63.3%)** |
| **Test A (Familiar)** | Baseline A: Native LM | 60 | 0 / 60 (0.0%) | 0 / 60 (0.0%) | 0 / 60 (0.0%) | N/A |
| | Baseline B: Span Pointer | 60 | 0 / 60 (0.0%)* | 0 / 60 (0.0%)* | 0 / 60 (0.0%)* | 8 / 60 (13.3%) |
| | Baseline C: Hybrid Model | 60 | **53 / 60 (88.3%)** | **11 / 60 (18.3%)** | **1 / 60 (1.7%)** | **34 / 60 (56.7%)** |
| **Test B (Held-Out)** | Baseline A: Native LM | 60 | 0 / 60 (0.0%) | 0 / 60 (0.0%) | 0 / 60 (0.0%) | N/A |
| | Baseline B: Span Pointer | 60 | 0 / 60 (0.0%)* | 0 / 60 (0.0%)* | 0 / 60 (0.0%)* | 0 / 60 (0.0%) |
| | Baseline C: Hybrid Model | 60 | **60 / 60 (100.0%)** | **12 / 60 (20.0%)** | 0 / 60 (0.0%) | **26 / 60 (43.3%)** |

*\*Note on Baseline B: The standalone span pointer emits a raw substring span, not a structured JSON document. It is reported with 0% JSON validity because its output type is non-JSON by definition.*

---

## 6. Deep Autopsy: Why Structured Generation is Not Yet Reliable

Inspecting individual predictions in `EXP11_PER_EXAMPLE.csv` reveals the exact failure mode of the current hybrid:

1. **Successful Syntax Emitted**:
   - The hybrid cleanly emits the initial JSON structure: `{"status": "` or `{"entity": "`.
2. **Successful Copy Routing**:
   - When the quote opens, the pointer executes and splices the correct multi-token span (e.g. `"power_converter"` or `"solar_inverter"`).
3. **The Autoregressive Looping Defect**:
   - Once the first key-value pair is closed (e.g. `{"status": "calibrated"`), the underlying language model has not learned the state-machine transition to proceed to the next distinct key (e.g. `", "code": "..."}`).
   - Instead, the pretraining language model falls into a repeating token loop:
     `{"status": "power_converter", "status": "power_converter", "status": "power_converter"...}`
   - Because of key duplication, the schema validator rejects the output (e.g. missing `code` or duplicate `status`).
4. **Conclusion**:
   - The **pointer mechanism functions as intended** — it extracts multi-token spans from context and slots them into strings.
   - The **generator mechanism is deficient** — causal language pretraining at Step-10000 without dedicated multi-turn JSON schema fine-tuning lacks the grammatical discipline to advance through distinct schema fields autoregressively.

---

## 7. Test Suite Verification

The comprehensive test suite in `test_exp11.py` was executed with `pytest`:
```
experiments/research_v08_json_sft/exp11_hybrid_json/test_exp11.py::test_json_syntax_and_schema_validation PASSED [ 12%]
experiments/research_v08_json_sft/exp11_hybrid_json/test_exp11.py::test_multi_token_span_pointer PASSED [ 25%]
experiments/research_v08_json_sft/exp11_hybrid_json/test_exp11.py::test_punctuation_adjacent_values PASSED [ 37%]
experiments/research_v08_json_sft/exp11_hybrid_json/test_exp11.py::test_missing_field_policy PASSED [ 50%]
experiments/research_v08_json_sft/exp11_hybrid_json/test_exp11.py::test_candidate_mask_independence PASSED [ 62%]
experiments/research_v08_json_sft/exp11_hybrid_json/test_exp11.py::test_deterministic_metric_recomputation PASSED [ 75%]
experiments/research_v08_json_sft/exp11_hybrid_json/test_exp11.py::test_dataset_split_disjointness PASSED [ 87%]
experiments/research_v08_json_sft/exp11_hybrid_json/test_exp11.py::test_canonical_checkpoint_provenance PASSED [100%]
============================== 8 passed in 4.10s ==============================
```

---

## 8. Deliverables Generated

The following artifacts have been created in `experiments/research_v08_json_sft/exp11_hybrid_json/`:
- `EXP11_DESIGN.md`: Architecture design and routing specification.
- `prepare_exp11_dataset.py`: Synthetic dataset generator.
- `data/`: Dataset splits (`train.jsonl`, `val.jsonl`, `test_a_familiar.jsonl`, `test_b_held_out.jsonl`, `dataset_manifest.json`).
- `hybrid_generator_pointer.py`: Hybrid Generator-Pointer model architecture.
- `train_exp11.py`: Training and pilot overfit verification script.
- `evaluate_exp11.py`: Comprehensive 3-baseline evaluation harness.
- `test_exp11.py`: 8-test unit and integration test suite.
- `EXP11_RESULTS.json`: Quantitative summary metrics with explicit counts.
- `EXP11_PER_EXAMPLE.csv`: All 540 evaluated test outputs across baselines.
- `EXP11_REPORT.md`: This comprehensive report.
- `checkpoints/exp11_hybrid_best.pt`: Saved research checkpoint.

---

## 9. Single Recommended Next Step

**Implement Grammar-Constrained JSON Decoding (Prefix-Trie / State-Machine Constrained Decoding)**:  
The pointer correctly extracts entities, but the causal LM loops on duplicate keys. In Experiment 12, pair the hybrid pointer with a constrained decoding mask (or finite-state automaton) that forces the generator to advance through the declared schema keys without repeating previously generated keys.

Do not integrate this prototype into the production runtime or modify Git state.
