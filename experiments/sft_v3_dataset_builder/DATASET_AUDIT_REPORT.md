# CHAKRVIEW SFT V3 DATASET AUDIT REPORT
**Milestone:** SFT V3 Grounded Dataset Builder & Pre-Flight Validation  
**Date:** 2026-10-19  
**Execution Environment:** Isolated `experiments/sft_v3_dataset_builder/`  
**Safety Gate:** Engineering & Validation Only (ZERO Training Authorized)

---

## 1. Executive Verdict

### **VERDICT: DATASET PREFLIGHT PASSED — READY FOR HUMAN REVIEW**

The SFT V3 dataset preparation pipeline has been constructed from scratch, achieving complete mathematical and factual grounding across all structured examples, verified template-family disjointness, zero leakage, byte-for-byte reproducibility, and 100% preflight pass rate across 200 training records and 40 validation records.

**Crucial Clarification:** Passing preflight establishes that the data is technically and structurally ready for human review. It does **NOT** authorize training or guarantee that the model will achieve full semantic competence on the first attempt.

---

## 2. Key Audit Findings & Grounded Innovations

### 1. Elimination of "Contextless Hallucination Forcing"
In SFT V2, models were commanded to emit arbitrary synthetic tuples (e.g. `{"status": "queued", "code": 202}`) without any factual source passage, forcing the model to memorize arbitrary literals per template index.
- In SFT V3, **100% of structured records (170/170 across train and val) contain explicit, factual `context` passages**.
- Every field value (`status`, `metric`, `feature`, `count`, `code`, `enabled`) is verifiable from the context.
- Evaluation on the candidate checkpoint confirms why this is necessary: candidate outputs on novel validation context passages show that the untrained model cannot guess unseen values natively, but the data now provides the exact source information needed for learning causal extraction.

### 2. Multi-Level Anti-Leakage & Template-Family Disjointness
- **Template Isolation**: `train_templates` and `val_templates` share **0 template family IDs**. Instructions in training (e.g., *"Extract the server status and code as a JSON object with keys '{k1}' and '{k2}'.\nResponse:"*) never appear in validation (e.g., *"From the report above, construct a JSON response with keys '{k1}' and '{k2}'.\nResponse:"*).
- **Entity & Value Isolation**: All hostnames, metrics, feature flags, and counts are disjoint between train and val.
- **Replay Document Isolation**: Replay continuation examples are sourced from disjoint pretraining document files (`corpus_english_cs` and `stage_b_systems` for train; `english_systems` for val), guaranteeing zero text overlap.
- **Held-out Test Split Protection**: `test.jsonl` was **NEVER accessed, opened, or read** at any stage.

### 3. Anti-Code Attractor Conditioning & Abstention Capabilities
- **Category 5 (`anti_code_attractor_guard`)**: Addresses the 85% code-attractor failure mode by pairing explicit negative instruction directives (*"Do NOT write Python code, doctests, or import modules..."*) with clean, positive JSON targets.
- **Category 6 (`missing_information_abstention`)**: Trains the model to avoid fabricating answers when requested facts are absent from the context, emitting explicit `null` fields and `{"error": "missing_information"}`.

---

## 3. Dataset Composition and Counts

| Dataset Split | Total Records | Structured Extraction | Replay Continuation | SHA-256 Checksum |
| :--- | :--- | :--- | :--- | :--- |
| **Train Split** | **200** | 140 (70.0%) | 60 (30.0%) | `57affaa613eaf0e31be752bc680dddcd3c748e1cee3752942db93e8dea270804` |
| **Validation Split** | **40** | 30 (75.0%) | 10 (25.0%) | `4c4c77d5db4884e06ed4e06550bd4badba6a4793f37a1d5a38d0c9512cbc1cf8` |

### Detailed Category Distribution:

| Category ID & Name | Train Count | Val Count | Data Role & Target Schema |
| :--- | :--- | :--- | :--- |
| `grounded_extraction_server_status` | 25 | 5 | Extract status (`str`) and HTTP code (`int`) from server reports |
| `grounded_telemetry_metric` | 25 | 5 | Extract metric name (`str`) and measurement (`float`) |
| `grounded_feature_flag` | 25 | 5 | Extract feature (`str`) and operational state (`bool`) |
| `grounded_inventory_item` | 25 | 5 | Extract item (`str`) and count (`int`, no leading zeros) |
| `anti_code_attractor_guard` | 20 | 5 | Suppress Python doctest boilerplate via prompt conditioning |
| `missing_information_abstention` | 20 | 5 | Emit explicit `null` and `error` when context omits requested data |
| `base_continuation_replay` | 60 | 10 | Unconditioned continuation replay to prevent catastrophic forgetting |

---

## 4. Preflight Validation & Leakage Audit Results

The independent command-line validator ([validate_sft_v3_dataset.py](file:///d:/Project/ChakrView/experiments/sft_v3_dataset_builder/validate_sft_v3_dataset.py)) verified all records against strict gates:

| Preflight Quality Gate | Requirement / Metric | Measured Result | Audit Verdict |
| :--- | :--- | :--- | :--- |
| **JSON Syntax Validity** | 100% parseable by RFC 8259 | 170/170 structured records | **PASS** |
| **Ground Truth Alignment** | Target matches `expected_answer` exactly | 170/170 records | **PASS** |
| **Required Schema Keys** | All declared required keys present | 100% | **PASS** |
| **Unknown Key Rejection** | Zero undeclared keys in response | 0 unknown keys | **PASS** |
| **Type Integrity** | Correct string, integer, float, bool, null | 100% match | **PASS** |
| **Leading Zero Prohibition** | No integers formatted as `060` | 0 violations | **PASS** |
| **Context Length Safety** | Sequence length <= 512 tokens | Max: **157 tokens**, Min: 13 tokens | **PASS** |
| **Truncation Risk** | Records truncated by context window | **0 records** (0.0%) | **PASS** |
| **Template-Family Overlap** | Zero shared prompt templates | **0 overlapping templates** | **PASS** |
| **Prompt String Overlap** | Zero exact prompt string collisions | **0 collisions** | **PASS** |
| **Factual Target Overlap** | Zero shared factual JSON targets | **0 collisions** | **PASS** |
| **Context Passage Overlap** | Zero shared context narratives | **0 collisions** | **PASS** |
| **Builder Reproducibility** | Byte-for-byte identical output on fixed seed | Verified bit-identical | **PASS** |

---

## 5. Non-Training Candidate Evaluation on SFT V3 Validation Split

We evaluated the candidate checkpoint (`sft_best_checkpoint.pt`) against the 40-record SFT V3 validation split using [evaluate_sft_v3_validation.py](file:///d:/Project/ChakrView/experiments/sft_v3_dataset_builder/evaluate_sft_v3_validation.py) to measure baseline capability prior to any SFT V3 training:

| Capability Metric | Native Generation (Layer A) | Schema-Constrained Decoding (Layer B) |
| :--- | :--- | :--- |
| **Valid JSON Rate** | **0.0%** (0/30) | **76.7%** (23/30) |
| **Exact Task Match (EM)** | **0.0%** (0/30) | **0.0%** (0/30) |
| **Decoder Deadlocks** | N/A | **0 / 30** (0.0% deadlock rate) |
| **Code-Attractor Rate** | **83.3%** (25/30) | **0.0%** (Masked by prefix-trie) |

### Key Diagnostic Observations:
1. **Confirmation of the Code Attractor**: In unconstrained generation, the model overwhelmingly generated Python boilerplate (`{"__main__":\n import doctest`) when encountering `\nResponse:`.
2. **Schema Trie Efficacy**: Under the hardened Schema Prefix Trie, valid JSON syntax reached **76.7%** with **0 deadlocks**.
3. **Need for Learned SFT V3 Weights**: Exact Task Match remained at 0% across both native and constrained layers because the model has never been trained on context extraction. This proves that runtime inference constraints alone cannot substitute for genuine neural learning.

---

## 6. Artifact & Safety Verification

All safety gates were adhered to strictly:

| Protected Artifact / File | Canonical Path | Verified Checksum / Status |
| :--- | :--- | :--- |
| **Canonical Checkpoint (Step 10,000)** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` |
| **Canonical Tokenizer Vocab** | `data/experiments/vocab_4096/vocab.json` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` |
| **Canonical Tokenizer Merges** | `data/experiments/vocab_4096/merges.json` | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` |
| **Runtime Session** | `chakrview/runtime/session.py` | **ZERO DIFF (Untouched)** |
| **Held-out Test Split** | `experiments/batch3111_3150/test.jsonl` | **NEVER ACCESSED** |
| **Training Steps** | Model parameters | **ZERO LEARNING STEPS** |

---

## 7. Recommended Next Step

**Milestone: Authorized SFT V3 Training Run & Checkpoint Evaluation**  
With the dataset fully constructed, validated, and frozen under `DATASET_MANIFEST.json`, the operator may consider authorizing a controlled CPU training pilot (e.g. 100 or 150 steps) using `train.jsonl` to establish whether the model learns native context extraction and anti-attractor compliance.
