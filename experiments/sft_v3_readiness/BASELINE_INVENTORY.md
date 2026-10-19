# BASELINE INVENTORY & AUDIT CONTEXT
**Directory**: `experiments/sft_v3_readiness/`  
**Date**: October 19, 2026  
**Scope**: SFT V3 Readiness, Data Quality & Native Capability Evaluation  

---

## 1. Architecture & Checkpoint Identity

- **Model Architecture**: `ChakrMicro`
  - Vocabulary: 4096 (BPE with byte fallback)
  - d_model: 192
  - Transformer layers: 6
  - Attention heads: 6
  - FFN dimension: 512
  - Context window: 512 tokens
  - Unique trainable parameters: 3,443,136 (with tied input embedding and LM head weights)
- **Canonical Pretraining Checkpoint**:
  - Path: `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
  - SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
- **Candidate Checkpoint (Batch 3111–3150 JSON SFT V2 Pilot)**:
  - Path: `experiments/batch3111_3150/json_sft_v2_pilot_run/sft_best_checkpoint.pt`
  - SHA-256: `436bdb89e62e0ef461e511b0ac410d01368ce626569b64c27d803ec97d953bd6`
- **Tokenizer Artifacts**:
  - `data/experiments/vocab_4096/vocab.json`: `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a`
  - `data/experiments/vocab_4096/merges.json`: `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7`
  - `release/chakrview-v0.5.0/data/experiments/vocab_4096/vocab.json`: `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a`
  - `release/chakrview-v0.5.0/data/experiments/vocab_4096/merges.json`: `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7`

---

## 2. Dataset Inventory

- **Batch 3111–3150 Dataset Pool**: `experiments/batch3111_3150/JSON_SFT_V2_DATASET.jsonl` (264 records)
  - Train: `experiments/batch3111_3150/train.jsonl` (200 records: 140 structured JSON across 4 core schemas x 35 instances + 60 continuation replay)
  - Validation: `experiments/batch3111_3150/val.jsonl` (30 records: 20 structured JSON across 4 core schemas x 5 instances + 10 continuation replay)
  - Held-Out Test Set (FROZEN / PROTECTED): `experiments/batch3111_3150/test.jsonl` (34 records: 12 familiar schema + 12 unseen schema + 10 replay)
  - Manifest: `experiments/batch3111_3150/DATASET_MANIFEST.json`

---

## 3. Existing Evaluation Entry Points

- `experiments/batch3111_3150/run_evaluation.py`: Unconstrained greedy decoding benchmark.
- `experiments/batch3111_3150/compare_three_way_decoding.py`: 3-way comparative evaluation (Unconstrained, Grammar-Only, Schema-Trie).
- `experiments/batch3111_3150/run_diagnostic_budget_eval.py`: Diagnostic budget scaling (96 tokens).
- Unit Tests:
  - `tests/test_json_constrained_decoder.py`
  - `tests/test_schema_prefix_trie.py`
  - `tests/test_fake_model_schema_decoder.py`
  - `tests/test_schema_decoder_hardening.py`

---

## 4. Current Known Failures & Technical Grounding

1. **Native Unconstrained Decoding**:
   - The candidate model learned the opening delimiter `{"` from SFT V2, but immediately diverts into Python code/comment boilerplate (`{"__main__":\n import doctest`).
   - Valid JSON: 0/20 (0.0%).
2. **Grammar-Only Constrained Decoding**:
   - Pure grammar PDA permits `"__main__"` because `"__main__"` is a valid quoted string. The model loops on valid strings and commas until budget exhaustion.
   - Valid JSON: 0/20 (0.0%).
3. **Schema-Constrained Prefix Trie Decoding**:
   - Eliminates `"__main__"` by restricting keys to active trie properties.
   - Enforcing `strict_spacing=True` and blocking trailing commas when no keys remain elevates completed valid objects to 10/20 (50.0%) on the 48-token budget.
   - **Semantic Value Failure**: 0/20 (0.0%) value match. Model predicts comma-placeholders `", "` or digit repetitions (`: 060`, `: 30070633...`).
   - Proves runtime constraints steer syntax, but cannot inject semantic intelligence into neural weights.

---

## 5. Pre-Existing User Changes to Preserve

The following modified working-tree files must be preserved strictly:
- `chakrview/runtime/assistant_cli.py`
- `chakrview/training/prompt_contract.py`
- `chakrview/training/sft_engine.py`
- `experiments/batch2181_2220/verified_tool_router.py`
- `experiments/batch2221_2260/integrated_assistant_adapter.py`
- `tests/test_conversation_memory.py`
- Untracked files in `experiments/batch*` and `tests/*`
- `chakrview/runtime/session.py` is UNTOUCHED and MUST REMAIN UNTOUCHED.
