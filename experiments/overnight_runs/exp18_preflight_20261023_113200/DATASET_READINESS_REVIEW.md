# Exp18 Dataset Feasibility & Readiness Review

**Experiment**: `exp18_scaled_instruction_format_v1`  
**Location**: `experiments/overnight_runs/exp18_preflight_20261023_113200/DATASET_READINESS_REVIEW.md`  
**Date**: 2026-10-23  
**Review Type**: Preflight Specification vs Infrastructure Audit  

---

## 1. Current State: Specification vs Real Data

- **Target Record Count**: 1,520 records (Train: 1,200 | Val: 160 | Test: 160).
- **Physical Data Files Status**: **DO NOT EXIST**.
  - In strict compliance with the Preflight Boundary, no official dataset files have been generated, serialized, or staged.
  - The Exp18 data specifications in [`DATASET_SPEC.md`](file:///d:/Project/ChakrView/experiments/research_v09_conversation_sft/exp18_scaled_instruction_format_v1/DATASET_SPEC.md) currently exist solely as a frozen architectural design.
- **Generator Script Status**:
  - The repository contains [`generate_exp17_datasets.py`](file:///d:/Project/ChakrView/experiments/research_v09_conversation_sft/exp17_instruction_sft_v1/generate_exp17_datasets.py) and [`candidate_banks.py`](file:///d:/Project/ChakrView/experiments/research_v09_conversation_sft/exp17_instruction_sft_v1/candidate_banks.py), which synthesize 400 records (320/40/40).
  - An expanded generator script for Exp18 (`generate_exp18_datasets.py`) must be authored in a future authorized generation step to scale from 400 to 1,520 records while purging synthetic token artifacts (`Apollo_train_0`).

---

## 2. Infrastructure Readiness & Verification Plan

When dataset generation is authorized, the future generation script and preflight pipeline must implement the following deterministic validation mechanisms:

### 2.1 Category Balance & Field Completeness
- Verify exact count of 150 training, 20 validation, and 20 test records across all eight categories:
  - `basic_arithmetic` (150/20/20)
  - `structured_formatting` (150/20/20)
  - `context_extraction` (150/20/20)
  - `reference_resolution` (150/20/20)
  - `instruction_following` (150/20/20)
  - `factual_qa` (150/20/20)
  - `uncertainty_abstention` (150/20/20)
  - `short_explanation` (150/20/20)
- Enforce required keys: `id`, `split`, `category`, `prompt_mode`, `prompt`, `context`, `formatted_prompt`, `response`, `expected_target`, `metadata`.

### 2.2 Token-Length Bounding
- Tokenize every formatted prompt and target response using the canonical BPE tokenizer (`vocab_4096`).
- Assert response lengths fall within 5 to 64 tokens.
- Assert total sequence length ($L_{\text{prompt}} + L_{\text{response}} + 2$) is $\le 512$ tokens (`ModelConfig.max_seq_len`).

### 2.3 Split Isolation & Leakage Prevention
- **Prompt Exact Deduplication**: Assert zero duplicate prompts across all 1,520 items.
- **Cross-Split Entity & N-Gram Disjointness**:
  - Build set of 5-grams from Train prompts; assert 0% 5-gram overlap with Test prompts.
  - Assert disjoint operand pairs in arithmetic (operands in Test must not appear in Train).
  - Assert disjoint schema keys and payload values in structured JSON formatting.

### 2.4 Ground Truth & Scorer Validation
- **Arithmetic**: Verify that every response matches an independent Python AST calculation.
- **Structured JSON**: Verify that every target response parses cleanly via `json.loads` (RFC 8259) and satisfies expected key-type schemas.
- **Uncertainty Abstention**:
  - Proof-of-absence: Confirm target attribute is absent from context for unanswerable items.
  - Proof-of-grounding: Confirm target attribute is present in context and response for control items.
- **Context Extraction & Reference Resolution**: Confirm exact substring containment within the context.

### 2.5 Reproducible Seed & Manifest Locking
- Fixed random seed (`42`).
- Upon generation, compute SHA-256 hashes for `train_exp18_v1.jsonl`, `val_exp18_v1.jsonl`, and `test_exp18_v1.jsonl`.
- Record hashes and record counts in `ARTIFACT_MANIFEST.json`.
