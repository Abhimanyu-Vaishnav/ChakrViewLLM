# ChakrView Exp17 — Generator Readiness Audit Report

**Document ID**: `GENERATOR_READINESS_AUDIT`  
**Date**: 2026-10-22  
**Target File**: `experiments/research_v09_conversation_sft/exp17_instruction_sft_v1/generate_exp17_datasets.py`  
**Associated Documents**: `DATASET_SPEC.md`, `EXPERIMENT_DESIGN.md`, `TRAINING_PROTOCOL.md`, `GATE_LEVEL_2_PREPARATION_REPORT.md`  
**Audit Status**: COMPLETE (Static Implementation Audit — Read-Only Mode)  
**Readiness Verdict**: **PREFLIGHT FOUNDATION VERIFIED — EXAMPLE SYNTHESIS LOGIC PENDING IMPLEMENTATION**

---

## 1. Executive Summary

A static implementation audit was performed on `generate_exp17_datasets.py` and its related design artifacts to determine its structural and algorithmic readiness prior to authorizing dataset generation.

### Key Audit Findings:
1. **Implementation Scope**:
   - `generate_exp17_datasets.py` currently contains a **preflight scaffolding and verification harness** (schema validator, leakage detector, whitespace parity checker, tokenizer loader).
   - It does **not** yet contain the actual deterministic synthesis templates/logic for generating the 400 prompt-response pairs.
   - It contains **zero file-writing operations** (no `.write_text()`, no `open(..., 'w')`, no command-line flag parser for `--write-splits`).
2. **Response-Length Discrepancy (5–35 vs. 3–45 tokens)**:
   - `DATASET_SPEC.md` Section 1 lists `5–35 tokens`.
   - `generate_exp17_datasets.py` docstring lines 20–21 and `GATE_LEVEL_2_PREPARATION_REPORT.md` Section 3 list `3–45 tokens (mean: ~12 tokens)`.
   - The discrepancy is reconciled with a clear, unified rule: **Strict Hard Bounds: 3–45 tokens; Primary Target Distribution: 5–35 tokens (mean target: 12 tokens)**.
3. **Safety & Zero-Overwrite Guarantee**:
   - The script references only `experiments/normal_chat_gate_v01/prompts.jsonl` in read-only (`"r"`) mode.
   - It cannot mutate canonical checkpoints, Exp16 datasets, tokenizer files, or `chakrview/runtime/session.py`.
4. **Leakage & Contamination Filter Audit**:
   - The filter iterates through all 20 Gate prompts.
   - It implements exact string matching, Levenshtein ratio $< 0.70$, and a 19-term entity blacklist.
   - *Audit Clarification*: An entity blacklist is a necessary heuristic guardrail, but **does not guarantee semantic independence**. True semantic decoupling requires domain stratification and explicit human review.

---

## 2. Item-by-Item Implementation Audit

### 2.1 Real Example Generation vs. Preflight Logic
- **Current File Contents** ([generate_exp17_datasets.py:1-225](file:///D:/Project/ChakrView/experiments/research_v09_conversation_sft/exp17_instruction_sft_v1/generate_exp17_datasets.py#L1-L225)):
  - Lines 1–36: Module documentation and path setup.
  - Lines 42–60: `FORBIDDEN_GATE_ENTITIES` (19 blacklisted phrases).
  - Lines 66–90: Category definitions (`CATEGORIES`), quotas (`QUOTAS`), and split seeds (`SPLIT_SEEDS`).
  - Lines 98–101: `compute_levenshtein_ratio()` using `difflib.SequenceMatcher`.
  - Lines 103–126: `check_gate_leakage()` checking entity presence, exact string match, and Levenshtein $\ge 0.70$.
  - Lines 133–190: `Exp17DatasetGenerator.build_record()` validating schema, metadata, and token lengths.
  - Lines 192–216: `Exp17DatasetGenerator.verify_serialization_parity()` checking runtime prompt formatting.
  - Lines 219–225: `if __name__ == "__main__":` executing only the verification checks.
- **Finding**: The script is currently a **validator/preflight contract**. The actual domain templates (e.g. generating specific math problems, factual items, context extractions) must be implemented in the generator body before dataset synthesis can occur.

### 2.2 Response-Length Limits Comparison & Reconciliation
- **Document Comparison**:
  - `DATASET_SPEC.md` Section 1: `"Responses are concise (5–35 tokens) to discourage repetitive runaway generation."`
  - `DATASET_SPEC.md` Schema (`metadata.target_token_count`): `"minimum": 1, "maximum": 64`.
  - `generate_exp17_datasets.py` Docstring line 20: `"All responses strictly between 3 and 45 tokens (avg ~12 tokens)..."`
  - `GATE_LEVEL_2_PREPARATION_REPORT.md` Section 3: `"Enforced target response token count between 3 and 45 tokens (mean target: 12 tokens)."`
- **Analysis**:
  - In short single-turn tasks (e.g., Cat 01 factual QA, Cat 06 arithmetic, Cat 08 JSON status), valid concise responses can be as short as 3–4 tokens (e.g., `{"status": "ok"}` = 5 tokens; `42.` = 2–3 tokens; `Mars.` = 2 tokens).
  - Conversely, short explanations (Cat 05) or bulleted lists (Cat 08) naturally reach 30–42 tokens without runaway repetition.
  - Setting a rigid lower bound of 5 tokens would force artificial padding on direct answers (e.g. forcing `The answer is 42.` instead of `42.`), while a hard cap of 35 tokens artificially truncates valid 2-sentence causal explanations.
- **Unified Rule Proposal (No files altered)**:
  - **Hard Validation Range**: `[3, 45]` tokens.
  - **Target Operational Distribution**: Mean $\mu \approx 12$ tokens, 90% of examples within `[5, 35]` tokens.

### 2.3 Category Quotas, Split Sizes, Stable IDs & Seeding
- **Category Quotas** ([generate_exp17_datasets.py:66-82](file:///D:/Project/ChakrView/experiments/research_v09_conversation_sft/exp17_instruction_sft_v1/generate_exp17_datasets.py#L66-L82)):
  - 8 Categories: `factual_qa`, `instruction_following`, `context_extraction`, `reference_resolution`, `short_explanation`, `basic_arithmetic`, `uncertainty_abstention`, `structured_formatting`.
  - Quotas: Train = 40/cat (320 total); Val = 5/cat (40 total); Test = 5/cat (40 total). Total = 400 records.
- **Split Seeds** ([generate_exp17_datasets.py:84-88](file:///D:/Project/ChakrView/experiments/research_v09_conversation_sft/exp17_instruction_sft_v1/generate_exp17_datasets.py#L84-L88)):
  - Train: `17001`
  - Val: `17002`
  - Test: `17003`
- **Output Destination** ([generate_exp17_datasets.py:28-29](file:///D:/Project/ChakrView/experiments/research_v09_conversation_sft/exp17_instruction_sft_v1/generate_exp17_datasets.py#L28-L29)):
  - Configured strictly to `experiments/research_v09_conversation_sft/exp17_instruction_sft_v1/data/`.
  - Parent directory creation is local and isolated; zero cross-directory write paths exist.

### 2.4 Contamination Filter Audit
- **Implementation Inspection** ([generate_exp17_datasets.py:103-126](file:///D:/Project/ChakrView/experiments/research_v09_conversation_sft/exp17_instruction_sft_v1/generate_exp17_datasets.py#L103-L126)):
  1. *Entity Matching*: Iterates through `FORBIDDEN_GATE_ENTITIES` (lines 44–58). Checks `if ent in full_text:`.
  2. *Exact Matching*: Loops over all loaded gate prompts: `if p_lower == gp_text:`.
  3. *Normalized Similarity*: Computes `compute_levenshtein_ratio(p_lower, gp_text)`. Fails if ratio $\ge 0.70$.
- **Audit Evaluation & Limitations**:
  - The script does correctly evaluate against all 20 gate records.
  - *Critical Clarification*: An entity blacklist and string similarity ratio prevent superficial overlap, but **do not guarantee semantic independence**. For instance, an arithmetic prompt asking `What is 18 plus 26?` has low string similarity and different numbers from `What is 17 plus 25?`, but represents an identical cognitive task. This must be acknowledged as a known structural characteristic of instruction fine-tuning, not hidden under claims of complete semantic disjointness.

### 2.5 Path Safety & Overwrite Risk Assessment
- **File System Write Calls**:
  - A comprehensive AST and regex scan revealed **zero `.write_text()`, `.write()`, or `open(..., 'w')` calls** in `generate_exp17_datasets.py`.
  - The script only opens `GATE_PROMPTS_PATH` in read-only mode (`open(..., "r")`).
  - It contains zero paths pointing to canonical checkpoints, Exp16 datasets, tokenizer artifacts, or `chakrview/runtime/session.py`.
- **Finding**: The script is completely non-destructive and cannot modify the environment in its current state.

---

## 3. Protected Artifact Integrity Verification

| Artifact Description | Classification | Expected Reference Hash | Computed Actual Digest | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Architecture Fingerprint** | Parameter Tensor Digest | `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` | `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` | **MATCH** |
| **Canonical Checkpoint** | Checkpoint File | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **MATCH** |
| **Tokenizer Vocab** | Tokenizer File | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | **MATCH** |
| **Tokenizer Merges** | Tokenizer File | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | **MATCH** |
| **Protected Runtime** | Source Code File | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **MATCH** |
| **Model Architecture Source**| Source Code File | `1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361` | `1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361` | **MATCH** |

---

## 4. Execution State & Git Working Tree

- **Non-Writing Test Executed**:
  Executed `python generate_exp17_datasets.py`. Output confirmed:
  `Serialization parity verified: True`
  `Gate v0.1 prompts loaded for leakage checking: 20`
  `Generator initialized in verification-only mode (No files written).`
- **Git Status**:
  `git status --short` confirms zero tracked files modified, zero commits made:
  ```text
  ?? ChakrView_Exp16_v3_1_Master_PreTraining_Audit_Prompt.md
  ?? docs/architecture_audit_2026_10/
  ?? docs/preflight_verification_gate_2026_10/
  ?? experiments/normal_chat_gate_v01/
  ?? experiments/research_v08_json_sft/exp12_constrained_hybrid/
  ?? experiments/research_v09_conversation_sft/
  ```

---

## 5. Exact Next Action & Implementation Plan (Pending Authorization)

To complete the implementation of `generate_exp17_datasets.py` when authorized by the user:
1. **Implement Concrete Domain Generators**:
   Add deterministic generator functions for each of the 8 categories (using domain-specific entity banks for history, science, geography, arithmetic, and extraction contexts).
2. **Apply Unified Token Bounds**:
   Enforce hard bounds `[3, 45]` tokens with target mean $\approx 12$ tokens.
3. **Implement Isolated Split Writing (`--write-splits`)**:
   Add atomic file writers targeting exclusively `experiments/research_v09_conversation_sft/exp17_instruction_sft_v1/data/`.
4. **Post-Generation Integrity Pass**:
   Verify record counts (320/40/40), validate JSON schema, compute split SHA-256 digests, and populate `ARTIFACT_MANIFEST_TEMPLATE.json`.

*Awaiting your explicit authorization before adding domain generation logic or writing dataset files.*
