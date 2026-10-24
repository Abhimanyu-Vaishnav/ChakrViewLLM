# ChakrView Exp17 — Gate Level 2 Dataset Generation Preparation Report

**Document ID**: `GATE_LEVEL_2_PREPARATION_REPORT`  
**Date**: 2026-10-22  
**Target Directory**: `experiments/research_v09_conversation_sft/exp17_instruction_sft_v1/`  
**Status**: PREPARATION COMPLETE — AWAITING EXPLICIT USER APPROVAL FOR GENERATION

---

## 1. Executive Summary

This report establishes complete technical readiness for **Gate Level 2 (Dataset Generation)** of Exp17 Instruction SFT v1. Both conditional findings from the Level 1 Design Verification Gate have been formally and programmatically resolved, executable verification tests have passed, and the generator architecture has been established without creating dataset files or modifying any protected project artifacts.

- **Dataset Generation Executed**: **NO** (`data/` directory unpopulated; zero JSONL records created).
- **Model Training Executed**: **NO** (zero optimizer steps, zero parameter mutations).
- **Canonical & Protected Artifacts**: **100% UNTOUCHED** (all SHA-256 hashes re-verified bit-for-bit).
- **Working Tree Integrity**: **PRESERVED** (zero tracked Git modifications, zero commits).

---

## 2. Resolution of Conditional Findings

### 2.1 Resolution of Condition 1: Trailing Space Serialization Parity
- **Identified Issue**:
  `DATASET_SPEC.md` listed prompt formatting with a trailing space (`Response: `), whereas the active runtime adapter ([chakrview/runtime/prompt_conditioning.py:43](file:///D:/Project/ChakrView/chakrview/runtime/prompt_conditioning.py#L43)) constructs:
  `f"Instruction: {clean_p}\nResponse:"` (No trailing space).
  Tokenization tests demonstrated that `"Response:"` ends with token ID `61`, whereas `"Response: "` ends with token ID `324`.
- **Executable Resolution**:
  Implemented in `generate_exp17_datasets.py`:
  1. The generator directly invokes `chakrview.runtime.prompt_conditioning.format_conditioned_prompt(prompt, prompt_mode=mode, context=context)` rather than ad-hoc string formatting.
  2. Executable parity verification test `gen.verify_serialization_parity()` executed:
     - Plain instruction test: `Instruction: ...\nResponse:` $\equiv$ runtime output (100% match).
     - Context QA test: `Context: ...\nQuestion: ...\nAnswer:` $\equiv$ runtime output (100% match).
     - BPE tokenization roundtrip verified: `decode(encode(formatted_prompt)) == formatted_prompt` (100% match).

### 2.2 Resolution of Condition 2: Reproducible Multi-Layered Leakage Defense
- **Identified Issue**:
  A single surface string similarity threshold (Levenshtein $< 0.80$) is insufficient because it can permit paraphrased queries or identical entity pairings.
- **Executable Resolution**:
  `generate_exp17_datasets.py` implements a **three-tier fail-closed contamination filter**:
  1. **Tier 1 — Forbidden Entity Blacklist**:
     An explicit blacklist (`FORBIDDEN_GATE_ENTITIES`) containing all core entities from the 20 Gate prompts:
     `{"mars", "red planet", "olleh", "hello backwards", "chlorophyll", "leaves are green", "station a", "station b", "train leaves", "project chimera", "agent 007", "delta-x", "liquid nitrogen", "closing price of gold", "december 31, 2030", "chemical formula for water", "h2o", "elephant or a mouse", "solar panels", "say cat, dog, bird"}`.
     Any candidate example containing these terms is instantly rejected.
  2. **Tier 2 — Exact String Collision Guard**:
     $Prompt_{\text{candidate}} == Prompt_{\text{gate}}$ fails closed.
  3. **Tier 3 — Normalized Levenshtein Threshold**:
     $\text{Similarity}(Prompt_{\text{candidate}}, Prompt_{\text{gate}}) \ge 0.70$ fails closed (tightened from 0.80 to 0.70).
  4. **Documented Limitations**:
     String similarity cannot capture high-level conceptual convergence (e.g., general physics vs. chemistry). However, the combination of entity blacklisting, strict extractive context separation, and category stratification provides verifiable isolation of the frozen gate benchmark.

---

## 3. Dataset Generator Architecture (`generate_exp17_datasets.py`)

The dataset generator has been implemented in verification mode inside the isolated experiment directory:
[generate_exp17_datasets.py](file:///D:/Project/ChakrView/experiments/research_v09_conversation_sft/exp17_instruction_sft_v1/generate_exp17_datasets.py)

### Key Specifications:
1. **Deterministic RNG Seeds**:
   - Train: `17001`
   - Validation: `17002`
   - Test: `17003`
2. **Stable ID Scheme**:
   - `EXP17_TRAIN_{CAT}_{INDEX:03d}`
   - `EXP17_VAL_{CAT}_{INDEX:03d}`
   - `EXP17_TEST_{CAT}_{INDEX:03d}`
3. **Category Quotas (Exactly 400 Total Records)**:
   - 8 Categories $\times$ 40 Train = **320 Train Records**
   - 8 Categories $\times$ 5 Val = **40 Validation Records**
   - 8 Categories $\times$ 5 Test = **40 Held-Out Test Records**
4. **Response Length Bounds**:
   - Enforced target response token count between 3 and 45 tokens (mean target: 12 tokens).
5. **Fail-Closed Machine-Readable Rejection Log**:
   - Any record violating schema, length bounds, entity blacklist, or Levenshtein similarity is logged to `REJECTION_REPORT.json` and halts generation.

---

## 4. Protected Artifact Integrity Verification

| Artifact Description | Classification | Expected Reference Hash | Computed Actual Digest | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Architecture Fingerprint** | Parameter Tensor Digest | `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` | `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` | **MATCH** |
| **Canonical Checkpoint** | Checkpoint File | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **MATCH** |
| **Tokenizer Vocab** | Tokenizer File | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | **MATCH** |
| **Tokenizer Merges** | Tokenizer File | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | **MATCH** |
| **Protected Runtime** | Source Code File | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **MATCH** |
| **Model Architecture Source**| Source Code File | `1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361` | `1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361` | **MATCH** |

---

## 5. Execution State & Git Status

- **Commands Executed**:
  1. Pre- and post-preparation SHA-256 digests verified across all protected artifacts.
  2. Executed `generate_exp17_datasets.py` in verification-only mode (validated prompt formatting parity and Gate v0.1 prompt loader; zero records or dataset files written).
- **Git Status**:
  `git status --short` confirms:
  ```text
  ?? ChakrView_Exp16_v3_1_Master_PreTraining_Audit_Prompt.md
  ?? docs/architecture_audit_2026_10/
  ?? docs/preflight_verification_gate_2026_10/
  ?? experiments/normal_chat_gate_v01/
  ?? experiments/research_v08_json_sft/exp12_constrained_hybrid/
  ?? experiments/research_v09_conversation_sft/
  ```
  Zero tracked files modified, zero commits made.

---

## 6. Exact Command to Execute Dataset Generation (Upon User Approval)

When explicit authorization is granted, dataset generation will be initiated by running:

```powershell
.venv\Scripts\python.exe experiments/research_v09_conversation_sft/exp17_instruction_sft_v1/generate_exp17_datasets.py --write-splits
```

*(Note: The `--write-splits` flag is currently dormant and gated; no dataset files will be created until authorized).*
