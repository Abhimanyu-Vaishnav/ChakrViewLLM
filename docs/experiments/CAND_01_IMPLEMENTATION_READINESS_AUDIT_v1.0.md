# CHAKRVIEW — CAND_01 IMPLEMENTATION READINESS AUDIT v1.0
## Codebase Subsystem Inspection, Capability Alignment, and Gap Register

**Document ID:** `AUD-EXP-CAND01-IMP-001`  
**Location:** `docs/experiments/CAND_01_IMPLEMENTATION_READINESS_AUDIT_v1.0.md`  
**Candidate ID:** `CAND_01_CONTROLLED_ENTITY_COPYING`  
**Parent Governance Standard:** `docs/experiments/EXPERIMENT_GOVERNANCE_PROTOCOL_v1.0.md` Section 14  
**Audit Mode:** STRICT READ-ONLY FORENSIC CODE INSPECTION  
**Status:** `AUDITED_WITH_GAPS_IDENTIFIED`  
**Date:** 2026-10-31  

---

## 1. Audit Scope & Objective

Before authorizing any implementation or execution of `CAND_01_CONTROLLED_ENTITY_COPYING`, an exhaustive forensic audit of the existing codebase was conducted to determine:
1. Whether existing training, collation, and loss subsystems support the candidate's operational requirements.
2. What software gaps must be resolved prior to candidate execution.
3. Whether resolving those gaps requires modifying protected files (`model.py`, `session.py`, checkpoint binaries) or can be executed strictly via self-contained experiment scripts.

---

## 2. Subsystem-by-Subsystem Audit Findings

### 2.1 Model Architecture (`chakrview/brain/model.py`)
- **Status:** **SUPPORTED (100% READY)**
- **Verification:**
  - File SHA-256: `1343d7d18224cc30e9dcf47ebfbb9ee5d7c8052bf89a31a982bb90033fa282d8` (Verified unaltered).
  - Forward pass accepts `input_ids [B, T]` and returns `logits [B, T, V]`.
  - Attention mechanism natively computes causal attention masks with RoPE.
  - No changes required. Modifying `model.py` is strictly forbidden.

### 2.2 SFT Engine & Loss Masking (`chakrview/training/sft_engine.py`)
- **Status:** **PARTIALLY SUPPORTED (GAP IDENTIFIED)**
- **Inspection:**
  - `sft_engine.py` supports `mask_prompt=True` and sets target sequence tokens to `-100` for all prompt tokens.
  - However, in `sft_engine.py`, the target sequence masks *only* prompt tokens; if a response contains template words (as in Arm C or standard conversational formats), those response template words are *not* masked.
  - For CAND_01 Arm A (minimal syntax where prompt is `PAIR: K=V. QUERY: K ->` and response is purely `[VAL]`), existing `mask_prompt=True` logic successfully masks all prompt tokens and computes loss exclusively on `[VAL] + [EOS]`.
  - **Gap:** `sft_engine.py` does not currently log token-level target rank or causal sensitivity delta ($\Delta p$) during evaluation intervals.

### 2.3 Loss Computation Modules (`chakrview/training/weighted_loss.py`, `loss.py`)
- **Status:** **SUPPORTED (100% READY)**
- **Inspection:**
  - `WeightedCausalLoss` in `chakrview/training/weighted_loss.py` supports arbitrary per-token loss weighting and `ignore_index=-100`.
  - `CausalLoss` in `chakrview/training/loss.py` supports standard causal cross-entropy.
  - Both modules are fully functional for target-only masking.

### 2.4 Tokenizer & Vocabulary (`chakrview/tokenizer/`, `data/experiments/vocab_4096`)
- **Status:** **SUPPORTED WITH CONSTRAINTS (GAP IDENTIFIED)**
- **Inspection:**
  - `data/experiments/vocab_4096/vocab.json` contains 4,093 active tokens.
  - ASCII characters, digits, and punctuation (including `:`, `=`, `-`, `>`, spaces) are present in the single-byte range [3, 258].
  - **Gap:** In the local environment, importing `chakrview/__init__.py` attempts to load runtime and cognitive modules that import optional or missing packages (e.g. `torch` in root environment), causing namespace import failure if loaded via package root instead of direct path or isolated virtual environment.

### 2.5 Dataset Collation (`chakrview/training/collator.py`)
- **Status:** **SUPPORTED (100% READY)**
- **Inspection:**
  - `CausalLanguageModelingCollator` handles `[B, T]` stacking with strict length validation (`max_context=512`) and zero silent truncation.

---

## 3. Formal Gap Register

The following gaps must be addressed prior to execution. In accordance with ChakrView governance, all gaps will be resolved using self-contained scripts under `scripts/` or `experiments/cand01_controlled_entity_copying/` without modifying core source files.

| Gap ID | Subsystem | Description of Missing Capability | Severity | Proposed Engineering Resolution |
|---|---|---|---|---|
| **GAP-01** | **Dataset Generation** | No generator exists to synthesize single-token disjoint key-value datasets across the 8 task families. | **BLOCKER** | Implement `scripts/generate_cand01_dataset.py` with strict seed 42, single-token validation, and zero cross-split leakage. |
| **GAP-02** | **Evaluator Harness** | No benchmark runner exists to evaluate exact first-token extraction, logit rank, and $\Delta p$ sensitivity. | **BLOCKER** | Implement `scripts/evaluate_cand01_benchmark.py` adhering to `CAND_01_EVALUATOR_SPECIFICATION_v1.0.md`. |
| **GAP-03** | **Execution Runner** | Need a dedicated non-interactive training script to execute the 4-arm factorial matrix under reproducible seeds. | **BLOCKER** | Implement `scripts/run_cand01_experiment.py` orchestrating the 4 arms sequentially with atomic checkpoint saving. |
| **GAP-04** | **Telemetry Logging** | Need automated JSON logging of per-step gradient norms, attention entropy, and validation accuracy. | **HIGH** | Integrate telemetry exporter into `run_cand01_experiment.py`. |
| **GAP-05** | **Environment Isolation** | Python environment execution requires verifying PyTorch availability in active subshell. | **HIGH** | Verify virtualenv execution path in Pre-Execution GO checklist. |

---

## 4. Audit Conclusion & Readiness Rating

- **Readiness Rating:** **`READY_WITH_GAPS_DOCUMENTED`**
- **Action Plan:**
  The architecture and training foundations are fully capable of executing CAND_01 without modifying any protected repository files. Execution remains blocked until Gap 01 through Gap 05 are implemented and verified in the subsequent pre-execution phase.

