# CHAKRVIEW — CAND_01 IMPLEMENTATION READINESS AUDIT v1.3 (REVIEWED)
## Codebase Subsystem Inspection, Component Capability Alignment, and Gap Register

**Document ID:** `AUD-EXP-CAND01-IMP-004`  
**Location:** `docs/experiments/CAND_01_IMPLEMENTATION_READINESS_AUDIT_v1.3_REVIEWED.md`  
**Supersedes:** `docs/experiments/CAND_01_IMPLEMENTATION_READINESS_AUDIT_v1.2_REVIEWED.md`, `docs/experiments/CAND_01_IMPLEMENTATION_READINESS_AUDIT_v1.1_REVIEWED.md`, `docs/experiments/CAND_01_IMPLEMENTATION_READINESS_AUDIT_v1.0.md`  
**Candidate ID:** `CAND_01_CONTROLLED_ENTITY_COPYING`  
**Parent Governance Standard:** `docs/experiments/EXPERIMENT_GOVERNANCE_PROTOCOL_v1.0.md` Section 14  
**Audit Mode:** STRICT READ-ONLY FORENSIC CODE INSPECTION  
**Status:** `AUDITED_WITH_GAPS_IDENTIFIED`  
**Execution Authorization:** **`UNAPPROVED`**  
**Date:** 2026-10-31  

---

## 1. Revision History & Audit Mandate (v1.3 Targeted Redesign)

This document represents the formal v1.3 scientific revision of `CAND_01_IMPLEMENTATION_READINESS_AUDIT_v1.2_REVIEWED.md`. It incorporates specific findings from the independent adversarial verification:
- **Clarification on GAP-07 (Pre-Flight Calibration Scope):** Extended the calibration duration to 60 steps to ensure coverage across the warmup peak (step 50) and specified logging of unclipped global gradient $L_2$ norm.
- **Clarification on GAP-05 (Telemetry Exporter):** Maintained that telemetry logging will be embedded directly within `run_cand01_experiment.py`, avoiding external library dependencies.
- **Pre-Execution Gating Integrity:** Maintained clear demarcation: complete documentation does not imply implementation readiness; execution remains blocked until Gaps 01 through 07 are implemented, tested, and audited.

---

## 2. Component-by-Component Implementation Readiness Matrix

| Subsystem / Component | Exact File Path / Function | Current Status | Verification Evidence & Needed Actions | Blocking Execution? |
|---|---|---|---|:---:|
| **Model Forward Pass** | `chakrview/brain/model.py` (`ChakrMicro.forward`) | `VERIFIED` | SHA-256: `1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361`. Native causal masking, RoPE embeddings, tied weights verified. | **NO** |
| **Model Configuration** | `chakrview/brain/config.py` (`ModelConfig`) | `VERIFIED` | 6L, 192D, 6H, max_context=512, vocab_size=4096. Matches canonical checkpoint architecture. | **NO** |
| **Batch Collation** | `chakrview/training/collator.py` (`CausalLanguageModelingCollator`) | `VERIFIED` | Bounds checking ($T \le 512$), shape validation, zero silent truncation verified. | **NO** |
| **Weighted Target Loss** | `chakrview/training/weighted_loss.py` (`WeightedCausalLoss`) | `VERIFIED` | Token-level weighting, `ignore_index=-100` masking supported and verified. | **NO** |
| **SFT Engine Core** | `chakrview/training/sft_engine.py` (`SFTEngine`) | `PARTIALLY AVAILABLE` | `mask_prompt=True` masks prompt tokens. However, lack of logit rank telemetry and evaluation $\Delta p$ measurement during training loops requires standalone runner. | **NO** (Bypassed via runner) |
| **Tokenizer Artifacts** | `data/experiments/vocab_4096/vocab.json`, `merges.json` | `VERIFIED` | 4,093 active tokens, single-byte tokens present in range [3, 258]. | **NO** |
| **Python Environment** | Active Python 3.14 Subshell | `REQUIRES VALIDATION` | Module import of `chakrview` requires dedicated virtual environment with PyTorch pre-installed. | **YES** |
| **Dataset Generator** | `scripts/generate_cand01_dataset.py` | `MISSING` | Code does not exist. Must synthesize single-token 4-partition key-value datasets with seed 42. | **YES** |
| **Evaluator Harness** | `scripts/evaluate_cand01_benchmark.py` | `MISSING` | Code does not exist. Must implement coupled $A_{\text{FT}}$ / $A_{\text{FC}}$, two-way clustered variance, and logit rank. | **YES** |
| **Evaluator Unit Tests** | `tests/test_cand01_evaluator.py` | `MISSING` | Code does not exist. Must verify negative controls and edge cases prior to benchmark runs. | **YES** |
| **Experiment Runner** | `scripts/run_cand01_experiment.py` | `MISSING` | Code does not exist. Must orchestrate the 4 factorial arms sequentially across seeds `[42, 137, 2026]` with embedded telemetry. | **YES** |
| **Pre-Flight Calibration** | 60-step gradient telemetry check | `MISSING` | Must execute on disposable scratch model copy to validate unclipped gradient norm stability under target-only loss. | **YES** |

---

## 3. Reassessed Gap Register & Action Items

| Gap ID | Subsystem | Description | Genuinely Required? | Planned Engineering Resolution |
|---|---|---|:---:|---|
| **GAP-01** | Dataset Generation | No generator exists to synthesize single-token 4-partition key-value datasets across the 8 task families. | **YES (CRITICAL BLOCKER)** | Implement `scripts/generate_cand01_dataset.py` with seed 42, percentile stratification, and zero cross-split leakage. |
| **GAP-02** | Evaluator Harness | No benchmark runner exists to evaluate coupled $A_{\text{FT}}$ / $A_{\text{FC}}$, two-way clustered variance, and $\Delta p$ sensitivity. | **YES (CRITICAL BLOCKER)** | Implement `scripts/evaluate_cand01_benchmark.py` adhering to `CAND_01_EVALUATOR_SPECIFICATION_v1.3_REVIEWED.md`. |
| **GAP-03** | Evaluator Unit Tests | No automated tests exist to verify negative controls, sequence termination gating, and error taxonomy parsing. | **YES (HIGH PREREQUISITE)** | Implement `tests/test_cand01_evaluator.py` validating scoring logic against synthetic edge cases. |
| **GAP-04** | Execution Runner | Need a dedicated non-interactive training script to execute the 4-arm factorial matrix across seeds `[42, 137, 2026]`. | **YES (CRITICAL BLOCKER)** | Implement `scripts/run_cand01_experiment.py` orchestrating the 4 arms sequentially with atomic checkpoint saving. |
| **GAP-05** | Telemetry Logging | Need automated step-level JSON telemetry of unclipped gradient norms, loss stability, and attention entropy. | **YES (SCRIPT-EMBEDDED)** | Embed telemetry exporter directly into `run_cand01_experiment.py` (no standalone package required). |
| **GAP-06** | Virtual Environment | Active shell must resolve PyTorch dependencies and enforce deterministic CPU thread execution. | **YES (OPERATIONAL BLOCKER)** | Verify virtual environment path and set `torch.set_num_threads(1)` in runner. |
| **GAP-07** | Pre-Flight Calibration | Separately authorized 60-step gradient telemetry check to validate $LR = 10^{-4}$ under target-only loss. | **YES (SAFETY GATE)** | Execute on disposable model copy prior to full confirmatory launch. |

---

## 4. Audit Conclusion & Readiness Rating

- **Documentation Readiness:** **`READY_FOR_PRE_EXECUTION_REVIEW`** (All v1.3 specifications, cardinality tables, and audit resolutions complete).
- **Implementation Readiness:** **`NOT_READY`** (Blocked by Gaps 01 through 07).
- **Execution Authorization Status:** **`UNAPPROVED`**
- **Action Plan:**
  The baseline architecture and core training collators are fully verified. Execution remains strictly prohibited until Gaps 01 through 07 are implemented, audited, and cleared in accordance with Experiment Governance Protocol v1.0 Section 14.

