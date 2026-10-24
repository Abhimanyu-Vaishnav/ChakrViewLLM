# ALIGNMENT GATE RESULTS & TRAINING DECISION

**Audit Directory**: `experiments/overnight_runs/alignment_gate_20261024_111800/`  
**Evaluation Date**: 2026-10-24  
**Gate Status**: **`PASS`**  
**Training Execution Decision**: **`READY_TO_PROPOSE`** *(Execution strictly deferred per safety rules pending explicit user authorization)*

---

## 1. Gate Criteria Verification

| Alignment Gate Criteria | Requirement | Observed Empirical Result | Verdict |
| :--- | :--- | :--- | :---: |
| **1. Prompt Conditioning Parity** | `SFTDataset` training prompt tokens must match inference prompt tokens bit-for-bit. | Verified in `tests/test_sft_prompt_alignment.py`. Training prompt slice exactly matches inference prompt slice. | **PASS** |
| **2. Loss Masking Invariance** | Target sequence must mask 100% of prompt tokens with `-100`. | Verified. Active target tokens begin strictly at response token 0 through EOS (token 1). | **PASS** |
| **3. Evaluator Substring Disambiguation** | Evaluator must not match short targets (e.g. 'W') as substrings of arbitrary words. | Word boundary matching `\bW\b` implemented in `task_evaluator.py` and regression tested. | **PASS** |
| **4. Unit & Regression Tests** | All focused, evaluator, and repository-wide tests must pass. | 3/3 prompt alignment tests passed. 17/17 evaluator tests passed. 126/126 discover tests passed. | **PASS** |
| **5. Protected Artifact Immutability** | All 8 protected assets must match authoritative SHA-256 hashes before and after. | Recomputed in `integrity_before.json` and `integrity_after.json`. 100% bitwise match across all 8 assets. | **PASS** |
| **6. No Official File Overwrite** | No official Exp18 JSONL files or canonical checkpoints modified. | Confirmed. Official dataset directories and canonical checkpoint untouched. | **PASS** |

---

## 2. Minimal Diff Summary

### Modified Tracked/Existing Files:
1. **`chakrview/training/sft_engine.py`**:
   - Added import of `format_conditioned_prompt`.
   - Updated `SFTDataset.get_batch` (lines 193-210) to format the prompt using `rec["formatted_prompt"]` or `format_conditioned_prompt(rec["prompt"], mode, context)` prior to encoding.
2. **`chakrview/evaluation/task_evaluator.py`**:
   - Updated `_score_factual_qa` (lines 670-680) to use regex word-boundary matching `\b{ent}\b` for entities of length $\le 2$.

### Added Test Files:
1. **`tests/test_sft_prompt_alignment.py`**:
   - Added 3 focused unit tests verifying prompt formatting parity, dynamic fallback formatting, and token-level slice identity between training and runtime inference.
2. **`tests/test_task_evaluator.py`**:
   - Added assertions for single-character entity boundary matching to prevent false-positive substring matches.

---

## 3. Proposal for Controlled Follow-Up Training Run (Awaiting User Authorization)

Per Mission Rule 4, **no training has been executed**. If authorized by the user, the proposed follow-up experiment is specified as follows:

- **Experiment Name**: `exp18_aligned_sft_cpu_300`
- **Output Directory**: `experiments/overnight_runs/exp18_aligned_sft_run/` (isolated from canonical weights)
- **Base Checkpoint**: Isolated copy of canonical checkpoint `step_10000.pt` (SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`)
- **Training Data**: Temporary isolated split of 1,200 Exp18 training records (with `formatted_prompt` populated).
- **Validation Data**: Temporary isolated split of 160 Exp18 validation records.
- **Test Set**: 160 Exp18 test records strictly held out.
- **Budget**: 300 steps on CPU (approx. 12 seconds runtime, batch size 2, gradient accumulation 2).
- **Optimizer**: AdamW, learning rate 5e-5 decaying to 5e-6 via cosine schedule, weight decay 0.01, gradient clipping 1.0.
- **Success Gate**:
  1. Complete suppression of base pre-training prior (`"\nStep-by-Step Solution:\nThe total number..."`) under formatted instruction prompts.
  2. Increase in scorable held-out task accuracy above 10.0%.
  3. Increase in EOS completion rate above 25.0%.
