# ChakrView Exp19 — Frozen Evaluation Protocol Specification

**Experiment Protocol**: `exp19_frozen_evaluation_protocol`  
**Target Checkpoint**: Evaluated experimental checkpoint from fine-tuning run  
**Held-Out Dataset**: 160 frozen test records from Exp18 generator  
**Test Manifest Reference**: `experiments/overnight_runs/preflight_exp18_20261024_114600/TEST_SPLIT_MANIFEST.json`  
**Status**: `FROZEN EVALUATION STANDARD`  

---

## 1. Frozen Test Set Integrity & Manifest

Evaluation for Exp19 must be executed against the identical 160 held-out test records established and verified in the preflight phase:
- **Total Test Records**: 160
- **Scorable Records**: 140 (20 `short_explanation` records held out as `NEEDS_REVIEW`)
- **Deterministic Content SHA-256**: `111db453d5ee52d436ef9a49f1ad6555ea572d118f84099ea0763e7f299ba047`
- **Record IDs SHA-256**: `44e8b0ff258ea2e0d92a0640fff5789cf98a620968983c4f7c49ded0329a6ea1`
- **No Overlap / Leakage**: Train vs Test prompt overlap is strictly 0.

---

## 2. Frozen Decoding Parameters & Runtime Contract

All model evaluations must adhere to the exact inference runtime specification:
- **Prompt Conditioning**: `format_conditioned_prompt(prompt, prompt_mode=rec['prompt_mode'], context=rec.get('context', ''))`
- **Decoding Method**: Greedy decoding (`temperature = 0.0`)
- **Generation Seed**: `seed = 42`
- **Max Generation Token Budget**: `45 tokens`
- **Stop Token**: Token ID `1` (`<|endoftext|>`)
- **Device**: `torch.device("cpu")`

---

## 3. Evaluator Implementation Standard

- **Evaluator**: `chakrview/evaluation/task_evaluator.py` (`TaskEvaluator v0.1`)
- **Entity Matching Policy**:
  - Short entity symbols ($\le 2$ chars, e.g. chemical symbols `"W"`, `"Fe"`) require word-boundary matching (`\b{ent}\b`) to avoid false-positive substring matches.
  - Default entity extraction strips leading English articles (`"the"`, `"a"`, `"an"`) to prevent common stop-word matches.
- **Ambiguous Scientific Explanations**:
  - `short_explanation` records are automatically flagged `NEEDS_REVIEW` (`outcome = NOT_APPLICABLE`). They are excluded from mathematical scorable accuracy to prevent metric contamination.

---

## 4. Quantitative Acceptance Gates for Exp19

Model promotion decisions must satisfy all four quantitative gates:

| Gate Identifier | Target Metric | Minimum Threshold | Clean Baseline | SFT Step 240 | Measurement Specification |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **GATE-01** | **Scorable Task Accuracy** | $\ge 5.0\%$ | 0.0% | 0.71% | Evaluated on 140 scorable records. $\ge 7$ correct answers across arithmetic, factual QA, extraction, reference resolution, and instruction following. |
| **GATE-02** | **EOS Termination Rate** | $\ge 50.0\%$ | 0.62% | 45.0% | Evaluated across all 160 test records. Requires $\ge 80$ clean terminations emitting token ID `1`. |
| **GATE-03** | **Prefix Monoculture Suppression** | $\le 25.0\%$ | - | 77.5% | No single 2-word response opening (e.g. `"The difference"`) may account for more than 25% of generated test completions. |
| **GATE-04** | **Zero Regression Suite** | 100% Pass | 126/126 Pass | 126/126 Pass | Full repository unit and integration test suite must pass with exit code 0. |

---

## 5. Strict Governance Against Test Overfitting

1. Checkpoint selection during training must be based exclusively on held-out validation loss ($L_{\text{val}}$).
2. Test records are evaluated only once per completed checkpoint run.
3. Test metrics are strictly prohibited from driving early stopping, learning rate tuning, or hyperparameter search.
