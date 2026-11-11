# CR-EXP-18: Phase B — CR-17R Multi-Turn Failure Reproduction Report

## 1. Reproduction Setup & Lineage
- **Target Checkpoint:** `releases/v0.9.1_conversational_preview/chakrview_v091_model.pt`
- **Target Checkpoint SHA-256:** `f161ccdc501bca34f6ed2019252f2e51d18fa113ce0cc90187c4ff0cdd180d10`
- **Evaluator Script:** `experiments/cognition/cr_exp_17r/scripts/evaluate_multiturn.py`
- **Evaluator Script SHA-256:** `2b359090fc304611a12fd4a6b7ef23c60b13c7b1caccd8623513a98abbcf2612`
- **Evaluation Settings:** Deterministic decoding (`temperature=0.0`, `repetition_penalty=1.15`, `no_repeat_ngram_size=3`, `max_new_tokens=40`).

## 2. Quantitative Reproduction Verification

| Metric Category | Evaluated Turns | Historical CR-17R Correct | Verified Independent Correct | Reproduction Status |
| :--- | :---: | :---: | :---: | :---: |
| `fact_retention` | 56 | 0 / 56 (0.0%) | 0 / 56 (0.0%) | **REPRODUCED (100% Match)** |
| `corrected_fact` | 2 | 0 / 2 (0.0%) | 0 / 2 (0.0%) | **REPRODUCED (100% Match)** |
| `reference_resolution` | 3 | 0 / 3 (0.0%) | 0 / 3 (0.0%) | **REPRODUCED (100% Match)** |
| `topic_change` | 30 | 0 / 30 (0.0%) | 0 / 30 (0.0%) | **REPRODUCED (100% Match)** |
| `early_fact_return` | 28 | 0 / 28 (0.0%) | 0 / 28 (0.0%) | **REPRODUCED (100% Match)** |
| `missing_info` | 27 | 0 / 27 (0.0%) | 0 / 27 (0.0%) | **REPRODUCED (100% Match)** |
| **Total Evaluated Turns** | **146** | **0 / 146 (0.0%)** | **0 / 146 (0.0%)** | **REPRODUCED (Strict Match)** |

## 3. Qualitative Verification of Generation Behavior
Inspection of all 30 conversations demonstrates that the failure is not an evaluation scoring glitch or strict regex mismatch. Rather:
1. **Turn 1 to 2:** The model produces fluent greetings and acknowledgments.
2. **Turn 3 onwards:** When the prompt contains $\ge 2$ prior turns, the model fails to retrieve the variable values (e.g. `User: My name is Tanvi... Assistant asks: What was my name again?` -> Model responds: `'You are describles in Neyouses a port programming...'`).
3. **Entity Intrusion:** The model persistently leaks memorized training entities (`"Arjun"`, `"Bangary"`, `"Berok"`, `"rack"`, `"node"`).

The 0/56 fact retention failure is **confirmed, authentic, and reproducible**.
