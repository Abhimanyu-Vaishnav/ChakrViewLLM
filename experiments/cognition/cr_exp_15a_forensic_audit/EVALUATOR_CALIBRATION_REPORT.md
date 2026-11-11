# CR-EXP-15A Evaluator Calibration Report
## Forensic Analysis of Evaluation Pipeline, Scoring Criteria, and Synthetic Validation

**Mission Identifier:** CR-EXP-15A  
**Auditor:** Autonomous Scientific Auditor & Benchmark Engineer  
**Date:** November 2026  
**Status:** Calibrated & Independently Validated  

---

## 1. Audit of the CR-15 Evaluation Pipeline

A line-by-line inspection of `experiments/cognition/cr_exp_15_architecture_binding/scripts/evaluate_cr15_benchmark.py` was conducted to determine whether the 0/1,200 score was an artifact of pipeline bugs:

1. **Benchmark Loading & Field Mapping:**
   - Probes contain keys: `id`, `category`, `prompt`, `subject`, `expected_target`, `earlier_value`, `distractor_values`.
   - In CR-15's final script, prompt construction appended `Assistant: ` only if not already present.
   - For all state update probes, prompt was ending in `Assistant: `, preserving context formatting.
2. **Tokenization and Generation:**
   - Context window capped cleanly at 250 tokens (`input_ids[:, -250:]`), matching model positional embedding limits.
   - Generation ran greedily with `max_new_tokens=30`.
   - Token decoding used `errors="replace"` to prevent Unicode crash on partial subwords.
3. **Stopping Conditions:**
   - Generation stopped on token `0` (pad) or when newline / next speaker (`\nUser:`, `\nTurn`) appeared.
   - Inspection of raw outputs confirmed complete sentences were generated (e.g., `Ankush is currently in Aurangabad.`). Generations were not prematurely truncated or empty.
4. **Scoring Logic:**
   - Non-abstention probes required `has_subject` AND `has_value`.
   - Substring matching was used (`target_lower in out_lower` and `value_lower in out_lower`).
   - Abstention probes checked for standard epistemic cues (`not mentioned`, `unknown`, `never mentioned`, `cannot be determined`, etc.).

---

## 2. Independent Evaluator Calibration Suite (12 Synthetic Unit Tests)

To establish mathematical rigor, an independent calibrated evaluator (`calibrated_evaluator.py`) was constructed with word-boundary regex matching and separate tracking of:
- `strict_correct` (requires correct subject, correct target value, and absence of stale value)
- `loose_correct` (requires correct target value and absence of stale value, ignoring subject)
- `subject_correct` (correct subject identity matched)
- `value_correct` (correct target value matched)
- `stale_value_retained` (earlier pre-update value retained)
- `distractor_retained` (distractor entity/location retained)
- `abstention_correct` (calibrated abstention on unanswerable probes)

### Unit Test Execution Matrix

| Test ID | Scenario Description | Model Response Simulated | Strict | Subject | Value | Status |
|---|---|---|---|---|---|---|
| **CASE_1** | Correct complete subject-attribute-value | `Rajnath is currently located in Saharanpur.` | `True` | `True` | `True` | **PASS** |
| **CASE_2** | Concise value-only answer where task asks for value | `Saharanpur.` | `False` | `False` | `True` | **PASS** |
| **CASE_3** | Wrong subject with correct value | `Amitabh is currently in Saharanpur.` | `False` | `False` | `True` | **PASS** |
| **CASE_4** | Correct subject with wrong value | `Rajnath is currently in Aurangabad.` | `False` | `True` | `False` | **PASS** |
| **CASE_5** | Correct subject with stale earlier value | `Rajnath is currently in Rourkela.` | `False` | `True` | `False` | **PASS** |
| **CASE_6** | Correct subject with wrong attribute | `Rajnath is a doctor.` | `False` | `True` | `False` | **PASS** |
| **CASE_7** | Valid abstention on unanswerable probe | `The location of Vikas is unknown and not mentioned...` | `True` | `None` | `False` | **PASS** |
| **CASE_8** | Incorrect false abstention on answerable probe | `I don't know where Rajnath is.` | `False` | `False` | `False` | **PASS** |
| **CASE_9** | Empty generation | `""` | `False` | `False` | `False` | **PASS** |
| **CASE_10**| Degenerate repetition loop | `Assistant: Assistant: Assistant: Assistant:` | `False` | `False` | `False` | **PASS** |
| **CASE_11**| Conflicting values (stale and current both in output) | `Rajnath moved from Rourkela to Saharanpur.` | `False` | `True` | `True` | **PASS** |
| **CASE_12**| Capitalization and punctuation variations | `rajnath lives in saharanpur!` | `True` | `True` | `True` | **PASS** |

**Calibration Verdict:** 12/12 unit tests passed. The evaluator accepts valid variations, enforces boundary constraints, and correctly isolates subject, value, and stale-value dynamics.

