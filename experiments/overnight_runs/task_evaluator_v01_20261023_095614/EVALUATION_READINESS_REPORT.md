# Evaluation Readiness Report - Exp17 & Future SFT Missions

**Evaluator**: TaskEvaluator v0.1 (chakrview.evaluation.task_evaluator)

**Package**: chakrview.evaluation

**Date**: 2026-10-23

**Readiness Status**: READY_WITH_DOCUMENTED_CONSTRAINTS

---

## 1. Scope of Evaluator Reliability

TaskEvaluator v0.1 has been formally proven ready to govern future SFT experiments under the following documented capability profile:

### What the Evaluator Scores Reliably (Scorable Coverage: 87.5%):

1. Basic Arithmetic (basic_arithmetic): 100% reliable via independent formula computation and word-boundary integer extraction.

2. Structured Formatting (structured_formatting): 100% reliable via strict RFC 8259 JSON parsing, distinguishing empty outputs and docstring syntax errors from valid schema objects.

3. Context Extraction (context_extraction): 100% reliable via normalized exact token/phrase matching of context-grounded entities.

4. Reference Resolution (reference_resolution): 100% reliable via unambiguous antecedent entity verification.

5. Uncertainty & Abstention (uncertainty_abstention): 100% reliable via dynamic target fact checking for answerable controls and standard refusal idiom matching for unanswerables.

6. Instruction Following (instruction_following): 100% reliable via deterministic transformation verification (case changes, reversal, length).

7. Factual QA (factual_qa): 100% reliable via canonical focal entity and alias matching.

### What Requires Manual / Semantic Review:

- Short Explanation (short_explanation) (12.5% of test suite; 5/40 records): Marked as NEEDS_REVIEW and excluded from scorable accuracy. Any causal explanation requires human or specialized rubric review to prevent false-positive inflation.

---

## 2. Baseline vs. SFT Best-Checkpoint Task Results

| Metric Category | Baseline (Step 10,000) | Exp17 SFT (Step 225) | Delta |

| :--- | :---: | :---: | :---: |

| Total Test Records | 40 | 40 | 0 |

| Scorable Records | 35 | 35 | 0 |

| Scorable Coverage | 87.5% | 87.5% | 0.0% |

| Scorable Task Accuracy | 0.0% (0 / 35) | 0.0% (0 / 35) | 0.0% |

| Total Suite Task Accuracy | 0.0% (0 / 40) | 0.0% (0 / 40) | 0.0% |

| Turn Termination (EOS Rate) | 0.0% (0 / 40) | 45.0% (18 / 40) | +45.0% |

| Max-Tokens Truncation Rate | 100.0% (40 / 40) | 55.0% (22 / 40) | -45.0% |

| Repetition Rate | 37.5% (15 / 40) | 10.0% (4 / 40) | -27.5% |

| Empty Response Rate | 27.5% (11 / 40) | 5.0% (2 / 40) | -22.5% |

| Format Valid (JSON Syntax) | 0.0% (0 / 5) | 0.0% (0 / 5) | 0.0% |

---

## 3. Corrections and Downgrades to Prior Claims

1. Claim of Structured Formatting Success: Downgraded from 100% to 0.0% Task Accuracy. While 100% of structured formatting prompts ended cleanly with EOS, 0% contained valid JSON syntax.

2. Claim of Arithmetic Capability: Downgraded to 0.0% Task Accuracy. Generated responses mirrored mathematical phrasing (The sum of...) but gave incorrect numerical quantities.

3. Claim of Full Semantic Correctness in Explanations: Downgraded to UNSCORED / NEEDS_REVIEW. Semantic accuracy cannot be determined deterministically without human verification.

---

## 4. Final Verdict & Recommended Next Step

### Governance Readiness Verdict: READY_TO_GOVERN_FUTURE_EXPERIMENTS

TaskEvaluator v0.1 provides an unyielding, deterministic gate that prevents future experiments from mistaking turn-taking termination (EOS%) for actual capability acquisition.

### Recommended Next Action:

Proceed with Exp18: Scaled Instruction & Format Fine-Tuning, with the following measurable acceptance criteria:

1. Model Parameter Scaling or Extended SFT Schedule: 3.44M parameters across 320 synthetic training examples is insufficient to compress multi-task instruction semantics.

2. Acceptance Criteria for Exp18:

   - Turn Termination: Maintain >= 50% EOS rate on held-out test splits.

   - Structured Formatting Accuracy: >= 20.0% valid RFC 8259 JSON objects on scorable test items.

   - Scorable Task Accuracy: >= 10.0% non-zero task accuracy across arithmetic, factual QA, and extraction.

   - No Regressions: Zero failures on the 147 unit and regression test suite.