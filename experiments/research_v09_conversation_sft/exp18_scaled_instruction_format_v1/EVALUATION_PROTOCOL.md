# ChakrView Exp18 — Evaluation Protocol Specification

**Experiment**: `exp18_scaled_instruction_format_v1`  
**Location**: `experiments/research_v09_conversation_sft/exp18_scaled_instruction_format_v1/EVALUATION_PROTOCOL.md`  
**Evaluator**: `chakrview/evaluation/task_evaluator.py` (Task Evaluator v0.1)  
**Status**: FROZEN EVALUATION STANDARD  

---

## 1. Dual-Track Evaluation Methodology

Exp18 mandates strict dual-track evaluation to ensure model improvements are genuine and free from benchmark overfitting:

```
                                  ┌──────────────────────────────────────────────┐
                                  │   Model Checkpoint (Greedy Decoding T=0.0)   │
                                  └──────────────────────┬───────────────────────┘
                                                         │
                             ┌───────────────────────────┴───────────────────────────┐
                             ▼                                                       ▼
               ┌───────────────────────────┐                           ┌───────────────────────────┐
               │    Exp18 Held-Out Test    │                           │  Normal Chat Gate v0.1    │
               │   (160 unseen examples)   │                           │       (20 prompts)        │
               └─────────────┬─────────────┘                           └─────────────┬─────────────┘
                             │                                                       │
                             ▼                                                       ▼
               ┌───────────────────────────┐                           ┌───────────────────────────┐
               │    Task Evaluator v0.1    │                           │  Stutter & Loop Detector  │
               │  - Scorable Task Accuracy │                           │  - Trigram Repetition %   │
               │  - JSON Format Validity   │                           │  - EOS Rate %             │
               │  - EOS Termination Rate   │                           │  - Degenerate Loop Flag   │
               └───────────────────────────┘                           └───────────────────────────┘
```

---

## 2. Quantitative Acceptance Gates & Statistical Justification

The following acceptance gates govern model promotion and decision-making for Exp18:

| Gate Identifier | Target Metric | Minimum Threshold | Exp17 Baseline | Exp17 SFT Best | Statistical Justification & Measurement Plan |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **GATE-01** | **Scorable Task Accuracy** | $\ge 10.0\%$ | 0.0% | 0.0% | Evaluated on 140 scorable test records (20 short explanations held out). $10\%$ requires $\ge 14$ correct answers across arithmetic, factual QA, extraction, reference resolution, abstention, and instruction following. Standard error $\sigma \approx \sqrt{0.10 \times 0.90 / 140} \approx 2.5\%$. |
| **GATE-02** | **Structured JSON Validity** | $\ge 20.0\%$ | 0.0% | 0.0% | Evaluated on 20 structured formatting test records. $20\%$ requires $\ge 4$ syntactically valid RFC 8259 JSON objects. Standard error $\sigma \approx \sqrt{0.20 \times 0.80 / 20} \approx 8.9\%$. |
| **GATE-03** | **EOS Termination Rate** | $\ge 50.0\%$ | 0.0% | 45.0% | Evaluated across all 160 test records. Requires $\ge 80$ clean terminations emitting token ID `1`. Prevents regression in turn-boundary mechanics. |
| **GATE-04** | **Repetition Suppression** | $\le 15.0\%$ | 46.2% | 13.3% | Evaluated as the mean trigram deficit across the 20 prompts of Normal Chat Gate v0.1. Directly guards against degenerate output looping. |
| **GATE-05** | **Zero Regression Suite** | 100% Pass | 147/147 Pass | 147/147 Pass | All 147 unit and regression tests in the repository must pass with process exit code 0. |

---

## 3. Disaggregated Category Reporting Policy

To avoid masking category failures behind a composite average, evaluation outputs MUST disaggregate metrics across all eight categories:
- Sample count ($N$)
- Scorable count ($N_{\text{scorable}}$)
- Task accuracy (%)
- Clean EOS rate (%)
- Max tokens rate (%)
- Empty response rate (%)
- Format validity rate (%)

### Handling of Ambiguous Scientific Explanations
In strict compliance with governance rules, **short scientific explanations are NOT scored with heuristic string matching**. Task Evaluator v0.1 automatically labels them `NEEDS_REVIEW` (`outcome = NOT_APPLICABLE`). They are compiled into a separate human-review packet for qualitative verification and never conflated with deterministic task accuracy.
