# CR-EXP-16 Stage A Capability Results
## Empirical Status Following Stage A Pilot Investigation

**Mission Identifier:** CR-EXP-16  
**Status:** **NOT DEMONSTRATED** (Pilot Triggered Repetitive Collapse Guardrail)  
**Date:** November 2026  

---

## 1. Capability Status Matrix

| Capability | Baseline Stage B | Stage A Pilot | Status | Limiting Mechanism |
|---|---|---|---|---|
| **Exact Sequence Copying** | 0.00% (0/100) | **0.00%** (0/100) | **NOT DEMONSTRATED** | Emits repeated static unigrams (`stone`, `storm`) |
| **Single-Token Copying** | 0.00% (0/100) | **0.00%** (0/100) | **NOT DEMONSTRATED** | Fails to route single token from prompt context |
| **Multi-Token Copying** | 0.00% (0/100) | **0.00%** (0/100) | **NOT DEMONSTRATED** | Multi-token representation unrouted |

---

## 2. Guardrail Determination

The capability criteria for Stage A full budget expansion require:
> 1. Dataset leakage audit passes: **PASS**
> 2. Evaluator calibration passes: **PASS**
> 3. Model generates non-empty outputs: **PASS**
> 4. Training loss behaves reasonably: **PASS** ($4.63 \to 1.88$)
> 5. Pilot shows measurable improvement over frozen baseline: **FAIL** ($0.00\%$)
> 6. Validation behavior does not indicate memorization or collapse: **FAIL** (Repetitive unigram collapse)

Because criteria 5 and 6 failed, **full Stage A training is withheld**.

