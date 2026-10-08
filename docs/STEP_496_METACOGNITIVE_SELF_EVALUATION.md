# STEP 496: Meta-Cognitive Self-Evaluation Specification

## Purpose
Define a formal, governed Meta-Cognitive Evaluator (`MetaCognitiveEvaluator`) that assesses reasoning trajectories, measures confidence vs uncertainty, estimates failure risks, and recommends recovery strategies without self-authorizing resource expansions.

---

## 1. Concrete Evaluation Interface

The evaluator inspects:
1. `task_difficulty`: Estimated chain depth, relation complexity, entity novelty.
2. `current_reasoning_sufficiency`: State delta stability across cycles ($\|s_t - s_{t-1}\|$).
3. `confidence`: Calibrated prediction margin and controller continuation logits.
4. `uncertainty`: Entropy of candidate readout distributions and epistemic status (`KNOWN`, `SUPPORTED`, `UNCERTAIN`, `CONFLICTING`, `UNKNOWN`).
5. `conflict_indicators`: Contradiction presence between premise clauses.
6. `trajectory_stability`: Divergence between successive attention weights and hidden norms.
7. `expected_value_metrics`: Expected value of additional reasoning cycles ($EV_{cycles}$), memory retrieval ($EV_{memory}$), or peer assistance ($EV_{peer}$).

### Structured Output: `MetaCognitiveAssessment`
- `current_state`: Summary of active recurrent state norm and cycle depth.
- `confidence`: $[0.0, 1.0]$ calibrated confidence score.
- `uncertainty`: $[0.0, 1.0]$ normalized uncertainty score.
- `sufficiency`: Boolean indicating whether current result meets halting threshold without defect risk.
- `failure_risk`: $[0.0, 1.0]$ estimated defect risk.
- `recommended_strategy`: Enumerated recommendation (`STRATEGY_DIRECT`, `STRATEGY_RECURSIVE`, `STRATEGY_MEMORY`, `STRATEGY_RETRY`, `STRATEGY_PEER_REQUEST`).
- `rationale`: Auditable epistemic justification for the recommendation.

---

## 2. Resource Governance Rule
- **Resource availability != permission != requirement**.
- The Meta-Cognitive Evaluator may recommend `STRATEGY_PEER_REQUEST` or `STRATEGY_RETRY`, but execution is gated by the external Resource Governor.
