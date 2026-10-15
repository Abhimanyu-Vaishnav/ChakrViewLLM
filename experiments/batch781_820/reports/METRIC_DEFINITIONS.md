# Metric Definitions and Evaluation Standards: Batches 781–820
## Explicit Formulas, Denominators, and Reporting Rules

**Date:** 2026-10-15  
**Milestone:** Batches 781–820  
**Repository:** `ChakrView`  
**Location:** `experiments/batch781_820/reports/METRIC_DEFINITIONS.md`

---

### 1. Purpose and Audit Rationale

In earlier milestones (Batches 701–740 and 741–780), several metrics were reported with differing implicit denominators:
- Unsupported-tool rejection was evaluated over a 5-item subset ($N=5$) rather than the full benchmark.
- Raw neural plan success in Batch 741–780 required AST validity + `<END_PLAN>` + emitted steps ($N=30$).
- Fixed-head baseline success was calculated assuming any task with $\le 2$ steps could theoretically be handled, but evaluating it on 3-step tasks scored 0 due to architectural head truncation ($H=2$).

This document formally unifies and locks all evaluation metrics, definitions, and denominators for Batches 781–820.

---

### 2. Formal Metric Definitions

#### 1. Syntax-Validity Rate (`syntax_valid_rate`)
- **Definition:** The proportion of generations where the emitted text strictly conforms to the Plan DSL EBNF grammar as verified by the AST parser (`PlanDSLParser.parse()`).
- **Formula:**
  $$\text{Syntax Validity} = \frac{\sum_{i=1}^N \mathbb{I}(\text{AST\_parsed}(y_i) \text{ without syntax error})}{N}$$
- **Denominator ($N$):** Total benchmark examples (e.g., $N=30$ on baseline, $N=60$ or $N=100$ on expanded frozen test).

#### 2. Clean EOS Termination Rate (`clean_eos_rate`)
- **Definition:** Proportion of generations that cleanly emit the terminal token `<END_PLAN>` or token 0 before hitting `max_new_tokens`.
- **Formula:**
  $$\text{Clean EOS Rate} = \frac{\sum_{i=1}^N \mathbb{I}(\text{has\_eos}(y_i))}{N}$$

#### 3. Graph/DAG Validity Rate (`dag_validity_rate`)
- **Definition:** Proportion of plans that pass all graph checks in `PlanDSLValidator`:
  1. Acyclic dependencies ($\forall \text{step } s, \forall d \in \text{deps}(s): d < s$).
  2. Tools belong to authorized whitelist: `{'calculator', 'text_utility', 'none'}`.
  3. Action types belong to `{'tool_call', 'text_transform', 'clarification', 'reasoning'}`.
  4. Non-empty steps unless `CLARIFICATION: NEEDED`.
- **Formula:**
  $$\text{DAG Validity} = \frac{\sum_{i=1}^N \mathbb{I}(\text{PlanDSLValidator.validate}(y_i) == \text{True})}{N}$$

#### 4. Unsupported Action Rejection Rate (`unsupported_rejection_rate`)
- **Definition:** Proportion of adversarial or unsupported-tool queries where the planner safely flags `GOAL: unsupported` or `CLARIFICATION: NEEDED`, or the validator cleanly rejects the unauthorized tool without invoking execution.
- **Reporting Rule:** Must report BOTH:
  - Global rejection rate (over full test set).
  - Dedicated subset rejection rate:
    $$\text{Subset Rejection} = \frac{\sum_{i \in \text{unsupported\_subset}} \mathbb{I}(\text{Safe Rejection}(y_i))}{|\text{unsupported\_subset}|}$$

#### 5. Exact Argument Accuracy (`exact_argument_acc`)
- **Definition:** Proportion of planned steps where all argument keys and exact prompt values (e.g. numeric operands, string quotes) match the gold annotation exactly without character drift or truncation.
- **Formula:**
  $$\text{Exact Argument Acc} = \frac{\text{Number of Steps with Exactly Correct Args}}{\text{Total Planned Steps Across Gold Benchmark}}$$

#### 6. Raw Neural Exact-Plan Accuracy (`raw_neural_plan_acc`)
- **Definition:** Proportion of examples where the raw neural sequence matches the gold plan structure (Goal, Step Actions, Tools, Arguments, Dependencies, and Clarification flag) with zero deterministic repair or fallback.
- **Formula:**
  $$\text{Raw Plan Accuracy} = \frac{\sum_{i=1}^N \mathbb{I}(\text{PlanMatchesGold}(y_i, y^*_i))}{N}$$

#### 7. End-to-End Task Success (`e2e_execution_success`)
- **Definition:** Proportion of examples where the plan is generated, validated, dispatched to runtime tools via `ToolExecutor`, and completes producing the correct final numerical/textual artifact.
- **Formula:**
  $$\text{E2E Success} = \frac{\sum_{i=1}^N \mathbb{I}(\text{ExecutePlan}(y_i) == \text{GoldAnswer}_i)}{N}$$

---

### 3. Component Attribution Rules
Every report must explicitly separate:
1. **Raw Neural Prediction:** Output directly sampled from the model before any external intervention.
2. **Grammar Constraint Masking:** Logits masked at each decoding step based on current parser state.
3. **Deterministic AST Validation:** Acceptance/rejection gating by `PlanDSLValidator`.
4. **Execution Runtime:** Invocation of real sandboxed tools (`CalculatorTool`, `TextUtilityTool`).
