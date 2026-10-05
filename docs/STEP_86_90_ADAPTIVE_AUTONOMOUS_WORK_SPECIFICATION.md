# Specification: Steps 86–90 Adaptive Autonomous Work Orchestration

## 1. Overview and Architectural Purpose
Milestone Steps 86–90 turns the existing Persistent Project Brain (PPB, Steps 78–81) and Persistent Task Orchestration (Steps 82–85) into a closed-loop, persistent, resource-aware cognitive work system.

The orchestration pipeline establishes:
```
USER GOAL
  ↓
Persistent Project Brain (PPB)
  ↓
Resource & Context Budget Planner (Step 87)
  ↓
Persistent Task Graph (DAG)
  ↓
Bounded Task Execution
  ↓
Observation / Result
  ↓
Reasoning + Critical Thinking + Self Evaluation
  ↓
UNKNOWN / INSUFFICIENT?
  ├── YES → Autonomous Investigation Loop (Step 88) → Evidence Verification → PPB Update
  └── NO
  ↓
Dynamic Task Expansion (Step 86) / Adaptive Re-planning (Step 89)
  ↓
Continue Work Loop (Bounded Iterations)
  ↓
Goal Verification Gate (Step 90)
  ↓
ACCEPT / PARTIAL / REVISE / REJECT / ABSTAIN
```

---

## 2. Implemented Components & Module Map

### 2.1 Dynamic Subtask Models (`chakrview/cognition/ppb/expansion_models.py`)
- **`DynamicSubtaskRequest`**: Explicit, deterministic contracts for generating subtasks in flight with parent IDs, dependencies, resource profiles, affected files, required symbols, and provenance.
- **`TaskProvenance`**: Tracks the origin of dynamic subtasks (`INITIAL_DECOMPOSITION`, `DYNAMIC_EXPANSION`, `REPLANNING`, `INVESTIGATION`, `RETRY`), including the triggering parent node ID, generation timestamp, and explicit rationale.
- **`GoalVerificationVerdict`**: Enum capturing root-goal audit verdicts: `ACCEPT`, `PARTIAL`, `REVISE`, `REJECT`, `ABSTAIN`.
- **`GoalVerificationResult`**: Authoritative completion report recording verdict, confidence score ($0.0 - 1.0$), satisfied/unsatisfied requirements, unresolved unknowns, suggested follow-ups, and audit justification.

### 2.2 Context & Resource Budget Planner (`chakrview/cognition/ppb/budget_planner.py`)
- Bounded planner integrating directly with `HardwareCapability`.
- Analyzes node context token demands against available RAM and hardware tier.
- Dynamically selects one of five budget actions:
  1. `EXECUTE_DIRECT`: Resources abundant; run node in full context.
  2. `RETRIEVE_PPB_ONLY`: Constrained memory; load only targeted PPB records, avoiding redundant filesystem reads.
  3. `SPLIT_TASK`: Context exceeds single-window threshold; partition affected files into subtasks.
  4. `DEFER_QUEUE`: Memory critically exhausted ($< 1.0\text{ GB}$ available); postpone execution.
  5. `REQUEST_INVESTIGATION`: Critical file or symbol context is missing or marked `UNKNOWN`.
- Fully supports `LOW_RESOURCE` (CPU-only), `STANDARD`, and `ACCELERATED` capability profiles through an explicit contract.

### 2.3 Autonomous Investigation Loop (`chakrview/cognition/ppb/investigation_loop.py`)
- Bridges structured reasoning's `InvestigationRequirement` with the existing `EvidenceVerifier` from Steps 74–77.
- When unknowns or knowledge gaps emerge during execution:
  - Dispatches candidate evidence to independent verification.
  - Preserves grounded epistemic statuses: `FACT`, `INFERRED`, `CONTESTED`, `STALE`, `UNKNOWN`, and `INSUFFICIENT`.
  - Prohibits converting ungrounded claims into `FACT` solely because an external source was cited.
  - Automatically commits verified or contested findings to SQLite PPB with auditable provenance.

### 2.4 Adaptive Re-planning Engine (`chakrview/cognition/ppb/replan_and_gate.py`)
- Evaluates execution results, newly discovered dependencies, and investigation outcomes.
- Employs Kahn's algorithm cycle prevention (`PersistentTaskGraph.validate_no_cycles`) to guarantee that dynamically added subtasks never create circular dependency deadlocks.
- Bounded replanning limit (default: 3 cycles) prevents infinite expansion loops.
- Preserves all historical completed work; never retroactively discards completed progress.

### 2.5 Goal Completion & Autonomous Work Gate (`chakrview/cognition/ppb/replan_and_gate.py`)
- Whole-goal verification barrier: does **not** assume completion solely because subtasks exited.
- Rigorous checklist:
  - Graph completeness (all nodes completed $\to$ if not, returns `PARTIAL`).
  - Failure/blocked audit (any failed/blocked $\to$ returns `REJECT`).
  - Affected file grounding (unresolved unknowns $\to$ returns `ABSTAIN`).
  - Stale record detection (stale records in affected scope $\to$ returns `REVISE` with generated follow-up tasks).
  - Validation/test requirement verification.
- On `ACCEPT`, records authoritative `EpistemicStatus.FACT` goal milestone in PPB.

---

## 3. Proven Capabilities
1. **Dynamic Task Expansion**: Successfully creates subtasks in-flight with deterministic IDs, explicit parents, provenance metadata, and Kahn's algorithm cycle validation.
2. **Deterministic Task Identity**: All node hashes and provenance stamps remain repeatable across runs.
3. **Context & Budget Decision Making**: Gracefully shifts execution strategies across `LOW_RESOURCE`, `STANDARD`, and `ACCELERATED` profiles without altering business logic.
4. **Autonomous Evidence Verification**: Automatically verifies evidence for knowledge gaps, preserves epistemic distinctions, and updates PPB.
5. **Adaptive Re-planning**: Dynamically adds inspection and validation nodes upon discovering uninspected dependencies, while strictly bounded by max replanning cycles.
6. **Persistence and Interruption Survivability**: Full DAG, lease status, and knowledge records survive simulated process termination and resume seamlessly.
7. **Authoritative Whole-Goal Verification**: Prevents false completions by evaluating graph state, epistemic grounding, and test outcomes into `ACCEPT`, `PARTIAL`, `REVISE`, `REJECT`, or `ABSTAIN`.
8. **Neural Invariant ($\Delta W = 0$)**: ChakrMicro model parameters (3,443,136) and SHA-256 weight hash (`c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`) remain frozen with zero neural authority over execution.

---

## 4. Partially Proven Capabilities
1. **Multi-Worker Execution Partitioning**: Clean worker and lease contracts exist for distributed execution (`TaskExecutionLease`, task payloads, context slices), but physical multi-node network clusters are not implemented (by specification design).
2. **Subtask Splitting Heuristics**: Automatic AST-level file chunking for large modules is mocked via chunk partitioning; advanced semantic code splitting will be refined in future milestones.

---

## 5. Unproven Capabilities
1. **Autonomous Neural Self-Modification**: Deliberately disabled; self-improvement is strictly restricted to project knowledge evolution, failure pattern recording, and replanning heuristics.
2. **Networked Distributed Consensus**: Multi-process clustering across physical machines is deferred to dedicated cluster milestones.

---

## 6. Authority & Security Boundaries
- **Zero Neural Write Authority**: The neural baseline cannot issue direct filesystem modifications, delete task graph nodes, or bypass approval gates.
- **Fail-Closed Gate**: Any detected ungrounded claim or unknown dependency forces the goal gate to `ABSTAIN` or `REVISE`, preventing silent partial deployments.
- **Auditable Provenance**: All dynamic nodes record the trigger node, generation timestamp, and causal rationale.
