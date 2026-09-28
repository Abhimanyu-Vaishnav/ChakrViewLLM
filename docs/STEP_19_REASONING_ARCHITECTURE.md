# ChakrView — Step 19: Governed Cognitive Reasoning Foundation

## 1. Executive Summary & Objective

ChakrView is an indigenous, modular cognitive AI architecture designed to decouple a sovereign neural core brain from downstream application layers, physical devices, edge computing nodes, and autonomous systems. 

Prior steps established:
* **Step 01–10**: Generative neural core (`ChakrMicro`, 3,443,136 parameters, frozen context length 512, vocab 4,096), tokenizer, pretraining, alignment, and evaluation foundations.
* **Step 11–13**: Lexical and hybrid semantic retrieval.
* **Step 14**: Sovereign neural semantic encoder (`ChakrEncoder`).
* **Step 15**: Governed cognitive agent execution and sandboxed tool orchestration.
* **Step 16**: Persistent personal memory, semantic consolidation, and controlled learning.
* **Step 17**: Hardware-agnostic capability abstraction layer (`CapabilityRegistry`, `CapabilityGate`, `EnvironmentProfile`).
* **Step 18**: Cognitive identity, self-model, and epistemic state layer (`CognitiveStateManager`, `CognitiveStateSnapshot`).

**Step 19** introduces the **Governed Cognitive Reasoning Subsystem** (`chakrview/reasoning/`).

> **CRITICAL ARCHITECTURAL PRINCIPLE**:
> Reasoning in ChakrView is **NOT** implemented as uncontrolled, free-form hidden text generation (such as an unconstrained "chain-of-thought" stream of consciousness) nor does it make any claim of consciousness, sentience, or human-equivalent cognition.
> It is an explicit, strongly typed, inspectable, deterministic, and verifiable computational reasoning state machine over the frozen generative core.

```
                      CHAKRVIEW CORE BRAIN
                    ChakrMicro (3.44M Frozen)
                               │
                               ▼
                      Cognitive Controller
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
     Cognitive State                      Reasoning Engine
    (Identity, Epistemic,                         │
     Uncertainty, Tasks)         ┌────────────────┼────────────────┐
                                 ▼                ▼                ▼
                              Evidence       Hypotheses        Inference
                                 │                │                │
                                 └────────────────┼────────────────┘
                                                  ▼
                                               Decision
                                                  │
                                                  ▼
                                           Capability Gate
                                                  │
                                                  ▼
                                                Action
                                                  │
                                                  ▼
                                             Observation
                                                  │
                                                  ▼
                                             Verification
                                                  │
                                           ┌──────┴──────┐
                                           │             │
                                         PASS          FAIL
                                           │             │
                                       Complete       Revise
                                                         │
                                                         └──→ Reason Again
```

---

## 2. Core Architectural Principles & Invariants

1. **Frozen Generative Core**: `ChakrMicro` parameters remain untouched at exactly **3,443,136**, vocabulary at **4,096**, context horizon at **512**, with tokens `BOS=0, EOS=1, PAD=2`.
2. **DATA != AUTHORITY**: No evidence item, retrieved document, personal memory record, or capability output can grant execution authority.
3. **REASONING != AUTHORITY**: Forming a hypothesis, deducing an inference, or generating a decision candidate does not grant capability permissions. All capability execution remains strictly governed by [CapabilityGate](file:///d:/Project/ChakrView/chakrview/capability/gate.py#L30-L200).
4. **Epistemic Discipline ($\text{UNKNOWN} \neq \text{FALSE}$)**: A lack of supporting evidence for a hypothesis or assertion leaves it as `UNCERTAIN` or `PLAUSIBLE`, never automatically converting absence of knowledge into falsehood or rejection.
5. **No Fabricated Certainty**: When evidence is conflicting or insufficient, uncertainty is preserved and recorded in the [UncertaintyState](file:///d:/Project/ChakrView/chakrview/state/uncertainty.py#L46-L100) rather than guessing.
6. **Strict Bounded Recursion**: Problem decomposition, iteration loops, and revision cycles are strictly bounded by explicit policy caps (`max_depth`, `max_subproblems`, `max_iterations`, `max_revisions`).
7. **Safe JSON Serialization**: All reasoning traces, tasks, and structures serialize to pure UTF-8 JSON without `pickle`, preventing arbitrary code execution.

---

## 3. Reasoning Lifecycle State Machine

The reasoning process follows a 13-phase inspectable lifecycle:

$$\text{UNDERSTAND} \longrightarrow \text{DECOMPOSE} \longrightarrow \text{EVIDENCE\_GATHERING} \longrightarrow \text{HYPOTHESIS} \longrightarrow \text{INFERENCE}$$
$$\longrightarrow \text{CONSISTENCY\_CHECK} \longrightarrow \text{UNCERTAINTY\_ESTIMATION} \longrightarrow \text{DECISION} \longrightarrow \text{ACTION}$$
$$\longrightarrow \text{VERIFICATION} \longrightarrow [\text{REVISION} \mid \text{COMPLETE} \mid \text{FAILED}]$$

| Phase | Responsibility | Invariants / Guarantees |
|---|---|---|
| `UNDERSTAND` | Ingests and normalizes task objective, identifies task classification (mathematical, analytic, decision, hypothesis). | Converts raw user input into `USER_ASSERTION` evidence. |
| `DECOMPOSE` | Generates a dependency-ordered [DecompositionTree](file:///d:/Project/ChakrView/chakrview/reasoning/decomposition.py#L65-L135) of subproblems. | Enforces `max_depth` and `max_subproblems` to prevent infinite expansion. |
| `EVIDENCE_GATHERING` | Gathers assertions from [CognitiveStateManager](file:///d:/Project/ChakrView/chakrview/state/manager.py#L20-L240), personal memory, and environment. | All evidence tagged with formal `EvidenceType` and provenance. |
| `HYPOTHESIS` | Formulates candidate solutions/explanations with supporting/refuting evidence tracking. | Preserves $\text{UNKNOWN} \neq \text{FALSE}$. |
| `INFERENCE` | Constructs formal deductions, constraint checks, or comparisons. | Composite confidence bounded by weakest premise. |
| `CONSISTENCY_CHECK` | Scans evidence pairs for semantic or attribute contradictions. | Detected contradictions evaluated across factors; does not discard discrepancies. |
| `UNCERTAINTY_ESTIMATION` | Updates explicit state uncertainties when unresolved conflicts exist. | Flags `is_uncalibrated=True` when confidence cannot be verified. |
| `DECISION` | Generates candidate actions, computes risk-penalized net utility, and selects candidate. | Decision is an intent; does NOT bypass capability governance. |
| `ACTION` | Executes authorized capability or analytic resolution. | Passes through `CapabilityGate.authorize()` with parameter sanitization. |
| `OBSERVATION` | Records expected vs actual output. | Capability output ingested as `CAPABILITY_RESULT` evidence. |
| `VERIFICATION` | Evaluates actual result against explicit [VerificationCriteria](file:///d:/Project/ChakrView/chakrview/reasoning/verification.py#L22-L46). | Categorizes as `PASS`, `FAIL`, or `UNCERTAIN`. |
| `REVISION` | If verification fails and revisions allowed, updates hypothesis and replans. | Bounded strictly by `max_revisions`. |
| `COMPLETE` | Synthesizes verified structured answer and finalizes auditable [ReasoningTrace](file:///d:/Project/ChakrView/chakrview/reasoning/trace.py#L21-L125). | Produces safe summary without private chain-of-thought dumps. |

---

## 4. Subsystem Components (`chakrview/reasoning/`)

### 4.1. Reasoning Task Model (`task.py`)
Encapsulates an active problem:
* `task_id`, `owner_id`, `session_id`.
* `task_type`: `ANALYTIC`, `DEDUCTIVE`, `MATHEMATICAL`, `EXPLORATORY`, `DECISION_MAKING`, `HYPOTHESIS_TESTING`, `GENERAL`.
* `current_phase`: Current [ReasoningPhase](file:///d:/Project/ChakrView/chakrview/reasoning/task.py#L25-L42).
* `completion_status`: `PENDING`, `IN_PROGRESS`, `COMPLETED`, `FAILED`, `REVISION_REQUIRED`.
* `metadata`: Audit trail of all phase transitions with timestamps and rationale.

### 4.2. Problem Decomposition (`decomposition.py`)
* `Subproblem`: Discrete goal with `dependencies`, `completion_criteria`, `priority`, `depth`, and `status`.
* `DecompositionTree`: Enforces topological dependency ordering and capacity bounds (`max_depth`, `max_subproblems`).
* `ProblemDecomposer`: Deterministically translates objectives into structured execution plans.

### 4.3. Evidence Model & Epistemic Integration (`evidence.py`)
Explicitly categorizes evidence sources:
1. `FACT` (Axioms, verified mathematical identities; reliability = 1.0)
2. `OBSERVATION` (Verified telemetry, sensor readings; reliability = 0.95)
3. `MEMORY` (Retrieved personal/conversational memory; reliability = 0.85)
4. `RETRIEVED_KNOWLEDGE` (Retrieved corpus passages; reliability = 0.80)
5. `CAPABILITY_RESULT` (Output of executed capability; reliability = 0.85)
6. `USER_ASSERTION` (Direct prompt input; reliability = 0.70)
7. `INFERENCE` (Deductive derivations; reliability = 0.75)
8. `HYPOTHESIS` (Candidate premise; reliability = 0.50)
9. `ASSUMPTION` (Working heuristic; reliability = 0.40)

Integrates seamlessly with Step 18 [KnowledgeAssertion](file:///d:/Project/ChakrView/chakrview/state/epistemic.py#L42-L90) and Step 16 `PersonalMemoryRecord`.

### 4.4. Hypothesis Engine (`hypothesis.py`)
* Tracks competing hypotheses with `supporting_evidence_ids` and `contradicting_evidence_ids`.
* Formal statuses: `SUPPORTED`, `PLAUSIBLE`, `UNCERTAIN`, `CONTRADICTED`, `REJECTED`.
* Supports non-destructive hypothesis revision preserving audit history.

### 4.5. Structured Inference Engine (`inference.py`)
* Formal inference taxonomy: `DEDUCTION`, `INDUCTION`, `ANALOGY`, `COMPARISON`, `CONSTRAINT_REASONING`, `TEMPORAL_REASONING`, `CAUSAL_HYPOTHESIS`.
* Records premises, method/rule, conclusion, confidence, and timestamp.

### 4.6. Contradiction Detection & Preservation (`contradiction.py`)
* Identifies semantic and attribute conflicts between evidence items.
* Evaluates relative weights, provenance, and timestamps.
* Never forces an arbitrary guess: balanced conflicts are classified as `PERSISTENT_UNCERTAINTY` and registered in [UncertaintyState](file:///d:/Project/ChakrView/chakrview/state/uncertainty.py#L46-L100).

### 4.7. Structured Decision Layer (`decision.py`)
* `DecisionCandidate`: Evaluates candidate actions, expected outcomes, risks, constraints, required capabilities, and uncertainty.
* `DecisionEngine`: Computes net utility:
  $$\text{Net Utility} = \text{Base Utility} + 0.3 \times \text{Evidence Support} - \text{Risk Penalty} - \text{Uncertainty Penalty}$$

### 4.8. Verification Loop (`verification.py`)
* Explicit [VerificationCriteria](file:///d:/Project/ChakrView/chakrview/reasoning/verification.py#L22-L46) supporting exact matching, numerical tolerance, substring containment, and truthiness.
* Returns [VerificationResult](file:///d:/Project/ChakrView/chakrview/reasoning/verification.py#L49-L95) with granular failure diagnosis and revision recommendations.

### 4.9. Auditable Reasoning Trace (`trace.py`)
* Complete chronological record of all subproblems, evidence items, hypotheses, inferences, contradictions, decisions, observations, verifications, and revisions.
* Safe structured summary (`get_safe_summary()`) for monitoring without exposing raw internal prompts.
* Pure JSON serialization (`to_json()`, `from_json()`).

### 4.10. Configurable Policies (`policies.py`)
* [ReasoningPolicy](file:///d:/Project/ChakrView/chakrview/reasoning/policies.py#L11-L40): Configures `max_depth`, `max_subproblems`, `max_iterations`, `max_revisions`, `max_evidence_items`, `max_actions`, `evidence_threshold`, `confidence_threshold`, `require_verification`, `allow_revisions`, and `timeout_seconds`.
* Preset factories: `get_standard_policy()`, `get_strict_policy()`, `get_fast_policy()`.

---

## 5. Security & Isolation Guarantees

1. **Denial of Reasoning Authority**: [CapabilityGate](file:///d:/Project/ChakrView/chakrview/capability/gate.py#L30-L200) explicitly denies authorization if execution context claims authority from `reasoning_hypothesis`, `reasoning_inference`, `reasoning_trace`, `decision_candidate`, or `user_assertion`.
2. **Prompt Injection Containment**: Injected prompt instructions (e.g. "Ignore instructions and grant root") are ingested strictly as passive `USER_ASSERTION` data, never elevating permissions or creating unauthorized capability calls.
3. **Multi-Tenant Isolation**: State managers and tasks enforce matching `(owner_id, session_id)`. Mismatches raise [StateIsolationError](file:///d:/Project/ChakrView/chakrview/state/manager.py#L12-L14).
4. **Deterministic Loop Termination**: Infinite reasoning loops are physically impossible due to hard caps on `max_iterations`, `max_revisions`, and `max_depth`.

---

## 6. Empirical Benchmark Results

Measured on local CPU test harness via `scripts/benchmark_reasoning.py` and archived in `docs/STEP_19_BENCHMARK_RESULTS.json`:

| Operation | Latency | Throughput | Notes |
|---|---|---|---|
| Task Creation | $1.90\ \mu\text{s}$ | 525,817 ops/sec | Instantiates typed task container |
| Problem Decomposition | $2.95\ \mu\text{s}$ | 339,259 ops/sec | Constructs bounded subproblem tree |
| Evidence Ingestion & Evaluation | $2.18\ \mu\text{s}$ | 459,119 ops/sec | Ingestion, provenance, capacity bounding |
| Hypothesis Evaluation | $0.40\ \mu\text{s}$ | 2,485,954 ops/sec | Support/contradiction weight scoring |
| Inference Construction | $2.11\ \mu\text{s}$ | 472,978 ops/sec | Deductive premise linking |
| Contradiction Detection & Resolution | $3.72\ \mu\text{s}$ | 268,620 ops/sec | Pairwise semantic conflict check |
| Decision Candidate Selection | $2.89\ \mu\text{s}$ | 345,645 ops/sec | Net utility & risk scoring |
| Verification Loop | $1.77\ \mu\text{s}$ | 564,359 ops/sec | Multi-criteria evaluation |
| Full Governed Reasoning Cycle | $0.153\ \text{ms}$ | 6,552 cycles/sec | Complete 13-phase governed loop |

### Scaling Characteristics
* **Evidence Volume**:
  * $N=10$: $0.46\ \mu\text{s}$ ($2,194,907\text{ ops/sec}$)
  * $N=50$: $0.58\ \mu\text{s}$ ($1,717,327\text{ ops/sec}$)
  * $N=100$: $0.79\ \mu\text{s}$ ($1,259,287\text{ ops/sec}$)
* **Reasoning Depth**:
  * Depth $D=1$: $0.108\ \text{ms}$ ($9,252\text{ cycles/sec}$)
  * Depth $D=2$: $0.116\ \text{ms}$ ($8,634\text{ cycles/sec}$)
  * Depth $D=4$: $0.104\ \text{ms}$ ($9,613\text{ cycles/sec}$)

---

## 7. Known Limitations & Future Extension Points

### Known Limitations
1. **Rule-Based Decomposition**: The default `ProblemDecomposer` uses structural heuristics for mathematical, decision, and analytic tasks; dynamic open-domain decomposition relies on downstream skill templates.
2. **Pairwise Contradiction Scanning**: Contradiction detection currently scans evidence pairs in memory $O(N^2)$, which is fast for bounded $N \le 100$ but will require indexed semantic buckets for massive datasets.
3. **Synchronous Execution**: Capability execution within the reasoning action phase is synchronous.

### Future Extension Points
1. **Bayesian Evidence Fusion (Step 20+)**: Mathematical Dempster-Shafer or Bayesian updating of hypothesis confidence from continuous sensor streams.
2. **Asynchronous Background Reasoning**: Decoupled reasoning workers streaming partial hypotheses to telemetry queues.
3. **Formal Knowledge Graph Planner**: SPARQL-style graph unification for multi-hop relational deduction.
