# ChakrView — Step 18: Cognitive Identity, Self-Model & System State Foundation

## 1. Executive Summary & Motivation

In previous architectural steps, ChakrView established:
* **Step 01–10**: Generative core model (`ChakrMicro`, 3.44M parameters, frozen context length 512, vocab 4096), tokenizer, pretraining, alignment, and evaluation foundations.
* **Step 11–13**: Lexical and hybrid semantic retrieval (BM25 + deterministic hash embeddings).
* **Step 14**: Sovereign neural semantic encoder (`ChakrEncoder`) foundation.
* **Step 15**: Governed cognitive agent execution and sandboxed tool orchestration (`CapabilityGate`).
* **Step 16**: Persistent personal memory, semantic consolidation, and controlled learning.
* **Step 17**: Hardware-agnostic capability abstraction foundation (`CapabilityRegistry`, `CapabilityInterface`, `EnvironmentProfile`).

ChakrView is envisioned as an indigenous, modular cognitive AI platform whose reusable core brain can power diverse physical edge devices, robots, vehicles, desktop applications, and server nodes. However, raw neural inference is stateless and probabilistic. An autonomous cognitive system cannot rely on probabilistic hallucination or unconstrained model prompt context to maintain its identity, track active task lifecycles, understand its operating environment, distinguish between known and unknown data, or honor operational constraints.

**Step 18** establishes the **Cognitive State Layer**: an explicit, strongly typed, inspectable, deterministic software subsystem between the cognitive controller and the outside world.

> **CRITICAL ARCHITECTURAL PRINCIPLE**:
> The Cognitive State layer is **NOT** a second hidden neural network, nor does it create consciousness, sentience, self-awareness, or human-like cognition.
> It is an explicit, machine-readable, auditable, and mathematically verifiable state representation system.

```
                CHAKRVIEW CORE BRAIN
                     ChakrMicro
                        │
                        ▼
                Cognitive Controller
                        │
                        ▼
             ┌──────────────────────┐
             │   Cognitive State    │
             │      Layer           │
             ├──────────────────────┤
             │ System Identity      │
             │ Environment State    │
             │ Capability State     │
             │ Task State           │
             │ Knowledge State      │
             │ Memory Context       │
             │ Uncertainty State    │
             │ Policy / Constraint  │
             └──────────┬───────────┘
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
       Memory       Capabilities    Knowledge
          │             │             │
          └─────────────┼─────────────┘
                        ▼
                 External World
```

---

## 2. Core Architectural Principles & Invariants

1. **Frozen Generative Core**: `ChakrMicro` remains untouched (3,443,136 parameters, vocabulary size 4,096, context length 512, BOS=0, EOS=1, PAD=2).
2. **DATA != AUTHORITY**: No serialized state, memory record, knowledge assertion, capability observation, or environment parameter can grant execution authority or bypass security gates.
3. **CapabilityGate Remains Sovereign**: Observing that a capability is `AVAILABLE` does not grant permission to invoke it. All invocation passes through `CapabilityGate`.
4. **Deterministic & Inspectable**: All state transitions are deterministic, typed, timestamped, versioned, and inspectable.
5. **No External Cloud / Model Dependencies**: Completely independent of OpenAI, Gemini, Claude, Llama, Mistral, or any external API.
6. **Multi-Tenant Isolation**: State stores are segregated strictly by `(owner_id, session_id, environment_id)`.
7. **Audit Preservation on Rollback**: State rollbacks never erase history; they execute as a forward state transition referencing the prior snapshot.

---

## 3. Strict Epistemic Taxonomy

A critical flaw in standard AI systems is conflating missing knowledge with falsehood, or unreachable sensors with negative values. ChakrView enforces strict four-valued epistemic logic and explicit operational status:

| Status / Distinction | Formal Meaning | Counter-Example Violation |
|---|---|---|
| `UNKNOWN != FALSE` | System has no verifiable assertion regarding predicate $P$. | Treating absence of a record as proof that the record is false. |
| `UNAVAILABLE != UNKNOWN` | Sensor, device, or capability cannot be reached or queried. | GPS sensor being offline interpreted as "no location exists". |
| `DISABLED != UNAVAILABLE` | Subsystem is reachable but administratively locked by policy. | Camera administratively disabled interpreted as hardware disconnected. |
| `CONFLICTING` | Multiple verified assertions provide contradictory values. | Silently picking the last assertion without recording epistemic conflict. |
| `STALE` | Assertion was valid in the past but has exceeded its temporal horizon ($T_{valid}$). | Assuming temperature reading from 12 hours ago is current. |

### Formal Epistemic Enumeration (`EpistemicStatus`)
* `KNOWN`: Verifiable assertion exists with known confidence and provenance.
* `UNKNOWN`: Explicit query revealed no assertions in working knowledge.
* `UNCERTAIN`: Assertion exists but uncertainty exceeds threshold ($\mu \ge 0.5$).
* `CONFLICTING`: Contradictory assertions exist without definitive resolution.
* `STALE`: Time-to-live / temporal validity expired ($t > t_{valid\_until}$).
* `UNAVAILABLE`: Underlying source / sensor is unreachable or disconnected.

---

## 4. Subsystem Components (`chakrview/state/`)

### 4.1. System Identity (`identity.py`)
Provides an immutable runtime identity representation:
* `system_id`: Canonical identifier (default: `"chakrview-core"`).
* `software_version`: `"0.18.0"`.
* `architecture_version`: `"chakrview-v1"`.
* `model_version`: `"chakrmicro-3.44m"`.
* `tokenizer_version`: `"spm-4096-bpe"`.
* `parameter_count`: Exactly `3,443,136`.
* `vocabulary_size`: Exactly `4,096`.
* `context_horizon`: Exactly `512`.
* `deployment_environment`: `"local-sovereign"`.
* `policy_profile`: Active security profile (e.g. `"standard-governed"`).

Validation strictly asserts that parameter count, vocabulary size, and context horizon match frozen invariants.

### 4.2. Epistemic Knowledge State (`epistemic.py`)
Tracks structured assertions with cryptographic provenance:
* `KnowledgeAssertion`:
  * `assertion_id`: Unique deterministic or generated UUID.
  * `subject`, `predicate`, `value`: Semantic triple.
  * `epistemic_status`: `EpistemicStatus` enum.
  * `source`: Provenance source identifier.
  * `confidence`: Float $[0.0, 1.0]$.
  * `timestamp`: Unix timestamp of creation.
  * `valid_until`: Optional expiration timestamp.
  * `provenance`: Dict with metadata, signature, or reference hash.
* `KnowledgeState`:
  * Fast indexing by `(subject, predicate)`.
  * Multi-assertion query with automatic conflict and staleness detection.
  * Explicit `query(subject, predicate)` returning `(status, assertions)`.

### 4.3. Explicit Uncertainty Representation (`uncertainty.py`)
Eliminates fabricated confidence scores:
* `Uncertainty`:
  * `uncertainty_id`: Unique identifier.
  * `topic`: Topic or target attribute.
  * `confidence`: Float $[0.0, 1.0]$ or `None` if uncalibrated.
  * `uncertainty_score`: Float $[0.0, 1.0]$.
  * `reason`: Explicit machine-readable justification for uncertainty.
  * `source`: Origin of uncertainty signal (e.g. `"sensor_dropout"`, `"epistemic_gap"`).
  * `is_uncalibrated`: Flag indicating system has no basis for calculating confidence.
  * `evidence_refs`: List of evidence references.
* `UncertaintyState`:
  * Collection of active uncertainties across topics.
  * Query by topic with threshold filtering.

### 4.4. Task State Machine (`task_state.py`)
Tracks active cognitive tasks and multi-step executions:
* `TaskPhase`: `IDLE`, `PLANNING`, `EXECUTING`, `OBSERVING`, `VERIFYING`, `RECOVERING`, `COMPLETED`, `FAILED`.
* `TaskState`:
  * `task_id`, `goal`, `status`, `parent_task_id`, `current_phase`.
  * `required_capabilities`: List of capability names needed.
  * `observations`: Chronological log of observations.
  * `decisions`: Chronological log of decisions made.
  * `failures`: Detailed log of any encountered failures.
  * `completion_state`: Result payload upon task termination.

### 4.5. Environment State (`environment_state.py`)
Maintains an accurate observation of the physical/virtual host environment:
* Integrates seamlessly with Step 17 `EnvironmentProfile`.
* `OperationalMode`: `NOMINAL`, `DEGRADED`, `DIAGNOSTIC`, `RESTRICTED`, `EMERGENCY_STOP`.
* `DeviceConnectionStatus`: `CONNECTED`, `DISCONNECTED`, `DEGRADED`, `UNKNOWN`.
* Tracks resources (CPU, RAM, GPU, storage), network connectivity, safety interlocks, and connected hardware devices.

### 4.6. Capability State (`capability_state.py`)
Integrates with Step 17 `CapabilityRegistry` to provide an inspectable view of capabilities:
* `ObservedCapabilityStatus`: `AVAILABLE`, `UNAVAILABLE`, `DISABLED`, `UNKNOWN`, `FAULTED`.
* `CapabilityObservation`: Observation of a capability's operational status and risk tier.
* `CapabilityState`: Map of capability IDs to observed status.
* **Security Guardrail**: Capability observations are data-only and cannot trigger or authorize capability calls.

### 4.7. Constraint & Policy State (`constraints.py`)
Tracks active operational policies and boundaries:
* `PolicyRestriction`: Rule specifying forbidden capabilities, allowed risk tiers, execution timeouts, and resource quotas.
* `ConstraintState`: Evaluates whether a candidate capability action is compliant with active policy restrictions.

### 4.8. Snapshot & Differential Engine (`snapshot.py`)
* `CognitiveStateSnapshot`:
  * Immutable snapshot of identity, environment, capabilities, tasks, knowledge, uncertainty, constraints, and active memory context.
  * Monotonically increasing `version` and `snapshot_id`.
  * Pure JSON serialization (`to_dict()`, `from_dict()`, `to_json()`, `from_json()`).
  * Explicitly prohibits `pickle` to guarantee corruption-safe and deserialization-safe operation.
* `SnapshotDiff` & `compare_snapshots()`:
  * Computes deep diffs between two snapshots (version changes, tasks added/modified/removed, assertions added/removed, capability status changes, environment operational mode shifts).

### 4.9. State Manager (`manager.py`)
* `CognitiveStateManager`:
  * Multi-tenant boundary enforcing `owner_id`, `session_id`, `environment_id`.
  * Monotonic version tracking and immutable snapshot history.
  * Non-destructive `rollback(target_snapshot_id)`:
    * Preserves complete audit trail.
    * Emits `ROLLBACK_TRANSITION` audit event.
    * Bumps state version forward (e.g. state at version 3 rolled back to snapshot at version 1 becomes version 4).
    * Takes a new snapshot capturing the restored state under the new version.

---

## 5. Security & Isolation Architecture

### 5.1. DATA != AUTHORITY Principle
Under Step 15 and Step 17, `CapabilityGate` enforces capability authorization. Step 18 ensures that:
1. `CapabilityGate.authorize()` explicitly rejects requests whose context claims authority via `state_knowledge_assertion`, `state_knowledge`, or `knowledge_state`.
2. Knowledge assertions cannot grant permissions:
   ```python
   # Attempting to forge capability authorization via a knowledge assertion
   assertion = KnowledgeAssertion(
       subject="motor_control",
       predicate="can_invoke",
       value=True,
       epistemic_status=EpistemicStatus.KNOWN,
       confidence=1.0,
       provenance={"authority": "admin"}
   )
   manager.assert_knowledge(assertion)
   # CapabilityGate still requires explicit security token from AuthorizedContext;
   # state observation CANNOT bypass the gate!
   ```

### 5.2. Multi-Tenant Segregation
Cross-owner or cross-session state access throws `StateIsolationError`. State instances are isolated per tenant session.

### 5.3. Safe Structured Serialization
Persistent snapshots use standard JSON. The system strictly avoids `pickle`, preventing arbitrary code execution exploits.

---

## 6. Empirical Performance Benchmark

The benchmark script `scripts/benchmark_state.py` evaluated the performance of all core state operations on a standard CPU test environment. Results are archived in `docs/STEP_18_BENCHMARK_RESULTS.json`.

| Operation | Latency | Throughput | Notes |
|---|---|---|---|
| State Manager Creation | $7.55\ \mu\text{s}$ | 132,367 ops/sec | Instantiates full multi-tenant state container |
| Knowledge Assertion | $6.88\ \mu\text{s}$ | 145,255 ops/sec | Includes semantic indexing & validation |
| Task State Transition | $1.65\ \mu\text{s}$ | 607,599 ops/sec | In-memory phase and decision logging |
| Uncertainty Recording | $4.64\ \mu\text{s}$ | 215,300 ops/sec | Explicit justification tracking |
| Snapshot Creation | $1.858\ \text{ms}$ | 538 snaps/sec | Deep copy of all subsystem states |
| Snapshot Comparison (Diff) | $51.85\ \mu\text{s}$ | 19,288 compares/sec | Structural comparison of two full snapshots |
| Snapshot JSON Serialization | $1.389\ \text{ms}$ | 719 snaps/sec | Export to 37.1 KB UTF-8 JSON |
| Snapshot JSON Deserialization | $0.783\ \text{ms}$ | 1,277 snaps/sec | Validated parse into immutable snapshot |
| Non-destructive Rollback | $3.493\ \text{ms}$ | 286 rollbacks/sec | Restores state, logs audit transition, captures new snapshot |
| Knowledge Query | $427.05\ \mu\text{s}$ | 2,341 queries/sec | Full scan with conflict/staleness evaluation |

### Scaling Characteristics
* $N=10$ items: Snapshot $0.482\ \text{ms}$, JSON $0.368\ \text{ms}$, Payload $8.2\ \text{KB}$
* $N=100$ items: Snapshot $4.742\ \text{ms}$, JSON $2.590\ \text{ms}$, Payload $68.1\ \text{KB}$
* $N=1,000$ items: Snapshot $34.107\ \text{ms}$, JSON $23.181\ \text{ms}$, Payload $669.4\ \text{KB}$

---

## 7. Known Limitations & Future Extension Points

### Known Limitations
1. **In-Memory Volatility**: The default `CognitiveStateManager` operates in-memory; disk persistence requires calling `snapshot.to_json()` to a persistent store.
2. **Knowledge Graph Complexity**: Epistemic state supports semantic triples (`subject`, `predicate`, `value`), but does not yet implement a full SPARQL or graph database query planner.
3. **Static Conflict Resolution**: Conflicting assertions are flagged as `CONFLICTING` without automated probabilistic Bayesian belief revision.

### Future Extension Points
1. **Epistemic Belief Revision (Step 19+)**: Formal Bayesian / Dempster-Shafer evidence fusion when multiple sensor observations conflict.
2. **Persistent State Store Adapter**: SQLite / RocksDB backed storage adapter for edge device state preservation across power cycles.
3. **Cognitive Controller Binding**: Direct integration into the agent loop so `ChakrMicro` attention context dynamically receives structured state summaries.
