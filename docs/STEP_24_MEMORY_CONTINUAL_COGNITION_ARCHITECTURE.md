# CHAKRVIEW — STEP 24 ARCHITECTURAL SPECIFICATION
## Governed Memory, Experience & Continual Cognition Foundation

---

## 1. Executive & Architectural Overview

Step 24 establishes a sovereign, modular **Continual Cognition and Governed Memory Subsystem** for ChakrView. It enables the system to maintain short-term working context, capture structured episodic experiences, curate versioned semantic knowledge, detect and track factual contradictions, perform deterministic multi-factor memory retrieval, and safely bridge verified declarative memories into candidate training records for the Step 22 offline learning pipeline.

### Core Architectural Diagram

```text
                    ┌──────────────────────┐
                    │     User / Task      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Cognitive State    │
                    └──────────┬───────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │       Memory System            │
              │                                │
              │ Working Memory (Bounded)       │
              │ Episodic Memory (Experiences)  │
              │ Semantic Memory (Versioned)    │
              │ Multi-Factor Retrieval         │
              │ Contradiction Handling         │
              │ Experience Consolidation       │
              └───────────────┬────────────────┘
                              │ Context & Evidence
                              ▼
                 ┌────────────────────────┐
                 │ Thinking / Critical    │
                 │ Thinking / Reasoning   │
                 └────────────┬───────────┘
                              │ Prompt & Context
                              ▼
                       ┌────────────┐
                       │ ChakrMicro │
                       │ 3.44M Core │  (Read-Only Runtime Weights)
                       └────────────┘

Experience (Runtime)
    │
    ▼
Episodic Store
    │
    ▼
Candidate Semantic Memory
    │
    ▼
Verification & Governance
    │
    ▼
Governed Learning Candidate
    │
    ▼
Step 22 Offline Dataset Builder
    │
    ▼
Step 22 Offline Training
    │
    ▼
Validation & Regression Gate
    │
    ▼
Explicit Administrative Promotion
```

---

## 2. Absolute Invariants & Security Boundaries

### Frozen Neural Core Invariants
- **Parameter Count**: Exactly `3,443,136` unique weights (weight-tied).
- **Vocabulary Size**: `4,096` tokens.
- **Maximum Context Length**: `512` tokens.
- **Special Tokens**: $\text{BOS}=0$, $\text{EOS}=1$, $\text{PAD}=2$.
- **Runtime Immutability**: Neural weights remain strictly read-only (`weights_modified == False`). No runtime weight updates are mathematically possible during inference, memory insertion, or retrieval.

### Authority Axioms
```text
DATA != AUTHORITY
MEMORY != AUTHORITY
EXPERIENCE != AUTHORITY
REASONING != AUTHORITY
THINKING != AUTHORITY
CRITICAL THINKING != AUTHORITY
NEURAL OUTPUT != AUTHORITY
```
Memories provide contextual evidence and factual observations; they **cannot independently authorize capability execution** or bypass the `CapabilityGate`. Any capability request originating from `continual_memory`, `episodic_memory`, `semantic_memory`, or `experience` is denied by `CapabilityGate` with a `CapabilityAuthorizationError`.

---

## 3. Memory Subsystem Taxonomy & Models

### A. Working Memory (`chakrview/memory/working.py`)
Session- and task-scoped temporary scratchpad maintaining active cognitive context:
- `objective`: Primary task goal.
- `active_context`: Recent prompt and document excerpts.
- `current_hypotheses`: Active candidate explanations.
- `relevant_evidence`: Associated empirical facts.
- `intermediate_decisions`: Incremental choices during multi-step execution.
- `active_constraints`: Task boundaries and negative constraints.
- `pending_questions`: Epistemic gaps requiring further resolution.
- `recent_observations`: High-frequency environment outcomes.

**Bounded Capacity**: Strictly enforces FIFO eviction when capacity is reached, preventing unbounded memory growth. Capacity dynamically scales with the active `MemoryExecutionPolicy`.

### B. Episodic Memory (`chakrview/memory/episodic.py`)
Records interaction events as they occurred, explicitly separating ground observations from interpretation:
- `episode_id`: Unique identifier (`ep_...`).
- `tenant_id` & `session_id`: Strict isolation keys.
- `situation`: Task input / environment state.
- `action_or_response`: Output produced by the system.
- `outcome`: Observed result or validation check.
- `evidence_refs`: Cited source documents or assertions.
- `confidence`: Assessment certainty in $[0.0, 1.0]$.
- `provenance`: Origin citation (`MemoryProvenanceSource`).
- `verification_status`: Lifecycle status (`MemoryVerificationState`).

### C. Semantic Memory (`chakrview/memory/semantic.py`)
Versioned declarative knowledge structured as subject-predicate-object propositions:
- `memory_id`: Unique identifier (`sem_...`).
- `subject`: Entity or concept.
- `predicate`: Relationship or property.
- `object_value`: Value or state.
- `version`: Monotonically increasing revision counter ($1, 2, \dots$).
- `previous_version_id`: Link to superseded record.
- **Non-Destructive Revision**: Updates archive previous records and link ancestors rather than destroying history.

---

## 4. Provenance & Verification State Machine

### Provenance Taxonomy
```text
USER_PROVIDED             Explicit user prompt or direct assertion
SYSTEM_OBSERVED           Direct sensor or execution environment telemetry
CAPABILITY_RESULT         Output from authorized capability invocation
REASONING_DERIVED         Derived via governed deductive logic
CRITICAL_THINKING_DERIVED Synthesized via hypothesis falsification & counter-evidence
RETRIEVED_SOURCE          Grounded document extraction
TRAINING_APPROVED         Signed off for offline model dataset inclusion
UNKNOWN                   Uncertain origin fallback
```

### Verification States
```text
CANDIDATE     Consolidated proposition, unvetted (excluded from trusted retrieval)
UNVERIFIED    Raw recorded state pending formal checks
VERIFIED      Formally validated against external ground truth or policy
CONTRADICTED  Active contradiction detected with another stored proposition
QUARANTINED   Suspected prompt injection, safety risk, or corrupted state
ARCHIVED      Superseded or retired from active working view
REJECTED      Refuted by verification checks or administrative directive
```

---

## 5. Deterministic Memory Retrieval

Memory retrieval operates entirely on CPU without requiring external vector databases.

### Multi-Factor Scoring Formula
$$\text{Score} = \text{Relevance} \times \text{Confidence} \times \text{VerificationFactor} \times \text{RecencyFactor} \times \text{ContradictionFactor}$$

Where:
- **Relevance**: Lexical token overlap between query and memory content normalized by query length, plus exact phrase bonus.
- **Confidence**: Record-level certainty $c \in [0.0, 1.0]$.
- **VerificationFactor**:
  - `VERIFIED`: $1.0$
  - `UNVERIFIED`: $0.6$
  - `CONTRADICTED`: $0.3$
  - `ARCHIVED`: $0.2$
  - `CANDIDATE`, `QUARANTINED`, `REJECTED`: $0.0$ (strictly rejected from trusted retrieval).
- **RecencyFactor**: Exponential decay $e^{-\lambda \Delta t}$ with 30-day half-life, clamped to $[0.2, 1.0]$.
- **ContradictionFactor**: $1.0$ if uncontradicted; $0.5$ if active unresolved conflict exists.

---

## 6. Contradiction Detection & Non-Destructive Resolution

### Principles
When a newly asserted fact conflicts with an existing memory (e.g., `Revenue = 20 lakh` vs `Revenue = 32 lakh`):
1. **No Silent Replacement**: Neither memory is deleted.
2. **Contradiction Record**: A structured `MemoryContradiction` is instantiated (`UNRESOLVED`).
3. **State Downgrade**: Conflicting memories have their status updated to `CONTRADICTED`, penalizing retrieval rank.
4. **Governed Resolution**: Through administrative sign-off or verified evidence, status transitions to `RESOLVED` (selecting the winning record and marking the other `REJECTED`) or `PERSISTENT_CONFLICT`.

---

## 7. Experience Consolidation Protocol

```text
Experience (Task Interaction)
       │
       ▼
Episodic Store (Structured Episode)
       │
       ▼
Consolidation Engine (Pattern Extraction & Clustering)
       │
       ▼
Contradiction Check (Detect Semantic Divergence)
       │
       ▼
Candidate Semantic Memory (status = CANDIDATE)
       │
       ▼
Evaluation / Verification Gate
       │
       ▼
Verified Semantic Memory (status = VERIFIED)
```

Consolidation evaluates recurring observations across active episodes. It creates **CANDIDATE** semantic memories. It is strictly forbidden from directly updating neural network parameters.

---

## 8. Step 22 Offline Training Integration

### Self-Improvement Pipeline
The continual cognition memory system safely interfaces with the Step 22 CPU training pipeline:
```text
Verified Semantic Memory (status == VERIFIED)
       │
       ▼
MemoryGovernanceBridge.create_learning_candidate(...)
       │
       ▼
LearningRecord (status == VERIFIED)
       │
       ▼
Administrative Review / Explicit Sign-Off
       │
       ▼
LearningRecord (status == TRAINING_APPROVED)
       │
       ▼
Step 22 TrainingDatasetBuilder (Manifest & Tokenization)
       │
       ▼
Step 22 Offline Trainer (CPU AdamW, Cosine Annealing)
       │
       ▼
Step 22 Regression Gate & Evaluator
       │
       ▼
Explicit Manual Promotion
```

At no point does runtime experience or memory directly alter model weights.

---

## 9. Hardware Adaptation Matrix

Memory operations automatically adapt to available host resources via `MemoryExecutionPolicy`:

| Parameter | LOW_RESOURCE | STANDARD | HIGH_RESOURCE | Hard Safety Ceiling |
| :--- | :---: | :---: | :---: | :---: |
| **Working Memory Capacity** | 10 items | 30 items | 60 items | **100 items** |
| **Max Retrieval Candidates** | 3 candidates | 8 candidates | 15 candidates | **30 candidates** |
| **Storage Scan Limit** | 50 records | 200 records | 400 records | **1000 records** |
| **Contradiction Depth** | 5 items | 15 items | 30 items | **50 items** |
| **Consolidation Batch Size** | 2 items | 5 items | 10 items | **20 items** |

Hard ceilings guarantee that no host environment can trigger unbounded memory allocation or infinite processing loops.

---

## 10. Multi-Tenant & Multi-Session Isolation

1. **Storage Indexing**: All episodic and semantic records are partitioned strictly by `tenant_id`.
2. **Access Control**: Queries without matching `tenant_id` return zero results. Cross-tenant leakage is mathematically impossible at the storage layer.
3. **Session Scoping**: Working memory and default episodic queries isolate execution by `session_id` unless `include_cross_session=True` is explicitly authorized within the same tenant.

---

## 11. Controlled Forgetting & Lifecycle Management

- **Retention**: Active memories remain valid within the working store.
- **Expiration**: Sweeps automatically transition records exceeding `max_age_seconds` to `EXPIRED`.
- **Archival**: Outdated or superseded records transition to `ARCHIVED` while preserving provenance links.
- **Quarantine**: Suspect, contaminated, or injected inputs are isolated in `QUARANTINED` status and removed from retrieval and training candidate queues.
- **Governed Deletion**: Explicit user deletion requests mark records `DELETED` and append an entry to the tamper-evident `LifecycleAuditEntry` log.

---

## 12. Implemented vs. Future Capabilities

### Implemented & Empirically Verified in Step 24
- [x] Bounded Working Memory with FIFO eviction.
- [x] Structured Episodic Memory separating situations, actions, and outcomes.
- [x] Versioned Semantic Memory propositions with revision lineage.
- [x] Multi-factor deterministic retrieval scoring (Relevance, Confidence, Verification, Recency, Contradiction).
- [x] Automated semantic contradiction detection and formal resolution lifecycles.
- [x] Continual experience consolidation pipeline.
- [x] Safe Step 22 learning candidate bridge requiring explicit verification and approval.
- [x] Controlled lifecycle management (archival, expiration sweeps, quarantine, audit logging).
- [x] Hardware execution policies (`LOW_RESOURCE`, `STANDARD`, `HIGH_RESOURCE`) with hard safety ceilings.
- [x] Strict tenant and session privacy isolation.
- [x] Deterministic schema validation (`schema_version: "24.1"`).
- [x] Full backward compatibility with Step 16 persistent personal memory and all Steps 0–23.

### Future Capabilities (Deferred to Subsequent Steps)
- [ ] Distributed multi-node memory replication.
- [ ] Cross-tenant governed memory federation with cryptographic proofs.
- [ ] Dynamic ontology restructuring and hypergraph relation mining.
- [ ] Continual token vocabulary expansion through offline token merge evaluation.

---

## 13. Known Limitations

1. **Syntactic Proposition Extraction**: Automatic pattern extraction during consolidation relies on deterministic grammatical heuristics; complex multi-clause open-domain relations rely on structured reasoning passes.
2. **Synchronous Consolidation Execution**: Consolidation sweeps execute synchronously within the engine thread context.
3. **CPU-First Lexical Retrieval**: Memory retrieval uses fast token-overlap and substring scoring; while lightweight and portable, it does not perform dense vector indexing.
