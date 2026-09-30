# ChakrView Step 43: Architecture Specification
## Persistent Cognitive Memory, Knowledge Retrieval & Adaptive Planning Integration

**Document Version:** 1.0  
**Date:** September 30, 2026  
**Status:** ARCHITECTURAL SPECIFICATION & RATIFIED DESIGN  
**Baseline Verified:** Step 42 Ratified (Commit: `458c858`)  
**Neural Invariant:** $\Delta W = 0$, Parameters = 3,443,136, Hash = `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`

---

## 1. Motivation

Step 42 established the **Federated Cognitive Orchestration** layer (`chakrview/cognition/federation/cognitive/`), successfully constructing a causal reasoning DAG (`CognitiveTaskGraph`), a bounded context envelope (`CognitiveContextEnvelope`), governed `ChakrMicro` inference with verified $\Delta W = 0$, and anti-majority synthesis.

However, the Step 42 layer left three fundamental cognitive capabilities unintegrated:
1. **Amnestic Episodes:** Episodes currently execute in purely volatile memory. When an episode commits, all insights, synthesized evidence, and minority objections evaporate from the system. A subsequent episode addressing the same domain must start entirely from scratch.
2. **Mock Researcher Capability:** The `RESEARCHER` role (`CAPABILITY_RESEARCHER`) executes a static `SimpleCognitiveCapability` stub that emits synthetic strings, completely disconnected from ChakrView's rich Okapi BM25 document ingestion and retrieval subsystem (`chakrview/runtime/knowledge.py`).
3. **Rigid Static Planning:** Every episode, whether a 2-word factual query or an intricate system architectural analysis, is forced into an identical, hardcoded 5-step pipeline (`ANALYST` $\to$ `RESEARCHER` $\to$ `CRITIC` $\to$ `SYNTHESIZER` $\to$ `VERIFIER`), violating ChakrView's foundational **Principle of Minimum Sufficient Bounded Cognition**.

The objective of Step 43 is to resolve these gaps by integrating the battle-tested single-node subsystems built in earlier steps into the federated cognitive orchestration engine:
- **Step 24/25 Persistent Memory** (`ContinualMemoryRetriever`, `EpisodicMemoryStore`, `SemanticMemoryStore`, `ContinualMemoryStorage`)
- **Step 9/11 Knowledge Retrieval** (`BM25KnowledgeIndex`, `LexicalRetriever`, `DocumentIngester`)
- **Step 28 Adaptive Planning** (`AdaptiveTaskPlanner`, `DeterministicWorkloadClassifier`, `WorkloadClass`)

> **Primary Architectural Directive:** REUSE AND INTEGRATE EXISTING ARCHITECTURE. DO NOT REIMPLEMENT EXISTING CAPABILITIES.

---

## 2. Integrated Architectural Overview

```text
                               +-------------------------------------+
                               |           User Objective            |
                               +------------------+------------------+
                                                  |
                                                  v
                               +-------------------------------------+
                               |   DeterministicWorkloadClassifier   |
                               |      (SIMPLE .. CRITICAL)           |
                               +------------------+------------------+
                                                  |
                                                  v
                               +-------------------------------------+
                               |         AdaptiveTaskPlanner         |
                               | (Creates Dynamic Role Dependencies) |
                               +------------------+------------------+
                                                  |
                                                  v
                               +-------------------------------------+
                               | PersistentCognitiveMemoryAdapter    |
                               | - ContinualMemoryRetriever          |
                               | - Fetches Verified Semantic/Episodic|
                               +------------------+------------------+
                                                  |
                                                  v
                       +-------------------------------------------------------+
                       |               CognitiveContextEnvelope                |
                       |  - Bounded Context <= 448 Tokens, Secret-Scanned      |
                       |  - Tenant-Isolated, Pre-populated with Past Memory    |
                       +--------------------------+----------------------------+
                                                  |
                                                  v
                       +-------------------------------------------------------+
                       |           FederatedCognitiveEngine                    |
                       |            (plan_adaptive_episode)                    |
                       +--------------------------+----------------------------+
                                                  |
                                                  v
                       +-------------------------------------------------------+
                       |                  CognitiveTaskGraph                   |
                       |      (Tailored DAG with Causal Dependencies)          |
                       +--------------------------+----------------------------+
                                                  |
                       +--------------------------+----------------------------+
                       |                          |                            |
                       v                          v                            v
               [Step: ANALYST]          [Step: RESEARCHER]              [Step: CRITIC]
                                                  |
                                                  v
                               +--------------------------------------+
                               | GovernedKnowledgeRetrievalCapability |
                               | - BM25KnowledgeIndex / Documents     |
                               | - Extracted Chunks & Provenance      |
                               +------------------+-------------------+
                                                  |
                                                  v
                               +--------------------------------------+
                               |      Augmented Context Envelope      |
                               |   (Evidence tagged RETRIEVED_SOURCE) |
                               +------------------+-------------------+
                                                  |
                                                  v
                               +--------------------------------------+
                               |      [Step: SYNTHESIZER]             |
                               |  - CognitiveSynthesisEngine          |
                               |  - Minority Dissents Preserved       |
                               +------------------+-------------------+
                                                  |
                                                  v
                               +--------------------------------------+
                               |       [Step: VERIFIER]               |
                               +------------------+-------------------+
                                                  |
                                                  v
                               +--------------------------------------+
                               |   CognitiveEpisode Finalization      |
                               |   - Step 40 BFT Consensus Proposal   |
                               |   - State -> COMMITTED               |
                               +------------------+-------------------+
                                                  |
                                                  v
                               +--------------------------------------+
                               | PersistentCognitiveMemoryAdapter     |
                               | - consolidate_episode()              |
                               | - Appends to EpisodicMemoryStore     |
                               | - Extracts SemanticMemory Candidate  |
                               | - Persists to Disk (schema 24.1)     |
                               +--------------------------------------+
```

---

## 3. Four-Tier Cognitive Memory Model

To prevent memory corruption, cross-contamination, and epistemic confusion, memory is partitioned into four distinct tiers:

### Tier 1: Working Context (Ephemeral)
- **Role:** Short-lived in-flight information required for the active episode.
- **Carrier:** `CognitiveContextEnvelope`.
- **Properties:** Bounded ($\le 448$ tokens), secret-scanned, FIFO-evicted (max 15 items, max 5 hypotheses, max 10 evidence items).
- **Lifecycle:** Exists only for the duration of the episode; destroyed when the episode concludes.

### Tier 2: Episodic Memory (Persistent Historical Record)
- **Role:** Structured record of what actually occurred during past cognitive episodes.
- **Model:** `Episode` (`chakrview/memory/models.py`):
  - `situation`: Task objective.
  - `action_or_response`: Synthesized conclusion from the episode.
  - `outcome`: "COMMITTED" (or final outcome string).
  - `evidence_refs`: IDs of evidence and contributing nodes.
  - `confidence`: Confidence score in $[0.0, 1.0]$.
  - `provenance`: `MemoryProvenanceSource.SYSTEM_OBSERVED`.
  - `verification_status`: `MemoryVerificationState.UNVERIFIED`.
- **Persistence:** In-memory `EpisodicMemoryStore`, persisted to atomic JSON files via `ContinualMemoryStorage` (schema `"24.1"`).

### Tier 3: Semantic / Knowledge Memory (Persistent Declarative Knowledge)
- **Role:** Generalized factual statements and subject-predicate-object propositions extracted from verified episodes.
- **Model:** `SemanticMemory` (`chakrview/memory/models.py`):
  - `subject`, `predicate`, `object_value`.
  - `version`: Monotonic integer.
  - `previous_version_id`: Link to prior revision for non-destructive lineage.
  - `verification_status`: `CANDIDATE` (unvetted) $\to$ `VERIFIED` (vetted).
  - `contradiction_refs`: IDs of any known contradicting memories.
- **Persistence:** In-memory `SemanticMemoryStore`, persisted via `ContinualMemoryStorage`.

### Tier 4: External Retrieved Knowledge (Untrusted Passive Data)
- **Role:** Unstructured or semi-structured information retrieved from external documents during cognitive research.
- **Source:** `BM25KnowledgeIndex` / `LexicalRetriever` via `GovernedKnowledgeRetrievalCapability`.
- **Properties:** Strictly PASSIVE DATA; tagged `RETRIEVED_SOURCE`. Never possesses execution authority ($\text{EXTERNAL KNOWLEDGE} \neq \text{VERIFIED MEMORY}$).

---

## 4. Context Model & Envelope Flow

The `CognitiveContextEnvelope` serves as the sole data carrier between nodes and steps:

```text
[Episode Initialization]
        │
        ▼
PersistentCognitiveMemoryAdapter.retrieve_context(objective, tenant_id, session_id)
        │
        ├─ Query SemanticMemoryStore (trusted, active)
        └─ Query EpisodicMemoryStore (same tenant, recent)
        │
        ▼
Populate CognitiveContextEnvelope.context_items
        │
        ▼
Envelope.validate()  ──► Token Ceiling Check (<= 448 tokens)
                     ──► Secret Keyword Scan (private_key, password, etc.)
                     ──► Tenant Isolation Check (tenant_id == expected)
```

During step execution:
1. When a step completes, its outputs (conclusions, hypotheses, evidence) are formatted and appended to the envelope.
2. If total tokens exceed 448, FIFO eviction removes the oldest context items first.
3. Every step invocation re-validates the envelope before execution.

---

## 5. RAG Integration: Governed Knowledge Retrieval

To replace the static `SimpleCognitiveCapability` stub for `RESEARCHER`, Step 43 introduces `GovernedKnowledgeRetrievalCapability` in `chakrview/cognition/federation/cognitive/capabilities.py`:

### Architectural Characteristics:
1. **Registered Capability:** Registered with `CapabilityGate` under `CAPABILITY_RESEARCHER` (`"cognitive_researcher_retrieval"`).
2. **Backend Engine:** Wraps `BM25KnowledgeIndex` and `LexicalRetriever` from Step 11 (`chakrview/runtime/knowledge.py`).
3. **Execution Semantics:**
   - Accepts `CapabilityRequest` containing `parameters["objective"]` and `parameters["query"]`.
   - Executes Okapi BM25 search across indexed `KnowledgeChunk`s.
   - Extracts top $k$ chunks (default: 3) with scores.
   - Formats evidence items containing:
     - `text`: Chunk text snippet.
     - `doc_id`: Origin document ID.
     - `chunk_id`: Unique chunk ID.
     - `content_hash`: Cryptographic SHA-256 digest of content.
     - `score`: BM25 score.
     - `provenance`: `MemoryProvenanceSource.RETRIEVED_SOURCE`.
4. **Safety Enforcement:**
   - Scans retrieved text for credential leaks (`PROHIBITED_CONTEXT_KEYWORDS`).
   - Caps evidence tokens within the envelope budget.
   - Treats all retrieved text as passive data ($\text{DATA} \neq \text{AUTHORITY}$).

---

## 6. Adaptive Planning Integration

In Step 42, `_build_standard_graph()` constructed a fixed 5-step graph. In Step 43, `FederatedCognitiveEngine` integrates Step 28's `AdaptiveTaskPlanner`:

### Workload Classification & Dynamic Topology:
1. `DeterministicWorkloadClassifier.classify_workload(objective, context)` determines the `WorkloadClass`:
   - `SIMPLE`: 1 step (`ANALYST`). Fast-path execution for simple definitions or single-turn questions.
   - `STANDARD`: 3 steps (`ANALYST`, `RESEARCHER`, `SYNTHESIZER`). `SYNTHESIZER` depends on `ANALYST` and `RESEARCHER`.
   - `COMPLEX`: 5 steps (`ANALYST`, `RESEARCHER`, `CRITIC`, `SYNTHESIZER`, `VERIFIER`). Structured causal dependencies with adversarial critique.
   - `AMBIGUOUS`: 4 steps (`ANALYST`, `RESEARCHER`, `CRITIC`, `SYNTHESIZER`). Focused on hypothesis formulation and resolution.
   - `CRITICAL`: 5 steps with strict verification gates and lower tolerance for ambiguity.

2. `AdaptiveTaskPlanner.plan_task()` emits a `TaskPlan` with explicit `role_dependencies`.

3. `plan_adaptive_episode()` translates `TaskPlan.role_dependencies` into a valid `CognitiveTaskGraph`:
   - Maps each role to a `CognitiveStep` with the appropriate `capability_id`.
   - Wires dependencies so steps execute strictly when prerequisites complete.
   - Validates the DAG for cycles (`_detect_cycles()`).

---

## 7. Memory Consolidation Lifecycle

When a `CognitiveEpisode` reaches completion:

```text
All DAG Steps Completed
        │
        ▼
CognitiveSynthesisEngine.synthesize_results()
  - Synthesizes consensus conclusion
  - Preserves minority dissents as ConflictRecord
        │
        ▼
Propose BFT Consensus (Step 40)
  - Records consensus_proposal_id
        │
        ▼
Episode State: FINALIZING ──► COMMITTED
        │
        ▼
PersistentCognitiveMemoryAdapter.consolidate_episode(episode)
  │
  ├─ 1. Create Episode Record:
  │     - situation: episode.objective
  │     - action_or_response: synthesis.synthesized_conclusion
  │     - outcome: "COMMITTED"
  │     - evidence_refs: participating_nodes + evidence hashes
  │     - confidence: 0.85
  │     - verification_status: UNVERIFIED
  │     - Store in EpisodicMemoryStore
  │
  ├─ 2. Create Semantic Proposition (if confident):
  │     - subject: objective entity
  │     - predicate: "has_conclusion"
  │     - object_value: synthesis.synthesized_conclusion[:120]
  │     - verification_status: CANDIDATE
  │     - Store in SemanticMemoryStore
  │
  └─ 3. Atomic Disk Sync:
        - ContinualMemoryStorage.save_to_file(storage_path, state)
        - Schema version "24.1" validated
```

---

## 8. Federation Boundaries & Consistency Model

1. **Local Sovereignty:** Each federated node maintains its own sovereign `ContinualMemoryStorage`. Remote nodes **never** write directly to local memory.
2. **Wire Protocol Decoupling:** Remote nodes communicate strictly via `CognitiveStep` work units and BFT consensus proposals. Raw memory tables never cross the wire.
3. **Consensus Invariant:** BFT consensus agrees upon the *finalization order of episodes*, never on capability authorization ($\text{CONSENSUS} \neq \text{AUTHORITY}$, $\text{LOCAL\_POLICY} > \text{CONSENSUS\_DECISION}$).
4. **Partition Tolerance:** During network partitions, nodes proceed with local memory retrieval and local fallback capabilities; consensus is recorded as `local_only`.

---

## 9. Invariants Enforced in Step 43

| Axiom / Invariant | Implementation Mechanism |
|---|---|
| $\text{LOCAL\_POLICY} > \text{CONSENSUS\_DECISION}$ | `CapabilityGate` authorizes all capability executions locally regardless of BFT consensus. |
| $\text{CONSENSUS} \neq \text{AUTHORITY}$ | Consensus proposals certify episode completion; they grant zero capabilities. |
| $\text{DATA} \neq \text{AUTHORITY}$ | Episode conclusions and memory items are data; never executed as code. |
| $\text{MEMORY} \neq \text{AUTHORITY}$ | Retrieved memories provide evidence; never sovereign authorization. |
| $\text{EXTERNAL KNOWLEDGE} \neq \text{VERIFIED MEMORY}$ | RAG chunks remain passive untrusted data tagged `RETRIEVED_SOURCE`. |
| $\text{ZERO SECRET EXPOSURE}$ | `CognitiveContextEnvelope._scan_for_secrets` blocks keywords. |
| $\text{ZERO NEURAL WEIGHT MUTATION}$ | Pre/post-flight SHA-256 weight hash verification enforces $\Delta W = 0$. |
| $\text{TENANT ISOLATION}$ | All queries, envelopes, and stores strictly keyed and validated by `tenant_id`. |

---

## 10. Non-Goals

1. **Autonomous Internet Scraping:** RAG retrieval operates strictly over explicitly indexed local documents or provided `KnowledgeIndex` instances.
2. **Vector DB Dependencies:** The architecture remains 100% CPU-only and self-contained; no Pinecone, Milvus, Chroma, or external vector service dependencies.
3. **Online Model Fine-Tuning:** Neural weights remain immutable ($\Delta W = 0$); memory and continual learning are achieved purely at the context and retrieval layers without parameter updates.
