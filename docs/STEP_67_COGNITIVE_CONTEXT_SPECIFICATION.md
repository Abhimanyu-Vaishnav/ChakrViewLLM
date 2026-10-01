# Step 67: Unified Cognitive Context Specification

## 1. Overview & Architecture

Step 67 establishes the unified cognitive composition layer between Step 64 repository grounding / Step 66 episodic memory recall and the downstream passive `NeuralProposalAdapter`.

```
                    ┌───────────────────────────────┐
                    │    CognitiveContextRequest    │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                 ┌──────────────────────────────────────┐
                 │   UnifiedCognitiveContextComposer    │
                 └──────────────┬────────────────┬──────┘
                                │                │
       ┌────────────────────────┴─┐            ┌─┴────────────────────────┐
       │ GroundedContextRetriever │            │ EpisodicMemoryRecall     │
       │ (Step 64 Static Evidence)│            │ Coordinator (Step 66)    │
       └──────────────────────────┘            └──────────────────────────┘
                                │                │
                                └────────┬───────┘
                                         ▼
                 ┌──────────────────────────────────────┐
                 │       Conflict & Budget Filter       │
                 │   - Deep Conflict Arbitration        │
                 │   - Explicit Task Constraints        │
                 │   - Budget Limits (Evidence/Memory)  │
                 │   - Deterministic Key Ordering       │
                 └───────────────────────┬──────────────┘
                                         ▼
                        ┌─────────────────────────────────┐
                        │      CognitiveContextBundle     │
                        │  - positive_evidence            │
                        │  - positive_memories            │
                        │  - negative_boundaries          │
                        │  - conflicted_items (quarantine)│
                        │  - excluded_items               │
                        └────────────────┬────────────────┘
                                         │ .to_neural_context()
                                         ▼
                        ┌─────────────────────────────────┐
                        │       Passive Neural Context    │
                        │   (Read-Only Dict Payload)      │
                        └────────────────┬────────────────┘
                                         │
                                         ▼
                        ┌─────────────────────────────────┐
                        │     NeuralProposalAdapter       │
                        │     (Proposal-Only Boundary)    │
                        └─────────────────────────────────┘
```

## 2. Core Typed Structures

### `CognitiveContextRequest`
- `task_description: str`
- `task_family: str`
- `target_files: Tuple[str, ...]`
- `target_symbols: Tuple[str, ...]`
- `allowed_files: Tuple[str, ...]`
- `explicit_constraints: Tuple[str, ...]`
- `repository_fingerprint: Optional[str]`
- `dependency_signature: Optional[str]`
- `budget: CognitiveContextBudget`

### `CognitiveContextBudget`
- `max_evidence_items: int = 15`
- `max_positive_memories: int = 3`
- `max_negative_boundaries: int = 3`
- `max_total_items: int = 25`
- `max_symbols: int = 20`
- `max_modules: int = 10`

### `CognitiveContextItem`
- `item_id: str`
- `source_type: CognitiveContextSource` (`REPOSITORY_EVIDENCE`, `EPISODIC_MEMORY`, `NEGATIVE_BOUNDARY`, `TASK_CONSTRAINT`)
- `source_identifier: str`
- `module_reference: Optional[str]`
- `symbol_reference: Optional[str]`
- `evidence_fingerprint: str`
- `status: CognitiveContextStatus` (`ACTIVE`, `NEGATIVE`, `CONFLICTED`, `STALE`, `SUPERSEDED`, `UNAVAILABLE`, `ABSTAIN`)
- `deterministic_reason: str`
- `epistemic_state: EpistemicState`
- `originating_episode: Optional[str]`
- `content_payload: str`
- `ordering_key: str`

### `CognitiveContextBundle`
- `request: CognitiveContextRequest`
- `repository_fingerprint: str`
- `positive_evidence: List[CognitiveContextItem]`
- `positive_memories: List[CognitiveContextItem]`
- `negative_boundaries: List[CognitiveContextItem]`
- `conflicted_items: List[CognitiveContextItem]`
- `excluded_items: List[CognitiveContextItem]`
- `active_symbols: List[str]`
- `active_modules: List[str]`
- `abstained: bool`
- `abstain_reason: Optional[str]`
- `telemetry: CognitiveContextTelemetry`

## 3. Invariants & Rules

1. **Deterministic Ordering**: Context items are ordered strictly using stable keys (e.g. `1_ev:<source_type>:<file>:<symbol>`, `2_neg:<memory_id>`, `3_pos:<memory_id>`). Identical inputs yield identical context bundles and context fingerprints.
2. **Provenance Guarantee**: 100% of context items specify originating source identifier, module, digest, reason, and epistemic state.
3. **Negative Isolation**: Negative failure patterns and boundaries enter exclusively via `negative_boundaries` and never enter positive solution patterns.
4. **Conflict Quarantine**: Conflicting memories, task constraint collisions, or missing module references are quarantined under `conflicted_items` as `CONFLICTED` and never exposed as trusted positive guidance.
5. **Passive Neural Boundary**: `to_neural_context()` produces read-only dictionary data containing zero execution handles, callbacks, or mutation hooks.
