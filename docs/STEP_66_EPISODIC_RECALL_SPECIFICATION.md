# Step 66 Episodic Recall Specification: Episodic Memory Recall Loop, Structural Relevance & Deterministic Validity Verification

## 1. Overview & Architectural Role

Step 66 implements an active **Episodic Memory Recall Loop** connecting historical verified repository memories with new tasks and the neural proposal boundary without modifying neural weights ($\Delta W = 0$).

```
Past Verified Episode -> Learned Semantic Memory -> Memory Recall Request
    -> Deterministic Candidate Retrieval -> Structural Relevance Scoring
    -> Validity / Drift Verification -> Conflict & Negative Boundary Handling
    -> Bounded Recalled Context Bundle -> Passive Neural Proposal Input
```

---

## 2. Core Schemas & Contracts

### 2.1 Memory Recall Request (`MemoryRecallRequest`)
Explicit, typed query contract:
- `task_description`: Natural/task specification string.
- `task_family`: Scoped task family identifier.
- `target_files`: Target files/modules identified for the task.
- `target_symbols`: Functions or classes relevant to the task.
- `repository_pattern`: High-level architectural pattern signature.
- `symptom_signature`: Symptom signature to match.
- `root_cause_signature`: Root cause classification.
- `dependency_signature`: Upstream/downstream dependency chain signature.
- `repository_fingerprint`: Cryptographic repository state fingerprint.
- `domain_tags`: Permitted domains for cross-domain negative transfer protection.
- `language`, `framework`: Environment constraints.

### 2.2 Recall Lifecycle States (`MemoryRecallStatus`)
Explicit classification for evaluated memory candidates:
- `RECALLABLE`: Valid, active, relevant positive solution pattern.
- `NEGATIVE_BOUNDARY`: Confirmed failure pattern or negative boundary constraint (`DO_NOT_APPLY`).
- `REJECTED_STALE`: Stale due to upstream or direct structural drift.
- `REJECTED_SUPERSEDED`: Inactive version superseded by a newer verified record.
- `CONFLICTED`: Collides with a negative boundary or an equally scored conflicting positive solution.
- `UNAVAILABLE`: Required target modules or files missing from the active repository state.
- `ABSTAIN`: Blocked due to cross-domain negative transfer or ungrounded ambiguity.

### 2.3 Bounded Recalled Context Bundle (`RecalledContextBundle`)
Structured container passed to the neural proposal boundary:
- `task_family`: Task family identifier.
- `positive_memories`: List of `RecalledMemoryItem` approved as positive guidance.
- `negative_boundaries`: List of `RecalledMemoryItem` exposing constraints not to repeat.
- `rejected_items`: List of rejected/conflicted items maintained for auditability.
- `abstained`: Boolean indicating fail-closed abstention.
- `abstain_reason`: Diagnostic rationale if abstained.
- `total_candidates_evaluated`: Count of evaluated index entries.
- `duration_ms`: Telemetry timing.

---

## 3. Relevance Scoring & Arbitration Policies

### 3.1 Deterministic Relevance Formula
Relevance scores are bounded in $[0.0, 1.0]$ based on explainable signals:
- **Task Family Match**: 0.35
- **Module Overlap**: 0.25 (Jaccard overlap between requested files and affected modules)
- **Pattern / Symptom Similarity**: 0.20 (Token Jaccard between query and stored signatures)
- **Dependency Match**: 0.10
- **Evidence / Success History**: 0.10 (Scaled by verified success count)

### 3.2 Conflict & Negative Boundary Policy
1. **Positive vs Negative Collision**: If an admitted negative boundary constraint touches any module targeted by a candidate positive solution, the positive candidate is classified as `CONFLICTED` and excluded from positive guidance.
2. **Equi-Scored Ambiguity**: If two top positive candidates differ in solution pattern on identical modules with a score delta $< 0.05$, both are classified as `CONFLICTED` rather than making a subjective choice.

### 3.3 Budgeting & Order Enforcement
Limits enforced by `RecallBudget`:
- Maximum candidate memories evaluated: 10
- Maximum positive memories exposed: 3
- Maximum negative boundaries exposed: 3
- Deterministic ordering: descending by relevance score, then alphabetically by `memory_id`.
