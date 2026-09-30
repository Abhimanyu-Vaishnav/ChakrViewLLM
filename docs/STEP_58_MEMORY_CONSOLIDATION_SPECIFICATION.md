# ChakrView Step 58: Memory Consolidation Specification

- **Document Version**: 1.0.0
- **Status**: Ratified Specification (Step 58)
- **Scope**: Multi-Tier Deterministic Memory Architecture & Offline Consolidation Pipeline

---

## 1. Multi-Tier Memory Distinction

To prevent uncontrolled memory bloat and model hallucination contamination, ChakrView establishes a three-tier memory hierarchy:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. WORKING MEMORY (Turn & Task Scope)                       │
│ - Task state, current attempts, immediate tracebacks        │
│ - Ephemeral; bounded to the active CognitiveWorkspace      │
└──────────────────────────────┬──────────────────────────────┘
                               │ (Upon Verified Success)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. EPISODIC MEMORY (Instance Scope)                         │
│ - Raw verified ExperienceRecords from completed episodes    │
│ - Contains concrete failure observations & specific fixes   │
│ - Strictly admitted only after 100% test verification       │
└──────────────────────────────┬──────────────────────────────┘
                               │ (Offline Consolidation Pipeline)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. SEMANTIC MEMORY (Generalized Pattern Scope)              │
│ - Abstracted patterns synthesized from >= 2 verified exps   │
│ - Validated, deduplicated, and evidence-backed              │
│ - Injected into CognitiveContext for structural transfer    │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Memory Admission Rules

A memory entry is strictly prohibited from entering episodic or semantic stores unless all admission criteria are satisfied:

1. **Deterministic Verification**: The episode must have executed against ChakrKshetra test suites with `evaluator.test_result.passed > 0` and zero syntax or runtime errors.
2. **Recorded Provenance**: Every entry must reference its originating `task_id`, `task_family`, timestamp, and execution trace.
3. **Trace Completeness**: Must contain concrete initial action, failure observation, structured diagnosis, and verified correction.
4. **No Unresolved Failures**: Negative experiences without a verified resolution cannot be promoted into reusable memory.
5. **No Hallucinated Knowledge**: Raw model outputs that were not executed and passed in ChakrKshetra are discarded.

---

## 3. Consolidation Pipeline (`MemoryConsolidator`)

The consolidation pipeline operates offline, separating online task solving from background knowledge consolidation:

```
[VERIFIED EPISODES]
         │
         ▼
[EXPERIENCE EXTRACTION]
         │
         ▼
[ADMISSION VALIDATION]
         │
         ▼
[DEDUPLICATION & CLUSTERING] (Group by task_family & diagnostic root)
         │
         ▼
[PATTERN EXTRACTION] (Requires >= 2 independent verified episodes)
         │
         ▼
[SEMANTIC CANDIDATE GENERATION]
         │
         ▼
[PROMOTION TO SEMANTIC STORE]
```

### Semantic Memory Entry Schema (`SemanticMemoryEntry`)
- `pattern_id`: Unique deterministic hash/identifier.
- `pattern_name`: Descriptive name (e.g., `ArithmeticOperatorInversionRepair`).
- `task_family`: Associated family (e.g., `function_repair`).
- `problem_archetype`: Root cause pattern (e.g., `Operator inverted: subtraction used instead of addition`).
- `solution_strategy`: Abstracted resolution strategy (e.g., `Invert binary operator to match specification assertions`).
- `supporting_evidence`: List of `experience_id` references providing empirical backing.
- `evidence_count`: Number of independent verified episodes ($\ge 2$ required for promotion).
- `success_count`: Times this pattern successfully aided a solution.
- `failure_count`: Times this pattern was retrieved but did not lead to immediate solution.
- `boundary_conditions`: Explicit limits on where the pattern does NOT apply.
- `timestamp_utc`: Creation / update timestamp.

---

## 4. Deterministic Explainable Retrieval

Memory retrieval must be deterministic, CPU-first, and fully explainable without opaque vector embeddings.

### Scoring Formula:
$$\text{Score}(M, Q) = w_{\text{fam}} \cdot S_{\text{family}} + w_{\text{diag}} \cdot S_{\text{diag}} + w_{\text{ver}} \cdot S_{\text{evidence}} - w_{\text{pen}} \cdot S_{\text{mismatch}}$$

Where:
- $S_{\text{family}} \in \{0.0, 1.0\}$: Exact task family match.
- $S_{\text{diag}} \in [0.0, 1.0]$: Jaccard / token-overlap similarity between current diagnosis and pattern problem archetype.
- $S_{\text{evidence}} = \min(1.0, \text{evidence\_count} / 5.0)$: Confidence scaled by empirical evidence count.
- $S_{\text{mismatch}}$: Penalty for known boundary condition violations.
- Default weights: $w_{\text{fam}} = 0.4$, $w_{\text{diag}} = 0.4$, $w_{\text{ver}} = 0.2$, $w_{\text{pen}} = 0.5$.

Every retrieval result returns an audit record explaining the score breakdown.
