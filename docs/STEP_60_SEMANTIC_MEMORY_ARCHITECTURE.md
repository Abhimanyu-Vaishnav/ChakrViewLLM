# ChakrView Step 60: Semantic Repository Memory Architecture

- **Document Version**: 1.0.0
- **Status**: Ratified Specification (Step 60)
- **Scope**: Scalable Semantic Repository Memory, Retrieval Arbitration & Conflict Resolution

---

## 1. Architectural Position

Step 60 adds a **Semantic Repository Memory Arbitration Layer** above the existing Step 59
Repository Cognition Engine. The layer separation is strict:

```
CHAKRMICRO
    ≠
COGNITIVE CONTEXT
    ≠
COGNITIVE WORKSPACE
    ≠
REPOSITORY COGNITION (Step 59)
    ≠
EPISODIC MEMORY (Step 57/58)
    ≠
SEMANTIC REPOSITORY MEMORY  ← Step 60 new
    ≠
RETRIEVAL / ARBITRATION     ← Step 60 new
    ≠
CHAKRKHETRA
    ≠
EVALUATOR
    ≠
HOST APPLICATION
```

---

## 2. New Components

### 2.1 RepositorySemanticRecord (`semantic_record.py`)

Strongly typed dataclass specialised for repository-level patterns.

Fields: `memory_id`, `task_family`, `language`, `framework`, `repository_pattern`,
`symptom_signature`, `root_cause_signature`, `dependency_signature`, `affected_modules`,
`solution_pattern`, `verification_requirements`, `known_boundaries`, `confidence`,
`evidence_count`, `successful_episodes`, `failed_episodes`, `created_at`, `updated_at`,
`version`, `source_episode_ids`, `active_version`, `superseded_by`, `supersedes`.

### 2.2 RepositoryMemoryIndex (`memory_index.py`)

Deterministic, thread-safe, in-memory index for `RepositorySemanticRecord` entries.

Operations:
- `insert(record)` — with versioning if memory_id already exists
- `delete(memory_id)` — returns bool
- `lookup(memory_id)` — returns record or None
- `candidate_set(task_family, language, framework, ...)` — exact-filter retrieval
- `to_json()` / `from_json()` — deterministic serialization
- `count()` — total stored entries

No opaque vector database. CPU-first. Fully serializable.

### 2.3 Arbitration Layer (`arbitration.py`)

**RepositoryQuery** — typed query contract built from the active repository context.

**ArbitrationWeights** — explicit, documented scoring weights (Step 58 derived + Step 60
extensions):

| Signal           | Weight | Source   |
|:-----------------|:------:|:---------|
| task_family      | 0.30   | Step 58  |
| language         | 0.10   | Step 60  |
| framework        | 0.10   | Step 60  |
| symptom_diag     | 0.20   | Step 58  |
| dependency       | 0.10   | Step 60  |
| module_overlap   | 0.10   | Step 60  |
| evidence         | 0.10   | Step 58  |
| boundary_penalty | 0.50   | Step 58  |

Total positive budget = 1.00 (matches Step 58 invariant).

**arbitrate()** — full pipeline:

```
QUERY
  ↓ candidate_set (exact filters)
  ↓ score every candidate (multi-signal)
  ↓ rank descending (tie-break: ascending memory_id)
  ↓ conflict detection (|ΔScore| ≤ δ AND different solution)
  ↓ threshold check (score ≥ 0.75)
  ↓ SELECTED / AMBIGUOUS / CONFLICTING / REJECTED / NO_MATCH
```

**ArbitrationStatus** — five possible outcomes:
- `SELECTED` — single clear winner above threshold
- `AMBIGUOUS` — tied top candidates with identical solutions
- `CONFLICTING` — near-tied candidates with incompatible solutions → ABSTAIN
- `REJECTED` — top score below confidence threshold → ABSTAIN
- `NO_MATCH` — no candidate records found

---

## 3. Weight Derivation

The Step 58 formula allocates:
```
Score = 0.40*S_family + 0.40*S_diag + 0.20*S_evidence - 0.50*S_mismatch
```

Step 60 carved four new signals from within this budget:
- language (0.10) carved from w_family (0.40 → 0.30)
- framework (0.10) carved from w_evidence (0.20 → 0.10)
- dependency (0.10) carved from w_diag (0.40 → 0.20 symptom + 0.10 dep + 0.10 mod)
- module_overlap (0.10) carved from w_diag

The penalty weight (0.50) and confidence threshold (0.75) are unchanged from Step 58.

---

## 4. Memory Versioning

When a `memory_id` is re-inserted:
1. The old record has `active_version = False` and `superseded_by = new_id`
2. The new record has `supersedes = old_id` and an incremented `version`

Historical records remain auditable. No destructive overwrites.

---

## 5. Conflict Detection

Conflict is declared when:
1. |score(A) - score(B)| ≤ `conflict_delta` (default 0.05), AND
2. `solution_pattern(A) ≠ solution_pattern(B)`

This prevents silent selection of one incompatible repair when the evidence is insufficient
to distinguish.

---

## 6. Safe Abstention

The arbitration system abstains when:
- `REJECTED`: top score < 0.75 confidence threshold
- `CONFLICTING`: near-tied candidates prescribe incompatible solutions
- `NO_MATCH`: no candidates passed the exact filters

This is a core cognitive safety property.
