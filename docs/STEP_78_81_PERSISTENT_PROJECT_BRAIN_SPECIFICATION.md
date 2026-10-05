# Steps 78–81: Persistent Project Brain (PPB) Architecture Specification

## 1. Architectural Motivation & Boundary Principles

The Persistent Project Brain (PPB) provides ChakrView with a persistent, project-aware cognitive memory that enables progressive understanding of large codebases without requiring the entire project to fit into one bounded prompt or context window.

### Non-Negotiable Core Principles
1. **Resource-Adaptive Runtime**: Respects hardware capacity profiles (`LOW_RESOURCE`, `STANDARD`, `ACCELERATED`, `DISTRIBUTED_READY`). Low-resource profiles operate sequentially with small bounded batches (e.g. 2 files/chunk); higher tiers utilize wider batches.
2. **Real Persistence**: Knowledge records are stored durably using SQLite with WAL mode, foreign keys, and atomic commits. Understanding persists across process termination, Python restarts, and new sessions.
3. **Strict Epistemic Provenance**: Every record maintains explicit provenance (`file_path`, `symbol_name`, `evidence_ids`, `source_chunk`, `repo_fingerprint`). Inferences never silently become `FACT`. Unseen knowledge is `UNKNOWN` or empty; insufficient evidence is `INSUFFICIENT`; conflicting changes are `CONTESTED`; modified knowledge is `STALE`; verified re-analysis yields `REVERIFIED`.
4. **Incremental Resumability**: Codebases are scanned in bounded chunks. Scan progress per file and hash is persisted; interrupted or resumed scans skip previously scanned unchanged units without redundant work.
5. **Change-Aware Selective Invalidation**: Modifications to a file (e.g. `billing/discount.py`) invalidate only directly and transitively affected symbols/modules (`STALE`), while leaving all unrelated module knowledge intact and `VALID`.
6. **Historical Auditability**: Knowledge records are versioned rather than deleted. Updates point to superseded historical records (`supersedes`, `active_version=False`).
7. **Neural Invariant**: Neural core remains frozen and bit-exact ($\Delta W = 0$, $3,443,136$ parameters, SHA-256: `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`).

---

## 2. Component Taxonomy

```
chakrview/cognition/ppb/
├── __init__.py           # Package exports
├── models.py             # EpistemicStatus, KnowledgeRecordType, KnowledgeRecord, ProjectIdentity, ProjectBrainState
├── storage.py            # PersistentBrainStorage (SQLite ACID WAL-mode database, versioning, migrations)
├── brain.py              # PersistentProjectBrain (Core adapter uniting storage, identity, and hardware policy)
├── scanner.py            # IncrementalProjectScanner (Bounded, resumable, change-skipping file batch scanner)
├── retrieval.py          # ProjectKnowledgeRetriever (Task-targeted, budget-enforced knowledge bundle extractor)
└── maintainer.py         # ChangeAwareBrainMaintainer (Selective invalidation, re-verification, task learning recorder)
```

---

## 3. Cognitive Pipeline Integration

The Persistent Project Brain interfaces directly into the repository cognition pipeline:
```
Repository / Codebase
       │ (Bounded AST batches)
       ▼
IncrementalProjectScanner
       │ (Structured records)
       ▼
PersistentProjectBrain (SQLite ACID WAL)
       │ (Task query & budget)
       ▼
ProjectKnowledgeRetriever
       │ (RetrievedKnowledgeBundle & EvidenceRecords)
       ▼
UnifiedCognitiveContextComposer (CognitiveContextSource.PROJECT_BRAIN)
       │ (Bounded CognitiveContextBundle)
       ▼
StructuredReasoningEngine -> CriticalThinkingEngine -> SelfEvaluator
       │ (Validated ProposalContract)
       ▼
PatchPlanner -> SafePatchExecutor
       │ (RepositoryDiff / ImpactReport)
       ▼
ChangeAwareBrainMaintainer -> PersistentProjectBrain (Selective Invalidation & REVERIFIED update)
```

---

## 4. Verification & Experiment Summary

The implementation was validated against a 20-point requirements matrix and an 11-phase demonstration experiment (`tests/test_step78_81_experiment.py`):
- **Phases 1–2**: Bounded partial project scan and persistent commit.
- **Phase 3**: Complete process termination / object deletion.
- **Phase 4**: Reload from SQLite and targeted retrieval without rescan.
- **Phase 5**: Resumed scan processing only remaining unscanned units.
- **Phase 6**: Targeted task reasoning over retrieved records.
- **Phases 7–8**: Safe patch application and durable task learning persistence.
- **Phases 9–10**: File modification, selective invalidation of dependent records to `STALE`, and targeted re-analysis to `REVERIFIED`.
- **Phase 11**: Query on unrelated module proving untouched facts remain `VALID` without full-project rescanning.
