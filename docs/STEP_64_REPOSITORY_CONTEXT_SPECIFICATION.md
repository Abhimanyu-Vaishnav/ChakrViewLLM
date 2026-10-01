# Step 64 Specification: Persistent Repository Context & Incremental Invalidation

## 1. Architectural Architecture
Step 64 establishes an incremental, persistent repository intelligence/cache layer (`RepositoryContextStore`) that eliminates full-repository rescans on repeated tasks while strictly preserving deterministic AST semantics.

```
       ProjectManifest (Source Files)
                    |
                    v
    [RepositoryContextStore.synchronize()]
      /                                 \
(Content Hash Unchanged)          (Content Hash Changed)
      |                                   |
 [Reused from Cache]             [AST Inspection Recomputed]
      \                                   /
       +---------------------------------+
                        |
                        v
               RepositoryState
                        |
           [GroundedContextRetriever]
                        |
                        v
              GroundedContextBundle
           (Evidence-Backed Context)
```

---

## 2. Component Specifications

### 2.1 `RepositoryContextStore`
Maintains:
- `_file_cache`: Map of relative file paths to `CachedFileEntry` (SHA-256 digest, size, test classification, `ModuleInspection`).
- `_dependency_graph`: Bidirectional `RepositoryDependencyGraph`.
- `_current_state`: Live `RepositoryState` snapshot.
- `telemetry`: Tracking `indexing_time_ms`, `files_inspected`, `files_reused_from_cache`, and `cache_hit_ratio`.

### 2.2 Incremental Invalidation Algorithm
When synchronization detects file modifications:
1. Compares SHA-256 content hashes of each manifest file against cached entries.
2. If hash is identical, skips AST parsing and reuses cached `ModuleInspection`.
3. If hash differs, executes `RepositoryInspector.inspect_source(path, content)` only for modified files.
4. Removed files are purged from the cache.
5. Updates dependency graph edges and recomputes the canonical `state_fingerprint`.

---

## 3. Grounded Context Budgeting
To avoid token bloat and unfocused retrieval, `GroundedContextRetriever` applies a strict `GroundedContextBudget`:
- `max_files` (default: 5)
- `max_symbols` (default: 20)
- `max_dependency_depth` (default: 2)
- `max_memory_records` (default: 3)

Every retrieved file, function, class, and dependency is bundled with an `EvidenceRecord` guaranteeing verifiable provenance.
