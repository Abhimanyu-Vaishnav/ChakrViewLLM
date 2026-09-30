# ChakrView Step 60: Experiment Protocol

- **Document Version**: 1.0.0
- **Status**: Active Protocol

---

## Benchmark Suite

### Benchmark A — Single Relevant Memory
- **Setup**: 1 memory record (5 verified episodes).
- **Query**: Exact match on all signals.
- **Expected**: `SELECTED`.
- **Measures**: Precision@1, latency.

### Benchmark B — 5 Superficially Similar
- **Setup**: 1 target + 4 records with same family but different dependency/symptom.
- **Query**: Same as A.
- **Expected**: `SELECTED` with target ranked #1.
- **Measures**: Precision@1.

### Benchmark C — 20 Mixed
- **Setup**: 1 target + 19 records across mixed families.
- **Query**: Same as A.
- **Expected**: `SELECTED` with target ranked #1.
- **Measures**: Precision@1, Recall.

### Benchmark D — 100 Mixed
- **Setup**: 1 target + 99 records across 4 families.
- **Query**: billing task.
- **Expected**: `SELECTED` with target ranked #1.
- **Measures**: Precision@1, latency (100-entry index).

### Benchmark E — 500 Competing
- **Setup**: 1 target + 499 records across 10 domains.
- **Query**: billing task.
- **Expected**: `SELECTED` with target ranked #1.
- **Measures**: Precision@1, latency (500-entry index).

### Benchmark F — Conflicting Memories
- **Setup**: 2 records with identical family/language/framework/symptom but incompatible solutions.
- **Query**: Exact match.
- **Expected**: `CONFLICTING` (abstain).
- **Measures**: Conflict Detection Rate.

### Benchmark G — Negative-Transfer Corpus
- **Setup**: 5 billing/auth/cache/django/fastapi records.
- **Query 1**: javascript async task → expected `NO_MATCH`.
- **Query 2**: billing task → expected `SELECTED` (correct family only).
- **Measures**: Negative Transfer Rate = cross-domain false positives.

### Benchmark H — Memory Ablation
- **Setup**: Empty index.
- **Query**: Billing task.
- **Expected**: `NO_MATCH`.
- **Measures**: Abstention Rate.

### Benchmark I — Memory Versioning
- **Setup**: Insert v1, then insert v2 with same memory_id.
- **Query**: Direct lookup.
- **Expected**: v2 is current record; v1 is superseded.
- **Measures**: Version correctness.

### Benchmark J — Consolidation
- **Setup**: Single consolidated record with 4 source_episode_ids.
- **Query**: Billing task.
- **Expected**: `SELECTED`.
- **Measures**: Evidence count preserved.

---

## Metrics Collected

| Metric                  | Definition |
|:------------------------|:-----------|
| Precision@1             | Top-ranked candidate is the correct target |
| Recall                  | Target is in the candidate set |
| False Positive Rate     | Irrelevant records above threshold |
| Negative Transfer Rate  | Cross-domain records returned for out-of-domain query |
| Abstention Rate         | Fraction of queries that result in ABSTAIN |
| Conflict Detection Rate | Fraction of conflicting pairs correctly detected |
| Retrieval Latency       | Wall-clock ms for `arbitrate()` call |
| Index Build Time        | ms to build from 500 records |
| Serialization Size      | JSON bytes for 500-entry index |
