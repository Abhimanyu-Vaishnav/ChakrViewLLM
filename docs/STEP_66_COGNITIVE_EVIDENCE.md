# Step 66 Cognitive Evidence: Episodic Memory Recall Benchmark Results

## 1. Benchmark Execution Summary

- **Script**: `scripts/experiment_step66_episodic_recall.py`
- **Evidence File**: `artifacts/step66/step66_episodic_recall_evidence.json`
- **Status**: **15 / 15 Conditions PASSED (100% Precision)**
- **Baseline Weight Hash**: `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`
- **Weight Delta**: $\Delta W = 0$ (Bit-Exact Invariant Preserved)
- **Model Parameter Count**: 3,443,136

---

## 2. Experimental Measurements

### Condition A: Exact Task-Family Recall
- Query Family: `billing_refactor`
- Recalled Positives: 2 (`mem_tax_service_v1`, `mem_discount_engine_v1`)
- Status: PASSED

### Condition B: Same-Module Recall
- Query Family: `pricing_logic`
- Target Module: `discount_engine.py`
- Recalled Item: `mem_discount_engine_v1`
- Status: PASSED

### Condition C: Structural-Pattern Recall
- Query Pattern: `"Corrected percentage calculation for rate multiplier"`
- Pattern Similarity Score: 0.20 (Token Jaccard match)
- Status: PASSED

### Condition D: Irrelevant Memory Rejection
- Query Family: `database_migration`
- Recalled Positives: 0
- Status: PASSED

### Condition E: Stale-Memory Rejection on Structural Drift
- Repository Action: Mutated imports in `tax_service.py`
- Rejection Status: `REJECTED_STALE`
- Stale Candidate: `mem_tax_service_v1`
- Status: PASSED

### Condition F: Superseded-Memory Rejection
- Inactive Candidate: `mem_tax_service_v0`
- Rejection Status: `REJECTED_SUPERSEDED`
- Superseded By: `mem_tax_service_v1`
- Status: PASSED

### Condition G: Negative-Boundary Recall
- Failure Memory: `sem_neg_concurrency_deadlock`
- Classification: `NEGATIVE_BOUNDARY`
- Solution Pattern: `"DO_NOT_APPLY"`
- Status: PASSED

### Condition H: Positive-vs-Negative Conflict Arbitration
- Collision Target: `tax_service.py`
- Conflicted Candidate: `mem_tax_service_v1`
- Status: `CONFLICTED` (excluded from positive guidance)
- Status: PASSED

### Condition I: Multiple-Memory Deterministic Ordering
- Ordered Candidates: `['mem_tax_service_v1', 'mem_discount_engine_v1']`
- Score Ordering: `0.7800 >= 0.5250`
- Status: PASSED

### Condition J: Recall Budget Enforcement
- Budget Limit: `max_recalled_memories = 1`
- Recalled Count: 1
- Rejected Count: 1
- Status: PASSED

### Condition K: Repository Fingerprint Mismatch
- Query Fingerprint: `nonexistent_old_fingerprint`
- Provenance Preserved: `EvidenceRecord` attached with source `SEMANTIC_MEMORY`
- Status: PASSED

### Condition L: Repeated-Run Determinism
- Serialized Bundle 1 == Serialized Bundle 2
- Equivalence: Bit-exact deterministic JSON equivalence
- Status: PASSED

### Condition M: Neural-Boundary Immutability
- Pre-Check Hash: `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`
- Post-Check Hash: `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`
- Parameters: 3,443,136
- Delta: $\Delta W = 0$
- Status: PASSED

### Condition N: Empty-Memory Abstention
- Index Size: 0
- Total Evaluated: 0
- Positive Guidance: Empty
- Status: PASSED

### Condition O: Cross-Domain Negative-Transfer Defense
- Query Domain: `auth_tokens`
- Permitted Domains: `{'billing_refactor', 'pricing_logic'}`
- Outcome: `abstained = True`
- Status: PASSED
