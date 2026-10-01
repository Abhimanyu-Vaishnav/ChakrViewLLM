# Step 64 Cognitive Evidence: Measured Results for Grounded Proposal & Persistent Context

## 1. Experiment Summary
* **Benchmark Script**: `scripts/experiment_step64_grounded_proposal.py`
* **Evidence Artifact**: `artifacts/step64/step64_grounded_proposal_evidence.json`
* **Status**: **13 / 13 Conditions PASSED (100% Precision)**
* **Neural Immutability**: $\Delta W = 0$, baseline hash invariant:
  `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`

---

## 2. Experimental Measurements

### Condition A: Persistent Context Caching
- Cold Scan Files Inspected: 7 (100% of files)
- Cold Files Reused: 0
- Warm Sync Files Inspected: 0 (0% re-scanned)
- Warm Files Reused: 7 (100% cache hit ratio)
- Fingerprint Invariance: Cold state fingerprint matches warm state fingerprint bit-exact.

### Condition B: Incremental Cache Invalidation
- Repository Action: Mutated `tax_service.py`
- Files Re-Inspected: 1
- Files Reused from Cache: 6
- Result: AST parsing eliminated for 85.7% of repository files.

### Condition C: Grounded Context Budgeting
- Budget Config: `max_files=3`, `max_symbols=5`
- Candidate Files Selected: 2 (<= 3)
- Relevant Symbols Selected: 2 (<= 5)
- Evidence Records Emitted: 4 (100% with traceable provenance)

### Condition D: Grounded Proposal Approval
- Proposal Target: `tax_service.py` -> `compute_tax`
- Grounding Gate Result: `is_grounded=True`, `epistemic_state=KNOWN`

### Condition E: Hallucinated File Rejection
- Proposed File: `ghost_payment_service.py`
- Grounding Gate Result: `is_grounded=False`, `epistemic_state=CONTRADICTED`
- Unfounded Files Captured: `["ghost_payment_service.py"]`

### Condition F: Hallucinated Symbol Rejection
- Proposed Symbol: `nonexistent_crypto_tax_algorithm`
- Grounding Gate Result: `is_grounded=False`, `epistemic_state=CONTRADICTED`
- Unfounded Symbols Captured: `["nonexistent_crypto_tax_algorithm"]`

### Condition G: Scope Leak Rejection
- Proposed Target: `models.py` (outside allowed task scope)
- Grounding Gate Result: `is_grounded=False`
- Reason: `"File 'models.py' is outside allowed task scope"`

### Condition H: Stale Memory Rejection
- Injected Memory: `rec_stale_step64`
- Perturbation: Structural import mutation in reference module
- SafetyGate Decision: `ABSTAIN` (blocked from execution)

### Condition I: Negative Transfer Defense
- Task Domain: `billing_refactor`
- Candidate Domain: `oauth2_authentication`
- SafetyGate Decision: `ABSTAIN` (Domain mismatch)

### Condition J: Malformed Neural Proposal Rejection
- Candidate Anomaly: Empty patch dictionary
- Normalizer Decision: `is_valid=False`, rejected before safety gate or coordinator.

### Condition K: Execution Failure Rollback
- Defect Injected: Candidate returned `-999.0` breaking tests
- Result: Atomic rollback executed; repository state fingerprint restored bit-exact.

### Condition L: Retrieval Determinism
- Repeated Trials: 3
- Variance Across Retreived Files, Symbols, and Evidence: 0.000%

### Condition M: Neural Baseline Immutability
- Pre-Hash: `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`
- Post-Hash: `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`
- Parameter Count: 3,443,136
- Status: **PASSED (Bit-Exact)**

---

## 3. Test Suites
- Step 64 Focused Tests (`tests/test_step64_grounded_proposal.py`): **10 / 10 passed** in 1.76s.
- Steps 59–64 Subsystem Suite: **90 / 90 passed** in 16.31s.
- Full Repository Regression Suite: **1,573 / 1,573 passed** in 332.89s.
