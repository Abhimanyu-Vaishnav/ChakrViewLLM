# Step 64 Experiment Protocol: Grounded Neural Proposal & Persistent Context

## 1. Objective
Empirically demonstrate persistent repository context caching, incremental invalidation, budgeted context retrieval, hallucination containment, and fail-closed safety across 13 exhaustive conditions (A through M).

---

## 2. Experimental Conditions

| Condition | Focus Area | Pass Criteria |
|---|---|---|
| **A** | Persistent context caching | Cold synchronization inspects all files; warm synchronization inspects 0 files and achieves 100% cache hit ratio on unchanged manifest. |
| **B** | Incremental cache invalidation | Single file modification re-inspects only 1 file; remaining unchanged files are reused from cache. |
| **C** | Grounded context budgeting | Retrieved context bundle obeys budget limits (`max_files`, `max_symbols`) and attaches `EvidenceRecord` for all items. |
| **D** | Grounded proposal approval | Proposal referencing verified files and functions passes `HallucinationContainmentGate` with `KNOWN` epistemic state. |
| **E** | Hallucinated file rejection | Proposal targeting a nonexistent file is rejected with `CONTRADICTED` epistemic state. |
| **F** | Hallucinated symbol rejection | Proposal referencing an imaginary function is rejected with `CONTRADICTED` epistemic state. |
| **G** | Scope leak rejection | Proposal modifying files outside task allowed boundaries is rejected. |
| **H** | Stale memory rejection | Stale memory pattern is flagged by `RepositoryImpactAnalyzer`; proposal is blocked. |
| **I** | Negative transfer protection | Unrelated domain tags trigger deterministic `ABSTAIN`. |
| **J** | Malformed proposal rejection | Malformed step declaration (empty patch) rejected by `CandidateNormalizer`. |
| **K** | Execution failure rollback | Neural proposal that breaks test logic triggers atomic rollback with bit-exact fingerprint restoration. |
| **L** | Retrieval determinism | 3 repeated runs produce bit-exact identical candidate files, symbols, and evidence counts. |
| **M** | Baseline core immutability | Pre- and post-experiment weight hashes match `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` bit-exact. |

---

## 3. Protocol Execution
Implemented in `scripts/experiment_step64_grounded_proposal.py` and output to `artifacts/step64/step64_grounded_proposal_evidence.json`.
