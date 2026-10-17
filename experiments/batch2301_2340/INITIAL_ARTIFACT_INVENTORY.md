# INITIAL ARTIFACT INVENTORY — BATCH 2301–2340

**Mission:** Forensic Evidence Reconciliation, Checkpoint Identity & Evaluation Integrity  
**Batch Scope:** 2301–2340  
**Repository Branch:** `dev/v0.6-v0.7-validation`  
**Starting HEAD Commit:** `32cce84` (`feat(batch2261-2300): evidence audit, runtime execution truthfulness, candidate integrity, and source-wise evaluation`)  
**Timestamp:** 2026-10-17T13:58:30+05:30  

---

## 1. Initial Repository State

- **Branch:** `dev/v0.6-v0.7-validation`
- **HEAD Commit:** `32cce84f91ae1bf91a22afded7197c87ea7a8f83`
- **Working Tree Status:** Clean with respect to preceding batch deliverables; uncommitted user work in `chakrview/runtime/session.py` and untracked documentation files in `docs/sprints/` and older experiment dirs are strictly preserved.
- **Protected Files:** `chakrview/runtime/session.py` remains untouched.

---

## 2. Inventory of Prior Batch Deliverables

### Batch 2221–2260 Deliverables (`experiments/batch2221_2260/`)
1. `integrated_assistant_adapter.py`: Adapter connecting CLI to adaptive runtime and verified tool router.
2. `adaptive_runtime_engine.py`: Adaptive hardware execution and benchmarking engine.
3. `verified_tool_router.py`: AST-based deterministic arithmetic calculator.
4. `run_independent_evaluation.py`: Standalone evaluator for the 290 test documents.
5. `run_inference_benchmarks.py`: Benchmark suite for CPU and GPU inference.
6. `REPRODUCED_EVALUATION.json`: Evaluation results and per-document loss records.
7. `HARDWARE_BENCHMARKS.json`: Latency, throughput, and memory measurements on NVIDIA RTX A1000 6GB Laptop GPU and Intel 20-core CPU.
8. `FINAL_EVIDENCE.json`: Synthesis of results from Batch 2221–2260.
9. `V0_6_CANDIDATE_READINESS.md`: Evaluation and packaging report for candidate v0.6.0-rc1.
10. `FEEDBACK_INTEGRATION_REPORT.md`: Consent and feedback integration audit.

### Batch 2261–2300 Deliverables (`experiments/batch2261_2300/`)
1. `EVALUATION_PROVENANCE_AUDIT.md`: Checkpoint provenance audit across 6 evaluation models.
2. `EVALUATION_AUDIT.json`: Recomputed evaluation metrics and per-document losses.
3. `RUNTIME_DEVICE_TRUTH_REPORT.md`: Granular breakdown of execution device attribution.
4. `HARDWARE_REPRODUCIBILITY_AUDIT.md`: Re-audit of hardware metadata and RAM discrepancy.
5. `CANDIDATE_INTEGRITY_AUDIT.md`: Detailed inspection of candidate packaging and invariants.
6. `candidate_manifest.json`: Verified candidate manifest with SHA-256 hashes.
7. `FEEDBACK_CONSENT_AUDIT.md`: Forensic audit of opt-in consent and feedback persistence.
8. `SOURCE_WISE_EVALUATION.md`: Mathematical analysis of source-wise performance and Sanskrit regression.
9. `MISSION_SUMMARY.md`: High-level summary of Batch 2261–2300 achievements.
10. `FINAL_EVIDENCE.json`: Gate decisions and observed evidence.

---

## 3. Discrepancies Flagged for Batch 2301–2340 Investigation

1. **Seed 42 Candidate Path and Hash Mismatch:**
   - Mention of path `experiments/batch2141_2180/checkpoints/cpt_seed_42/best_model.pt` with SHA-256 `d3268fc43f114c07d3fa8f5c9a75d9e5d4cb05f1542f5674cfffcfe2db894101` vs `experiments/batch2141_2180/cpt/seed_42/cpt_best_checkpoint.pt` with SHA-256 `d3268fc49378f8c5b9a30e71d4bab58ce7c4db6bcaaa1f32932ccfd8694b07c3`.
2. **Corpus Manifest and Invariant Hashes:**
   - Multiple historical hashes cited for cleaned corpus manifest and initialization digests.
3. **Source-Wise and Overall Cross-Entropy Aggregation:**
   - Verification of mathematical aggregation from source-level token losses to total benchmark loss.
4. **Hardware Identification:**
   - Reconcile RTX A1000 6GB vs RTX 3050 and 31.64 GB vs 15.65 GB RAM.
