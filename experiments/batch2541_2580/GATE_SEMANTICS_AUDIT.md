# GATE SEMANTICS AUDIT: Phase 2 Harden Gate Semantics
**Batch:** 2541–2580  
**Repository:** ChakrViewLLM (`dev/v0.6-v0.7-validation` @ `3b30f86`)  
**Target:** Evidence Validator Gate Disambiguation & Hardening  

---

### 1. Conceptual Disambiguation and Gate Decoupling

In prior batches (e.g. Batch 2501–2540), gate definitions occasionally conflated structural conformance with substantive scientific rigor, or working-tree file modifications with historical asset provenance. In Batch 2541–2580, we formally separate and codify these four distinct concepts:

| Semantic Dimension | Structural / Preservation Dimension | Scientific / Provenance Dimension |
| :--- | :--- | :--- |
| **Preregistration** | **`preregistration_structure`**: Verifies that the JSON/MD schema contains required top-level keys (`experimental_arms`, `controls_and_invariants`, `evaluation_metrics`, `decision_rules`), that 4 arms are declared, and formatting is valid. | **`preregistration_scientific_readiness`**: Verifies that experimental design is unconfounded (e.g., Arms 1–3 anchored to canonical Step 10,000 checkpoint; leakage screening protocol is multi-tiered; tokenizer preflight has empirical char/token thresholds; statistical decision rules handle variance). |
| **Protected File (`session.py`)** | **`protected_file_preservation`**: Verifies that the protected file `chakrview/runtime/session.py` was **NOT** modified, staged, overwritten, or tampered with by the current batch. Verified by working tree diff isolation. | **`protected_file_provenance`**: Analyzes the historical origin and timestamp of the existing uncommitted diff (ChatML session runner modification with mtime `2026-10-11 16:04:15` predating Batch 2381). |

---

### 2. Failure Modes Prevented by Decoupled Gates

1. **False Readiness Passing via Superficial JSON Completeness:**
   - *Previous Defect:* `validate_preregistration_completeness` only checked for the existence of `["experimental_arms", "controls_and_invariants", "evaluation_metrics", "decision_rules"]`. A JSON file specifying confounded arms (e.g. comparing step 10k baseline to step 10.5k candidate without matching controls) would pass as `VERIFIED`.
   - *Hardened Semantic:* `preregistration_scientific_readiness` explicitly inspects checkpoint lineage, controls, tokenizer preflight criteria, and statistical decision rules. If any required scientific amendment is unresolved (or marked `REQUIRES_REVISION`), this gate evaluates strictly to `FAILED` or `REQUIRES_REVISION`.

2. **Conflation of Protected File Tampering with Pre-existing User Diff:**
   - *Previous Defect:* A single gate `protected_file_provenance` returned `PARTIALLY_VERIFIED` because of the uncommitted diff, obscuring whether the current batch had touched the file.
   - *Hardened Semantic:* `protected_file_preservation` evaluates strictly to `VERIFIED` if the file hash and diff remain identical to the starting state of the batch. `protected_file_provenance` remains `PARTIALLY_VERIFIED` to truthfully document that an uncommitted working-copy change exists in the user workspace.

3. **Absence of Execution Treated as Success:**
   - Any missing execution log or test result defaults to `UNVERIFIED` or `FAILED`. Gate aggregations require positive execution proofs (exit code 0, matching pass counts).

---

### 3. Implementation of Decoupled Gates in Evidence Validator

The evidence validator architecture in Batch 2541–2580 enforces these rules across 9 distinct gates:
1. `artifact_integrity` (SHA-256 canonical hash verification)
2. `training_coverage_derivation` (per-document exact tokenization, zero Sanskrit training coverage verified)
3. `held_out_evidence_validity` (bidirectional set equality, non-finite loss rejection, exact NLL summation)
4. `validator_adversarial_robustness` (rejection of corrupted evidence files, malformed manifests)
5. `regression_test_execution` (subprocess execution, 25/25 passed, exit code 0)
6. `preregistration_structure` (presence of required sections and schema keys)
7. `preregistration_scientific_readiness` (lineage unconfounded, leakage screening defined, empirical tokenizer criteria, variance decision rules)
8. `protected_file_preservation` (zero modifications introduced to `chakrview/runtime/session.py`)
9. `protected_file_provenance` (historical mtime analysis and commit tracking)
