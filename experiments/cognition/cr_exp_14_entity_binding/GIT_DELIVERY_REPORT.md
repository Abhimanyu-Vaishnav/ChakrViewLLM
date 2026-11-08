# CR-EXP-14 Git Delivery Report

## 1. Lineage & Safety Verification
- **Branch:** `exp/cbr-01-conversational-breakthrough`
- **Parent Commit:** `267a6d4`
- **CR-EXP-14 Commit:** `15275853ad00e3c75477a0ff134b7a13c37ff98b`
- **Remote:** `origin/exp/cbr-01-conversational-breakthrough`
- **Push Status:** Successfully synchronized to `https://github.com/Abhimanyu-Vaishnav/ChakrViewLLM.git`
- **Runtime Integrity:** `chakrview/runtime/session.py` remained byte-for-byte untouched (SHA-256: `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67`).
- **Zero Weight Commit Policy:** Strict compliance verified. No `.pt` checkpoint weights or temporary binary caches staged or committed.

---

## 2. Key Deliverables Delivered & Tracked in Git
- `experiments/cognition/cr_exp_14_entity_binding/`
  - `CR14_PARENT_AUDIT.md`
  - `CR14_BENCHMARK_SPEC.md`
  - `CR14_BENCHMARK_MANIFEST.json`
  - `CR14_LEAKAGE_AUDIT.md`
  - `CR14_FROZEN_TEST_HASH.txt`
  - `BASELINE_EVALUATION.md`
  - `TRAINING_INTERVENTIONS.md`
  - `TRAINING_RESULTS.md`
  - `STRICT_ENTITY_BINDING_RESULTS.md`
  - `FAILURE_ANALYSIS.md`
  - `STATISTICAL_COMPARISON.md`
  - `ARCHITECTURE_DIAGNOSTIC.md`
  - `UPDATED_CAPABILITY_REGISTRY.md`
  - `V0_9_1_READINESS.md`
  - `FINAL_CR_EXP_14_REPORT.md`
  - `ARTIFACT_HASHES.txt`
  - `benchmark_data/` (Contains sealed 1,200 benchmark json & manifest)
  - `data/` (Disjoint training curricula and validation sets)
  - `scripts/` (Sealed benchmark generator, training engine, evaluators)
  - `eval_outputs/` (Detailed evaluation JSONs for all 6 models)
  - `checkpoints/` (Training loss histories for Candidates A, B, and C)
