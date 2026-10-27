# ChakrView Experiment Archiving & Artifact Lifecycle Policy

**Document Status**: Active Repository Architecture Policy
**Target Audience**: ML Researchers, Validation Engineers, and System Architects
**Scope**: All research batches, overnight runs, benchmarks, and experimental checkpoints

---

## 1. Principles of Research Reproducibility

ChakrView is an indigenous neural language model and cognitive architecture research project. In order to keep Git repository size bounded while ensuring complete scientific reproducibility, artifacts are governed by strict tiered retention rules.

```
+-------------------------------------------------------------------------------+
|                      TIER 1: VERSIONED RESEARCH RECORD                        |
|   (Code, Config, Manifest, Summary Reports, Hypotheses, Metrics)              |
|   --> Committed to Git repository (`experiments/`, `docs/research/`)          |
+-------------------------------------------------------------------------------+
                                        |
+-------------------------------------------------------------------------------+
|                      TIER 2: RESEARCH ARTIFACT ARCHIVE                        |
|   (Intermediate Checkpoints, Large Prediction Traces, Replay Logs)            |
|   --> Preserved locally in `experiments/` / External Artifact Storage        |
+-------------------------------------------------------------------------------+
                                        |
+-------------------------------------------------------------------------------+
|                      TIER 3: REPRODUCIBLE LOCAL SCRATCH                       |
|   (Disposable Virtual Envs, AST Sandbox Outputs, Bytecode, OS Dumps)          |
|   --> Excluded via `.gitignore`; safely disposable without data loss          |
+-------------------------------------------------------------------------------+
```

---

## 2. Artifact Classification Taxonomy

### Tier 1: Versioned Research Record (`COMMIT_RESEARCH_EVIDENCE` / `COMMIT_SOURCE_OR_TESTS`)
What belongs in version control:
- **Experiment Source Code**: Python runner scripts, data generators, and diagnostic test harnesses.
- **Experiment Configurations**: Structured JSON/YAML files defining hyperparameters, learning rates, seeds, and token limits.
- **Run Manifests**: Cryptographic hashes of input datasets, prompt splits, and evaluation record counts.
- **Final Reports & Dossiers**: High-level markdown summaries detailing hypothesis, outcome, quantitative results, and root cause analysis.
- **Human Evaluation Records**: Reconciled scoring rubrics, rejection logs (e.g. Exp18 20/20 rejections), and reviewer notes.

### Tier 2: Research Artifact Archive (`ARCHIVE_RESEARCH_ARTIFACT`)
What belongs in local or external artifact storage (NOT in Git commit history):
- **Intermediate Model Weights (`.pt`, `.ckpt`, `.bin`)**: Checkpoint snapshots from intermediate training steps (e.g. Step 30, Step 60, Step 90) and pilot runs.
- **Large Prediction Traces (`> 1 MB .jsonl`)**: Raw generation token dumps containing tens of thousands of generation logs.
- **Storage Strategy**: Retained in dedicated local directories (`experiments/overnight_runs/`) or pushed to cloud object storage (S3/GCS/Git LFS) with SHA-256 references recorded in Tier 1 manifests.

### Tier 3: Reproducible Local Scratch (`KEEP_LOCAL_REPRODUCIBLE_SCRATCH`)
What must NEVER enter version control:
- **Virtual Environments**: `.venv/`, `venv/`, `packaging_validation/test_clean_venv/`.
- **Packaging Sandbox Workspaces**: `test_fresh_user_space/`, `dist/`, `build/`.
- **Python Bytecode & Caches**: `__pycache__/`, `*.pyc`, `.pytest_cache/`.
- **Operating System Artifacts**: `Thumbs.db`, `.DS_Store`.

---

## 3. Preservation Invariants

1. **No Silent Destruction**: Untracked experiment outputs are NEVER deleted via blanket commands like `git clean -fdx`.
2. **Negative Evidence Retention**: Negative results, catastrophic forgetting proofs, and human review rejections are permanent scientific assets; they must never be deleted or sanitized to make an experiment look successful.
3. **Weight Sovereignty**: The canonical Step 10,000 checkpoint (`training_checkpoint_step_10000.pt`) must never be modified, overwritten, or substituted with experimental weights.
