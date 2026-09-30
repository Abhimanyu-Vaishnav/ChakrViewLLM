# ChakrView Step 51: Implementation Plan — Project Arena Foundation & Closed-Loop Controller

- **Milestone**: Step 51
- **Focus**: Coding Language Acquisition & Project Arena Foundation
- **Baseline Invariant**: `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` ($\Delta W_{\text{baseline}} = 0$)
- **Model Invariants**: Parameters = 3,443,136 | Vocab = 4,096 | Max Context = 512 | CPU-First execution

---

## 1. Context & Architectural Principles

ChakrView is an indigenous, modular, low-resource neural intelligence architecture moving toward AGI. Coding is chosen as the initial capability domain because software engineering provides a deterministic, real-world feedback signal:
$$\text{Requirement} \to \text{Plan} \to \text{Generate} \to \text{Execute} \to \text{Observe} \to \text{Test} \to \text{Diagnose} \to \text{Repair} \to \text{Retest} \to \text{Evaluate} \to \text{Memory} \to \text{Learn}$$

### Strategic Invariant
The model **must not** modify ChakrView itself. It must learn first by building, testing, debugging, and repairing independent real-world external projects in a strictly isolated Project Arena.

---

## 2. Phase Breakdown & Status

### Phase 0: Reality-Check Repository Audit (COMPLETED)
- Inspect codebase, git log, tests, and documentation.
- Rigorously distinguish verified code from speculative/exaggerated claims.
- Artifact: `docs/STEP_51_IMPLEMENTATION_AUDIT.md`.

### Phase 1: Project Arena Foundation (COMPLETED)
- Implement directory layout:
  ```
  workspace/
  ├── specification/
  │   └── project_spec.json
  ├── source/
  │   └── [source files]
  ├── tests/
  │   └── [test files]
  ├── runtime/
  │   └── [configs/entrypoints]
  ├── logs/
  │   └── [execution logs]
  ├── artifacts/
  │   └── diffs/
  └── project_manifest.json
  ```
- Subsystem components:
  - `chakrview/arena/models.py`: Data models, exceptions, typed artifacts.
  - `chakrview/arena/workspace.py`: Path traversal protection (`_assert_within_workspace`), quota limits (max 50 files, 10MB), patch diff application, reset recovery.
  - `chakrview/arena/executor.py`: Sanitized subprocess runner with timeout enforcement and traceback diagnostics.
  - `chakrview/arena/evaluator.py`: AST syntax validator and full manifest evaluation.
  - `chakrview/arena/dataset.py`: Isolated project splitting and n-gram deduplication.

### Phase 2: Closed-Loop Feedback Controller (COMPLETED)
- Component: `chakrview/arena/loop.py` (`ArenaClosedLoopController`).
- Orchestrates multi-turn repair:
  - Formats initial spec prompt.
  - Runs tests via `SandboxedExecutor`.
  - Extracts failure diagnostics (exception type, error message, failing line).
  - Prompts model/generator with error context.
  - Applies patch via unified diff and re-executes tests up to `max_iterations`.
  - Emits `ExecutionHistory` with all iteration records and diffs.

### Phase 3: Curriculum Progression (DEFINED, AWAITING EMPIRICAL GATING)
- 11-level ladder from basic syntax to full autonomous multi-file development.
- Gating Rule: No curriculum level is marked complete without measured empirical pass rates from the model.

### Phase 4: Low-Resource Training Strategy (DEFINED)
- Model: 3.44M parameters, vocab 4096, context window 512.
- Curriculum learning on complete trajectories:
  $$\text{SPEC} \to \text{CODE} \to \text{RUN} \to \text{ERROR} \to \text{DIAGNOSIS} \to \text{PATCH} \to \text{RESULT}$$

### Phase 5: Memory / RIL Bridge (COMPLETED)
- Component: `chakrview/arena/memory.py` (`ArenaMemoryBridge`).
- Transforms `ExecutionHistory` into episodic experience records for future RIL consumption.

### Phase 6: Distributed Computing (SPECIFIED INTERFACES)
- Modular separation: MODEL $\neq$ TOOLS $\neq$ ARENA $\neq$ EVALUATOR $\neq$ MEMORY $\neq$ HOST SOURCE.
- Future network interface hooks preserved for off-node compilation/execution.

### Phase 7: Self-Improvement Roadmap (SPECIFIED)
- Strictly gated 10-stage progression. Current state: Stage 1 (External project learning).

---

## 3. Test Verification
- All 27 dedicated tests in `tests/test_step51_coding_arena.py` pass.
- Regression suite passing:
  - Step 50: 20/20 passed
  - Step 49: 20/20 passed
  - Step 48: 15/15 passed
- Baseline immutability verified ($\Delta W_{\text{baseline}} = 0$, hash intact).
