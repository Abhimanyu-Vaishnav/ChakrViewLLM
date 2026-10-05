# Steps 82–85: Persistent Task Decomposition, Scheduling, Work Loop & Knowledge Evolution

## 1. Subsystem Architecture Overview

Steps 82 through 85 introduce a persistent, resource-aware task orchestration layer above the Persistent Project Brain (PPB) and connect it to the cognitive reasoning, patch planning, and execution cycle:

```
User Task ("Refactor checkout flow...")
       │
       ▼
DeterministicTaskDecomposer (Step 82)
       │ (Grounded in PPB retrieved context)
       ▼
PersistentTaskGraph (DAG in SQLite WAL)
 ├── Step 1: Inspection & Architecture (INSPECTION)
 ├── Step 2: Dependency & Grounding (DEPENDENCY_ANALYSIS)
 ├── Step 3: Proposal Design & Self-Evaluation (REASONING)
 ├── Step 4: Safe Patch Execution (PATCH_EXECUTION)
 └── Step 5: Verification & Brain Evolution (VERIFICATION)
       │
       ▼
ResourceAwareTaskScheduler (Step 83)
       │ (Maps to HardwareCapability: LOW_RESOURCE / STANDARD / ACCELERATED)
       ▼
TaskExecutionLease
       │ (Allocates memory bounds, concurrency, exclusive write lock, accelerator flag)
       ▼
IncrementalCognitiveWorkLoop (Step 84)
       │ (Bounded iterations, checkpointed per-node progress, process restart resilience)
       ▼
Structured Knowledge Evolution (Step 85)
       │ (Evolves PPB with TASK_HISTORY, OBSERVATION, and verified facts)
       ▼
PersistentProjectBrain Updated
```

---

## 2. Core Architectural Principles Enforced

1. **Persistent Task Graph (Step 82)**:
   - High-level requests are broken down into a typed DAG of [`PersistentTaskNode`](file:///d:/Project/ChakrView/chakrview/cognition/ppb/task_models.py#L38) records stored in SQLite via [`PersistentTaskStorage`](file:///d:/Project/ChakrView/chakrview/cognition/ppb/task_storage.py#L32).
   - Complete process restarts do not lose progress; completed tasks are never rediscovered or re-run.
   - Cycle detection via Kahn's algorithm validates the graph on node addition.

2. **Resource-Aware Scheduling (Step 83)**:
   - Evaluates ready nodes against [`HardwareCapability`](file:///d:/Project/ChakrView/chakrview/runtime/resource.py#L13) and [`ResourcePolicy`](file:///d:/Project/ChakrView/chakrview/runtime/resource.py#L24).
   - `LOW_RESOURCE`: Strictly sequential execution (1 node at a time), bounded memory (256 MB), CPU-only.
   - `STANDARD`: Up to 2 concurrent read-only inspection tasks; exclusive lock for `PATCH_EXECUTION`.
   - `ACCELERATED`: Enables GPU acceleration for `REASONING` where beneficial, up to 4 concurrent leases.

3. **Incremental Cognitive Work Loop (Step 84)**:
   - Bounded execution iterations.
   - Interrupted execution preserves state: completed nodes stay `COMPLETED`, remaining nodes stay `PENDING`.
   - ChakrView restart reloads graph directly from SQLite and resumes from the exact boundary without redundant full-project rescans.

4. **Project Knowledge Evolution (Step 85)**:
   - Completed work produces structured, epistemically classified project knowledge (`TASK_HISTORY`, `OBSERVATION`, `MODULE`, `SYMBOL`).
   - Inferences never become `FACT` without passing verification.
   - Code changes trigger selective invalidation in PPB (`STALE`), keeping unaffected modules `VALID`.

5. **Neural Core Invariant**:
   - Zero execution authority in the neural core ($\Delta W = 0$).
   - Parameter count: `3,443,136`, SHA-256 weight hash: `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`.

---

## 3. Empirical Verification & Test Results

* **Dedicated Unit & Invariant Tests** ([test_step82_85_task_orchestration.py](file:///d:/Project/ChakrView/tests/test_step82_85_task_orchestration.py)): 7/7 tests passed.
* **Realistic Multi-File Project Experiment** ([test_step82_85_experiment.py](file:///d:/Project/ChakrView/tests/test_step82_85_experiment.py)): 1/1 passed.
  * Verified initial bounded scan (2 files), process kill, restart, resumed scan (5 remaining files), task decomposition (5 nodes), execution under `LOW_RESOURCE` constraints, durable knowledge evolution, and zero rescanning on subsequent tasks.
* **Steps 59–85 Regression Suite**: 202/202 tests passed in `21.17s`.
