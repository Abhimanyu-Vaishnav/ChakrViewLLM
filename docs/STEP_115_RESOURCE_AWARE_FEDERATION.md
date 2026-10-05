# STEP 115: Resource-Aware Worker Federation

## 1. Overview
Step 115 extends ChakrView's worker federation layer to incorporate hardware-bounded resource profiling and deterministic placement. Instead of arbitrary scheduling, worker capacity is mapped against task requirements without requiring GPUs.

## 2. Resource Models
- **Capacity Levels**: `LOW_RESOURCE`, `STANDARD`, `HIGH_RESOURCE`
- **`WorkerResourceProfile`**: Tracks CPU cores, memory budget (MB), max context tokens (strictly $\le 512$), concurrency limits, and current active task load.
- **`TaskResourceRequirements`**: Specifies minimum capacity level, memory requirements, and context budget.

## 3. Placement Algorithm
`ResourceAwareWorkerSelector.select_best_worker()` deterministically scores eligible workers:
- Rejects unhealthy, mismatched, or overloaded workers.
- Avoids over-allocating `HIGH_RESOURCE` nodes for `STANDARD` tasks.
- Prioritizes workers with lower load and breaks ties deterministically by `worker_id`.

## 4. Verification
Verified in `test_01_step115_resource_aware_selection` and EXP 1 of the Step 120 unified benchmark.
