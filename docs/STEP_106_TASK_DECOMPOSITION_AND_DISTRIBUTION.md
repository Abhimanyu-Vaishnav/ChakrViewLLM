# ChakrView Step 106: Task Decomposition & Distributed Execution Foundation

- **Milestone Designation**: Step 106 (Hierarchical Task Decomposition, Dependency Sequencing & Distributed Abstraction)
- **Status**: COMPLETE & RATIFIED
- **Date**: October 5, 2026
- **Architecture Principle**: Sovereign Worker Nodes, Strict Topological Sequencing, Idempotent Execution

---

## 1. Large-Task Problem: Big Goals on Constrained Machines

ChakrView rejects monolithic script execution. High-level user objectives are systematically converted into a Directed Acyclic Graph (DAG) of discrete, typed, bounded subtasks:

```
[ HIGH-LEVEL OBJECTIVE ]
           |
           v
 [ Task Decomposition Engine ]
           |
  +--------+--------+
  |                 |
[ SubTask A ]     [ SubTask B ]
  |                 |
  +--------+--------+
           |
           v
     [ SubTask C ]  (Depends on A + B)
           |
           v
     [ Verification ]
```

---

## 2. Persistent Task Graph Lifecycle & Invariants

Defined in [`chakrview.cognition.ppb.task_models`](file:///d:/Project/ChakrView/chakrview/cognition/ppb/task_models.py):

1. **Deterministic Fingerprints**: Every task node generates a content-derived SHA-256 fingerprint:
   $$\text{Fingerprint} = \text{SHA256}(\text{graph\_id} \mathbin{\Vert} \text{title} \mathbin{\Vert} \text{resource\_type} \mathbin{\Vert} \text{sorted(deps)} \mathbin{\Vert} \text{sorted(files)})$$
2. **Cycle Prevention**: Kahn's algorithm validates cycle freedom on every node addition (`validate_no_cycles`).
3. **Topological Eligibility**: A node enters `READY` status if and only if all prerequisite dependencies have status `COMPLETED`.
4. **Failure Propagation**: If node $X$ fails, all downstream dependents are immediately transitioned to `BLOCKED` with an explicit failure reason, preventing premature execution.
5. **Bounded Retries**: Maximum retries are strictly bounded ($\le 2$), after which the node transitions to `FAILED`.

---

## 3. Low-Resource Decomposition Strategy ("Big Task, Tiny Machine")

When task resource demands exceed host context limits (e.g. $\ge 3$ files or estimated tokens $> \text{max\_context}$ on `LOW_RESOURCE` hardware):
1. The [`ContextResourcePlanner`](file:///d:/Project/ChakrView/chakrview/cognition/ppb/budget_planner.py) triggers action `SPLIT_TASK`.
2. The task is decomposed into single-file bounded child subtasks (`DynamicSubtaskRequest`).
3. Intermediate artifacts are persisted to PPB SQLite storage.
4. Each chunk is independently executed and verified, keeping resident memory usage $< 256$ MB RAM.

---

## 4. Distributed Federation Abstraction

Defined in [`chakrview.cognition.distributed`](file:///d:/Project/ChakrView/chakrview/cognition/distributed):
- **Node vs Authority**: Compute nodes (`PRIMARY`, `WORKER`, `ARBITER`) provide processing capacity but hold zero authority to modify neural weights or bypass capability gates.
- **Lease Management & Heartbeats**: Workers acquire bounded execution leases (`TaskLease`). Stale workers expire automatically.
- **Cryptographic Envelopes**: All inter-node messages use HMAC signing, replay protection trackers, and monotonic nonces.
- **Pluggable Transport**: Supports `LoopbackTransport` (in-process multi-worker simulation) and network transports without rewriting orchestration logic.
