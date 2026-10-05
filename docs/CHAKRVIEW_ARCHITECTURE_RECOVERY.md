# ChakrView Architecture Recovery & Current Ratified Subsystems

- **Document Version**: 1.0.0
- **Ratification Date**: October 5, 2026
- **Current Milestone**: Step 103 Completed -> Step 104-109 Wave Initiated
- **Baseline Invariant**: Parameters: `3,443,136` | SHA-256 Digest: `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`

---

## 1. Architectural System Map

```
+-----------------------------------------------------------------------------------------+
|                                    CHAKRVIEW SYSTEM                                     |
+-----------------------------------------------------------------------------------------+
|  [COGNITIVE & REASONING LAYER]                                                          |
|  - Structured Reasoning (Assumptions, Hypotheses, Contradictions, Evidence)             |
|  - Critical Thinking Engine (Dialectical Evaluation, Verification Queries)             |
|  - Unified Cognitive Context (Budget-aware composition, Prompt formatting)             |
+-----------------------------------------------------------------------------------------+
|  [PERSISTENT PROJECT BRAIN (PPB)]                                                       |
|  - Storage Engine (SQLite, WAL mode, Schema migrations, Entity indexes)                 |
|  - Project Knowledge Evolution (Entities, Relations, Lessons, Regression Memory)        |
|  - Persistent Task Graph (DAG, Deterministic fingerprints, Replay log)                  |
+-----------------------------------------------------------------------------------------+
|  [TASK ORCHESTRATION & RESOURCE ADAPTATION]                                             |
|  - Task Planner & Decomposer (Tasks -> Bounded SubTasks with typed dependencies)         |
|  - Resource Classifier & Profiler (LOW_RESOURCE, STANDARD, HIGH_RESOURCE, ACCELERATED)  |
|  - Autonomous Work Loop (Resumable, Lease management, Heartbeat, Restart recovery)      |
+-----------------------------------------------------------------------------------------+
|  [GOVERNED LEARNING & SELF-EVALUATION]                                                  |
|  - Governed Self-Evaluator (Failure analysis, Invalidation taxonomy, Audit reports)     |
|  - Cognitive Strategy Registry (Active/Deprecated strategies, Empirical utility)        |
|  - Evaluation Regression Memory (Capability tracking, Benchmark gates)                  |
+-----------------------------------------------------------------------------------------+
|  [DISTRIBUTED COMPUTING & FEDERATION FOUNDATION]                                        |
|  - Node Identity & Capability Registry (Host specifications, Hardware tags)             |
|  - Deterministic Task Router & Leases (Idempotency, Replay protection, HMAC signing)    |
|  - Pluggable Transport (Loopback in-process, Asynchronous messaging, Failure recovery)  |
+-----------------------------------------------------------------------------------------+
|  [LEARNING & TRAINING INFRASTRUCTURE]                                                   |
|  - ChakrMicro Causal Transformer (6 layers, d_model=192, 6 heads, SwiGLU, RoPE)        |
|  - Byte-Level BPE Tokenizer (Vocab=4096, 256 byte primitives, 3,837 merges)             |
|  - Checkpoint Manager (Atomic write-replace, Model/Tokenizer identity verification)    |
|  - Streaming Sharded Token Dataset (Stage B: 405k tokens; Stage C: 7.7M tokens)        |
|  - Syntactic & Capability Probes (Deterministic 40-probe benchmark, Held-out tests)    |
+-----------------------------------------------------------------------------------------+
```

---

## 2. Subsystem Breakdowns

### A. Neural Layer
- **Purpose**: Low-resource, CPU-first sovereign neural core providing next-token representations and conditioned proposals.
- **Implementation**: [`chakrview.brain.model.ChakrMicro`](file:///d:/Project/ChakrView/chakrview/brain/model.py), [`chakrview.brain.config.ModelConfig`](file:///d:/Project/ChakrView/chakrview/brain/config.py).
- **Invariants**: 6 layers, $d_{\text{model}}=192$, $d_{\text{ff}}=512$, 6 heads, head dim 32, SwiGLU, RoPE ($\theta=10000$), Pre-RMSNorm ($\epsilon=10^{-5}$), tied embeddings, 4096 vocabulary, exactly 3,443,136 parameters.
- **Canonical Baseline Hash**: `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` ($\Delta W \equiv 0$).
- **Experimental Checkpoints**: Isolated under `artifacts/` (e.g. Step 102 candidate `d5886e...`).

### B. Cognitive & Reasoning Layer
- **Purpose**: Explicit reasoning, hypothesis testing, critique, and verification independent of raw token generation.
- **Implementation**: [`chakrview.cognition.reasoning.structured`](file:///d:/Project/ChakrView/chakrview/cognition/reasoning/structured.py), [`chakrview.cognition.reasoning.critical`](file:///d:/Project/ChakrView/chakrview/cognition/reasoning/critical.py), [`chakrview.cognition.unified`](file:///d:/Project/ChakrView/chakrview/cognition/unified).
- **Interfaces**: `ReasoningEngine`, `DialecticalCritiqueEngine`, `UnifiedContextComposer`.
- **Invariants**: Reasoning state is structured data (assumptions, alternatives, contradictions). Neural generation does not execute arbitrary cognitive actions.

### C. Persistent Project Brain (PPB)
- **Purpose**: Single source of durable project memory surviving process restarts.
- **Implementation**: [`chakrview.cognition.ppb.storage.PPBStorage`](file:///d:/Project/ChakrView/chakrview/cognition/ppb/storage.py), [`chakrview.cognition.ppb.task_storage.TaskStorage`](file:///d:/Project/ChakrView/chakrview/cognition/ppb/task_storage.py).
- **Persistent State**: SQLite with WAL mode, indices on entities, relations, tasks, and audit logs.
- **Invariants**: Atomic transactions, idempotent entity updates, zero duplicate file rescans without content hash modification.

### D. Memory
- **Purpose**: Multi-tier memory combining semantic repository memory, conversational memory, episodic experience, and regression tracking.
- **Implementation**: [`chakrview.cognition.repository.memory_index`](file:///d:/Project/ChakrView/chakrview/cognition/repository/memory_index.py), [`chakrview.cognition.governed_learning.evaluation_memory`](file:///d:/Project/ChakrView/chakrview/cognition/governed_learning/evaluation_memory.py).
- **Invariants**: Queryable by symbol, file, or topic; versioned against repository commit hashes.

### E. Task Orchestration
- **Purpose**: Transform high-level objectives into a Directed Acyclic Graph (DAG) of verified subtasks.
- **Implementation**: [`chakrview.cognition.ppb.task_models.PersistentTaskGraph`](file:///d:/Project/ChakrView/chakrview/cognition/ppb/task_models.py), [`chakrview.cognition.ppb.scheduler.TaskScheduler`](file:///d:/Project/ChakrView/chakrview/cognition/ppb/scheduler.py), [`chakrview.cognition.ppb.work_loop.AutonomousWorkLoop`](file:///d:/Project/ChakrView/chakrview/cognition/ppb/work_loop.py).
- **Invariants**: Strict dependency resolution (dependent tasks never run early), leases with heartbeats, bounded retries ($N \le 2$).

### F. Learning & Training
- **Purpose**: Governed, reproducible training runs producing validated candidate checkpoints.
- **Implementation**: [`chakrview.training.trainer.Trainer`](file:///d:/Project/ChakrView/chakrview/training/trainer.py), [`chakrview.training.checkpoint.CheckpointManager`](file:///d:/Project/ChakrView/chakrview/training/checkpoint.py), [`chakrview.training.curriculum_runner`](file:///d:/Project/ChakrView/chakrview/training/curriculum_runner.py).
- **Invariants**: Checkpoints record tokenizer checksum, model config, parameter count, and dataset manifest hash. Training fails loudly on identity mismatch.

### G. Governance & Self-Evaluation
- **Purpose**: Audit task and model performance, classify failure modes, and gate releases.
- **Implementation**: [`chakrview.cognition.governed_learning.self_evaluator.GovernedSelfEvaluator`](file:///d:/Project/ChakrView/chakrview/cognition/governed_learning/self_evaluator.py), [`chakrview.cognition.governed_learning.release_gate.ReleaseGateAuditor`](file:///d:/Project/ChakrView/chakrview/cognition/governed_learning/release_gate.py).
- **Invariants**: Zero autonomous runtime weight updates. Only governed, verifiable experiments may propose candidate models.

### H. Tool & Environment Interface
- **Purpose**: Safe inspection, verification, and atomic patch application.
- **Implementation**: [`chakrview.cognition.tool_gate.ToolGate`](file:///d:/Project/ChakrView/chakrview/cognition/tool_gate.py), `PatchPlan`, `SafePatchExecutor`.
- **Invariants**: Read-only tools cannot modify state; write tools require explicit validation, patch planning, and atomic rollback capability.

### I. Resource Adaptation
- **Purpose**: Detect host hardware profiles and throttle execution parameters to keep ChakrView functioning on low-spec hardware.
- **Implementation**: [`chakrview.cognition.adaptation.hardware.HardwareProfiler`](file:///d:/Project/ChakrView/chakrview/cognition/adaptation/hardware.py), [`chakrview.cognition.adaptation.policy.AdaptiveExecutionPolicy`](file:///d:/Project/ChakrView/chakrview/cognition/adaptation/policy.py).
- **Invariants**: Hardware detection adjusts token budgets, batch sizes, worker counts, and thinking steps without mutating neural model architecture or invariants.

### J. Distributed & Federation Foundation
- **Purpose**: Support multi-node worker clusters and federated cognition without coupling to a specific transport.
- **Implementation**: [`chakrview.cognition.distributed.engine.DistributedFederatedEngine`](file:///d:/Project/ChakrView/chakrview/cognition/distributed/engine.py), [`chakrview.cognition.distributed.registry.DistributedNodeRegistry`](file:///d:/Project/ChakrView/chakrview/cognition/distributed/registry.py).
- **Invariants**: Node != Authority. Zero weight mutation across nodes. Cryptographic HMAC/nonce verification. Idempotent task dispatch.
