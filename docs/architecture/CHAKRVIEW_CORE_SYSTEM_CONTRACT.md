# ChakrView Core System Contract

**Document Status**: Canonical Architecture Contract  
**Version**: `v0.5.0-canonical`  
**Baseline Commit**: `4a4aba9` (`feat(qa): prepare auditable founder review and admission gate`)  
**Scope**: Reusable, from-scratch, hardware-adaptive intelligence and work-completion system.

---

## 1. Executive Vision and System Identity

ChakrView is an indigenous, from-scratch, hardware-adaptive cognitive intelligence core and autonomous work-completion engine. It is explicitly **not** a wrapper around external pre-trained Large Language Models, not a hardcoded chatbot, and not a single-purpose coding assistant.

### 1.1 The Coding Test Domain Principle
Software engineering and repository navigation are utilized strictly as an **early, demanding, and measurable test domain**. Coding provides an objective arena with deterministic compilation, type systems, abstract syntax trees, regression test suites, and strict verification gates. However, coding does not define ChakrView's ultimate identity. The core engine is designed as a domain-agnostic cognitive system capable of knowledge acquisition, multi-hop reasoning, stateful planning, tool synthesis, and resilient task completion across arbitrary domains.

### 1.2 Canonical Neural Invariants
1. **From-Scratch Construction**: The canonical neural core is built entirely without third-party pre-trained weights, Hugging Face model wrappers, or external black-box weights.
2. **CPU-First Canonical Path**: Primary correctness, deterministic validation, mathematical reproducibility, and canonical baseline execution must run natively on standard CPU hardware without mandatory hardware accelerators.
3. **Immutability of Frozen Canonical Weights**: In production and evaluation phases, canonical neural weights remain strictly immutable ($\Delta W = 0$). All non-parametric knowledge acquisition, user affirmations, and contextual updates are tracked via audited cognitive structures (such as candidate ledgers, working memory, and episodic stores) rather than uncontrolled neural weight drift.
4. **Separation of Cognitive Layers**: Neural core representation, tokenizer, working memory, long-term retrieval, domain adapters, tool interfaces, task orchestration, and physical hardware execution maintain rigid interfaces with zero circular dependencies.

```
+-----------------------------------------------------------------------------------+
|                            ChakrView System Stack                                 |
+-----------------------------------------------------------------------------------+
|  Interface & Interaction: CLI, Interactive Console, Work Completion Interfaces    |
+-----------------------------------------------------------------------------------+
|  Task Orchestration & Governance: Adaptive Decomposition, Resumption, Verification|
+-----------------------------------------------------------------------------------+
|  Cognitive Subsystem: Relational Grounding, Planning, Working & Episodic Memory   |
+-----------------------------------------------------------------------------------+
|  Domain Modules & Tools: Dynamic Adapters, Parsers, Synthesis, Sandbox Tools      |
+-----------------------------------------------------------------------------------+
|  Canonical Neural Core (ChakrMicro): Autoregressive Representation & Tokenizer    |
+-----------------------------------------------------------------------------------+
|  Hardware Adaptation Engine: Resource Governance, Dynamic Profiles, Device Fallback|
+-----------------------------------------------------------------------------------+
|  Physical Compute Layer: Heterogeneous Nodes (CPU / Discrete GPU / iGPU / OS)     |
+-----------------------------------------------------------------------------------+
```

---

## 2. Core Intelligence and Modularity Contract

### 2.1 Component Boundaries
Each architectural layer within ChakrView must adhere to strict boundary separation:

| Subsystem | Canonical Responsibility | Forbidden Behaviors |
| :--- | :--- | :--- |
| **Neural Core (`ChakrMicro`)** | Autoregressive token sequence modeling, internal representation geometry, compositional latent attention. | Calling system APIs, mutating disk files, holding durable application state, importing external proprietary weights. |
| **Tokenizer (`BPETokenizer`)** | Deterministic UTF-8 lossless byte/subword encoding, fixed vocabulary mapping ($V=4096$). | Non-deterministic decoding, mutating vocabulary post-freeze, language-specific heuristic coupling. |
| **Working Memory (`WorkingMemory2`)** | Bounded, task-scoped hypothesis and evidence scratchpad, contradiction isolation. | Unbounded heap growth, silent cross-task pollution, replacing verified ground truth with hallucinated beliefs. |
| **Retrieval & Evidence (`HybridRetriever`)**| Multi-source indexing, lexical BM25 + dense neural embedding lookup, provenance tracking. | Silently dropping citation locators, mutating source corpus files, assuming unverified passages are facts. |
| **Cognitive Planner (`HierarchicalPlanner`)** | Goal decomposition into DAG subgoals (`RETRIEVE`, `VERIFY`, `SYNTHESIZE`), dependency resolution. | Direct execution of destructive host shell commands without tool sandbox gating, treating plans as finished work. |
| **Domain Modules (`DomainAdapter`)** | Syntactic AST parsing, domain-specific prompt realization, specialized linguistic rules. | Hardcoding domain assumptions into the neural core, bypassing global security sandboxes. |
| **Tool Execution Engine** | Sandboxed execution of verified operations with strict timeout and permission checks. | Unrestricted host filesystem access, network egress without explicit policy authorization. |
| **Hardware Adaptation Engine** | Hardware capability discovery, resource governor, dynamic batch/thread/chunk scheduling. | Conflating hardware device capabilities with cognitive reasoning ability or altering model weights based on compute. |

### 2.2 Honest Scientific Status
Future design goals, research prototypes, and architectural aspirations must never be documented or advertised as functional production implementations. Every documented capability must be accompanied by its verifiable code path, regression test suite, and operational limitations.

---

## 3. Hardware Adaptation and Usable Device Verification

### 3.1 The Fundamental Discovery Law
$$\mathbf{DETECTION} \neq \mathbf{USABLE\ CAPABILITY} \neq \mathbf{PERMITTED\ EXECUTION}$$

A hardware probe that reveals the presence of a hardware device (e.g., an NVIDIA GPU via PCI bus or `nvidia-smi`) does **not** constitute proof of usable execution capability. Usable execution requires:
1. Operational host device drivers.
2. A compatible, functional compute runtime (e.g., PyTorch built with CUDA support matching device compute capability).
3. Sufficient unreserved device memory (VRAM) to fit model weights and activation buffers.
4. Active runtime validation (a successful probe tensor allocation and execution).
5. Explicit administrative permission to utilize the accelerator.

If any check fails, the system must deterministically degrade to safe CPU-first execution without crashing.

### 3.2 Hardware Detection Metrics
The hardware discovery subsystem must empirically detect and record:
- **Processor**: Model string, architecture (`x86_64`, `ARM64`), physical cores, logical execution threads, and hardware vector instruction capabilities (AVX2, AVX-512, NEON).
- **Host Memory (RAM)**: Total physical RAM, currently available RAM, process working set limits, and paging pressure.
- **Hardware Accelerators (GPU/iGPU)**: Discrete GPU vs. integrated GPU (shared system memory), driver presence, compute API availability (CUDA, ROCm, Metal/MPS, Vulkan), total VRAM, and unallocated VRAM.
- **Storage Subsystem**: Available disk volume, write throughput, and checkpoint I/O latency.
- **Operating System Environment**: Windows, Linux, macOS, or Android; battery status, AC power connection, thermal throttling flags, and background process priority.

### 3.3 Dynamic Execution Profiles
Hardware profiles govern execution strategies without altering model weights or cognitive identity:
- `CPU_ONLY`: Minimal baseline; single-process execution, vectorized CPU math, serialized operations.
- `CPU_HIGH_MEMORY`: Multi-threaded CPU execution (capped at physical cores), in-memory caching of search indices, expanded token context buffers.
- `GPU_ACCELERATED`: High-throughput tensor batching on dedicated accelerator with pre-allocated VRAM buffers and asynchronous transfer.
- `GPU_MEMORY_CONSTRAINED`: Small batch tensor operations on accelerator, immediate activation buffer eviction, streaming chunks from host RAM.
- `IGPU_SHARED_MEMORY`: Conservative CPU-first preference to protect unified host RAM from GPU driver starvation.
- `LOW_RESOURCE`: Aggressive serialization, minimal 1-thread execution, maximum disk swap, strict memory quotas.
- `BALANCED`: Standard desktop profile balancing responsive UI latency and background inference.

---

## 4. Resource Governance Contract

### 4.1 The Core Governance Axiom
$$\mathbf{AVAILABILITY\ IS\ NOT\ PERMISSION,\ AND\ PERMISSION\ IS\ NOT\ REQUIREMENT}$$

The host machine belongs to the human user. ChakrView is a guest process and must respect host stability above all computational objectives.

### 4.2 Governance Invariants
1. **Hard Upper Resource Budgets**: ChakrView must never reserve or consume 100% of host RAM, VRAM, or CPU threads. Default policies cap host memory usage to a safe fraction (e.g., $\le 50\%$ available RAM in standard mode, $\le 15\%$ during interactive user workloads).
2. **Foreground Workload Protection (Cooperative Yielding)**: When interactive user activity is detected, ChakrView must immediately yield CPU cycles, throttle background task threads to $\le 1$, reduce memory footprints, and pause intensive workloads.
3. **Minimum Sufficient Allocation**: Tasks must be assigned the lowest resource footprint capable of fulfilling their completion criteria within target latency bounds.
4. **No Silent Permission Expansion**: If a task requires more memory, disk, or network privileges than authorized, the system must **fail safely or pause with an honest blocker report**. It must never silently expand its permissions, spawn unmanaged processes, or bypass sandbox constraints.
5. **Separation of Estimates vs. Observations**: Pre-execution resource estimates and post-execution empirical telemetry must be tracked and logged as distinct variables to calibrate future planning.

---

## 5. Adaptive Task Decomposition and Chunking Contract

### 5.1 Decomposition Strategies
When presented with a complex objective or large data workload, the system must evaluate its resource budget and choose among four deterministic execution strategies:

```
                          [Incoming Task Payload]
                                     |
                          [Evaluate Budget & Size]
                                     |
               +---------------------+---------------------+
               |                                           |
     Fits Budget & Low Overhead               Exceeds Budget or Modular
               |                                           |
      (SINGLE-PASS EXECUTION)                              |
                                           +---------------+---------------+
                                           |                               |
                                  Linear Context Flow             Independent Subgoals
                                           |                               |
                                 (SEQUENTIAL CHUNKING)          (PARALLEL DECOMPOSITION)
                                           |                               |
                                           +---------------+---------------+
                                                           |
                                                Execution Fault or Drift?
                                                           |
                                               (ADAPTIVE RE-PLANNING)
```

1. **Single-Pass Execution**: Selected when the payload and working context fit comfortably within allocated memory and context length limits, and decomposition provides no structural benefit.
2. **Sequential Chunking**: Selected when total workload size exceeds context length or RAM budgets, requiring streaming processing where intermediate state and rolling context are checkpointed sequentially.
3. **Parallel Subtasks**: Selected when subgoals have zero mutual causal dependencies and expected speedup substantially exceeds thread synchronization andIPC overhead.
4. **Adaptive Re-Planning**: Triggered dynamically when runtime resource consumption, execution duration, or error rates deviate significantly from estimates.

### 5.2 Dynamic Chunk Sizing
Chunk sizes are **not universal hardcoded constants**. They must be computed dynamically as a function of:
- Task modality (code analysis, natural language reading, relational triple extraction).
- Available RAM/VRAM budget.
- Context window capacity.
- Retry and failure isolation cost.

### 5.3 Subtask State Invariants
Every subtask or chunk must maintain an immutable descriptor:
- Unique chunk/subtask identifier (`chunk_id`, `subgoal_id`).
- Explicit inputs, schemas, and source locators.
- Declared dependencies (`depends_on`).
- Explicit output schemas and verification assertions.
- Allocated resource budget (threads, timeout, memory cap).
- Current status (`PENDING`, `RUNNING`, `COMMITTED`, `FAILED`, `BLOCKED`).

---

## 6. Heterogeneous Distributed Execution Contract

### 6.1 Heterogeneous Node Support
The intended distributed architecture encompasses heterogeneous worker nodes across operating systems (Windows, Linux, macOS, Android), architectures (`x86_64`, `ARM64`), and compute configurations.

### 6.2 Distributed Invariants and Operational Rules
1. **Capability-Gated Scheduling**: Workloads are scheduled only to nodes that have affirmatively demonstrated compatible runtimes, sufficient resource profiles, and security trust certificates.
2. **Accounting for Network Cost**: Distributed dispatch is authorized only when the estimated compute savings clearly exceed data serialization, network transit latency, and remote coordinator reconciliation overhead.
3. **Dynamic Topology (Churn Tolerance)**: Nodes may join, degrade, disconnect, or rejoin without crashing the coordinator or invalidating overall task state.
4. **Idempotency and Side-Effect Isolation**: Retried remote subtasks must either be strictly idempotent or execute in sandboxed staging environments to prevent duplicate side effects (e.g., duplicated filesystem writes or database commits).
5. **Local Fallback Guarantee**: If all remote workers become unreachable, the coordinator must assess whether the remaining workload can be completed locally. If local resources permit, execution must continue locally without user disruption.
6. **Coordinator Resumption & State Authority**: Task progress must be journaled to durable local storage. A restarted coordinator must be capable of recovering uncommitted task state and verifying worker claims before marking tasks complete.
7. **Strict Architectural Separation**:
   - **Distributed Task Execution**: Distributing independent work items or subgoals across compute nodes.
   - **Distributed Neural Training**: Synchronizing gradient steps and weight updates across a cluster (explicitly outside canonical runtime scope).
   - **Model Parallelism**: Sharding layer weights of a single neural core across multiple physical devices.
   - **Shared Cognitive State**: Synchronizing relational triples and working memory across agents.

---

## 7. Memory, Checkpointing, and Recovery Contract

### 7.1 Resumption Principle
$$\mathbf{CHUNK\ OR\ NODE\ FAILURE} \neq \mathbf{TASK\ FAILURE}$$

A transient failure in a single worker, network socket, or processing chunk must never invalidate already committed, verified progress.

### 7.2 Checkpointing and Durability Rules
1. **Durable Journaling**: Subtask results must be persisted to disk or durable memory immediately upon successful verification.
2. **Latest Valid Committed State**: Upon crash or process restart, the executor resumes execution from the latest valid committed checkpoint, re-executing only pending, failed, or unverified stages.
3. **Protection Against Corrupted State**: All saved checkpoints must store content hashes (SHA-256) and schema versions. A corrupted checkpoint must be detected, quarantined, and safely rolled back to the preceding valid checkpoint.
4. **Duplicate Result Rejection**: Asynchronous or duplicate completions from stale workers must be checked against idempotency keys and safely dropped without causing race conditions.

---

## 8. End-to-End Task Completion and Verification Contract

### 8.1 Completion Criteria
A task is never marked `COMPLETED` based merely on execution loop termination or lack of fatal errors. Completion requires:
1. All planned subgoals or chunks have reached `COMMITTED` status.
2. The synthesized final output matches the requested output schema.
3. Explicit post-execution verification assertions pass (e.g., syntax checks, test execution, schema validation, semantic integrity checks).

### 8.2 Honest Failure Reporting
If a task is blocked by resource exhaustion, missing permissions, invalid input preconditions, or unrecoverable worker failures, the system must immediately halt and generate an auditable incident report containing:
- Specific root-cause blocker.
- Completed and verified subtasks.
- Unfinished subtasks and dependencies.
- Available recovery or remediation actions.

---

## 9. Modular Independence of Task Orchestration

Task planning, scheduling, chunking, and distributed coordination are **modular algorithmic subsystems**. They must **never** be conflated with the autoregressive neural model's intrinsic learned reasoning abilities. 

ChakrView achieves autonomous work completion through the rigorous architectural synthesis of:
1. Compact, indigenous autoregressive representation (`ChakrMicro`).
2. Structured relational semantic grounding (`TextNormalizer`, `CompositionalSentenceParser`).
3. Algorithmic hierarchical planning and verification (`HierarchicalPlanner`, `RecoveryManager`).
4. Hardware-adaptive resource governors (`HardwareAdaptiveGovernor`, `ChunkedAdaptiveExecutor`).

This contract establishes the canonical engineering foundation against which all present implementations and future enhancements are judged.
