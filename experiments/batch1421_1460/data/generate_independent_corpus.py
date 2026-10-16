"""
Generator for the independent, multi-domain evaluation corpus for Batch 1421–1460.
Location: experiments/batch1421_1460/data/generate_independent_corpus.py

Creates 30 new, independent technical documents (approx. 4,500 target tokens) across 5 categories:
1. os_and_architecture (6 docs)
2. python_and_concurrency (6 docs)
3. distributed_and_consensus (6 docs)
4. mathematical_and_numerical (6 docs)
5. explanatory_technical_prose (6 docs)

Outputs corpus files to experiments/batch1421_1460/data/corpus/*.md
Outputs manifest to experiments/batch1421_1460/data/independent_corpus_manifest.json
"""
import hashlib
import json
from pathlib import Path

OUT_DIR = Path("experiments/batch1421_1460/data/corpus")
MANIFEST_PATH = Path("experiments/batch1421_1460/data/independent_corpus_manifest.json")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOCUMENTS = [
    # Category 1: os_and_architecture
    ("os_virtual_memory_paging.md", "os_and_architecture", """# Virtual Memory Management and Page Table Translation

## 1. Paging Architecture
- Hardware page table traversal: Four-level hierarchical page table structure with 48-bit canonical virtual address spaces.
- Translation Lookaside Buffer (TLB): Two-tiered split L1 data and instruction TLB with 64 fully associative entries.
- Page table walk latency: 14 to 28 nanoseconds on hardware misses before falling back to physical DRAM pages.
- Large page support: Transparent Huge Pages allocate contiguous 2 megabyte physical frames to reduce TLB misses.
- Page replacement algorithm: Generalized clock-pro algorithm approximating least recently used page evictions.
"""),
    ("os_interrupt_handling_apic.md", "os_and_architecture", """# Advanced Programmable Interrupt Controller and Vector Dispatch

## 1. Interrupt Subsystems
- Vector allocation: Fixed vector ranges from 32 to 255 allocated across symmetric multicore processing units.
- Non-maskable interrupt (NMI) delivery: Dedicated physical pin triggering panic handlers during uncorrectable ECC faults.
- Inter-processor interrupts (IPI): Directed unicast signals coordinate cache invalidation shootdowns across CPU sockets.
- Interrupt mitigation window: Dynamic interrupt coalescing delays packet arrivals by 50 microseconds under high throughput.
- Direct Memory Access (DMA): Ring buffer circular descriptors receive raw packets without CPU intervention.
"""),
    ("os_process_scheduler_cfs.md", "os_and_architecture", """# Completely Fair Scheduler and Task Hierarchy

## 1. Scheduling Mechanics
- Red-black tree hierarchy: Runnable threads indexed by virtual runtime nanoseconds tracking CPU resource fairness.
- Target latency window: Minimum scheduling granularity capped at 0.75 milliseconds across active thread pools.
- Thread affinity masking: Bitmask pinning confines latency-critical workers to physical CPU cores with isolated L3 caches.
- Dynamic priority weighting: Nice values mapped logarithmically across 40 priority levels scaling execution time slices.
- Context switch overhead: Thread yield takes approximately 1.2 microseconds per voluntary synchronization call.
"""),
    ("arch_cache_coherence_mesi.md", "os_and_architecture", """# MESI Cache Coherence Protocol and Snoop Filters

## 1. Coherence States
- Four discrete states: Modified, Exclusive, Shared, and Invalid states govern multicore cache line transactions.
- Snoop bus broadcast: Cache write invalidations propagate across bidirectional ring interconnects within 8 clock cycles.
- False sharing mitigation: Struct members padded to 64 bytes prevent concurrent invalidation of adjacent fields.
- Write-back buffer allocation: Dirty lines persist in non-volatile write caches before commit to memory controllers.
- Memory barrier primitives: Full fence instructions prevent out-of-order store reordering across speculative pipelines.
"""),
    ("arch_branch_prediction_tage.md", "os_and_architecture", """# TAGE Branch Predictor and Speculative Execution

## 1. Predictor Structures
- Tagged geometric history: Multiple predictor tables indexed by geometric progression of global branch histories.
- Branch Target Buffer (BTB): 4096-entry associative array caches indirect branch target addresses for immediate fetch.
- Return Stack Buffer (RSB): 32-entry circular stack predicts subroutine return pointers with 99.4 percent accuracy.
- Misprediction penalty: Pipeline flush incurs 16 clock cycles of instruction re-fetch from primary instruction caches.
- Speculative window depth: Reorder buffer tracks up to 256 instructions in flight awaiting in-order retirement.
"""),
    ("arch_pcie_bus_dma.md", "os_and_architecture", """# PCI Express Generation 5 Bus Protocol and Direct Memory

## 1. Interconnect Performance
- Raw signaling rate: 32 gigatransfers per second per lane using 128b/130b encoding schemes.
- Total duplex bandwidth: 16-lane configurations deliver 64 gigabytes per second in each direction simultaneously.
- Packet header protocol: Transaction Layer Packets (TLP) carry 32-bit and 64-bit physical memory addresses.
- Flow control credit units: Receiver buffer updates transmit periodic credit advertisements preventing packet drops.
- Root complex arbitration: Strict round-robin scheduling balances peer-to-peer traffic across endpoint devices.
"""),

    # Category 2: python_and_concurrency
    ("py_gil_multithreading_internals.md", "python_and_concurrency", """# Global Interpreter Lock Implementation and Mutex Internals

## 1. Concurrency Model
- GIL mutex mechanism: Single operating system condition variable coordinates thread release after 5 millisecond intervals.
- Bytecode instruction counter: Check intervals force thread evaluation switches every 100 virtual machine instructions.
- Sub-interpreter isolation: Per-interpreter GIL allocations enable parallel execution without shared state memory.
- Atomic reference counting: Py_INCREF and Py_DECREF primitives maintain deterministic garbage collection lifetimes.
- Free-threaded execution: Specialized build flags disable global lock mechanisms using per-object bi-directional mutexes.
"""),
    ("py_asyncio_event_loop.md", "python_and_concurrency", """# Asyncio Event Loop and Coroutine Scheduling

## 1. Loop Architecture
- I/O multiplexing engine: Epoll and Kqueue selector backends poll ready file descriptors with zero CPU idle consumption.
- Task wrapper abstractions: Coroutine frames suspended via yield expressions wrapped inside Task task futures.
- Callback execution order: Ready callbacks execute sequentially before timers and pending network socket signals.
- Custom loop exception handlers: Unhandled coroutine exceptions routed through configurable global logging callbacks.
- ThreadPoolExecutor offloading: Blocking CPU-bound routines dispatched onto background thread pools via run_in_executor.
"""),
    ("py_memory_allocator_pymalloc.md", "python_and_concurrency", """# Pymalloc Small Object Memory Arena Hierarchy

## 1. Allocation Tiers
- Small object ceiling: Allocations under 512 bytes handled by specialized arena allocators to reduce fragmentation.
- Arena sizing: Physical 256 kilobyte arenas subdivided into 4 kilobyte memory pools dedicated to uniform size classes.
- Size class indexing: 64 distinct size classes incrementing by 8 bytes satisfy requests from 8 to 512 bytes.
- Large object fallback: Memory requests exceeding 512 bytes route directly to standard system malloc routines.
- Free list recycling: Singly linked lists maintain vacant blocks within active pools for immediate re-allocation.
"""),
    ("py_bytecode_interpreter_ceval.md", "python_and_concurrency", """# Bytecode Dispatch Loop and Frame Execution

## 1. Interpreter Engine
- Opcode evaluation loop: Optimized indirect switch statements dispatch 256 distinct bytecode instruction opcodes.
- Value stack organization: Fast local variable lookups indexed directly via tuple offsets rather than hash tables.
- Specialized instructions: Dynamic quickening replaces generic opcodes with type-specialized versions after execution.
- Exception table mapping: Zero-cost exception handling maps bytecode offset ranges to target handler blocks without overhead.
- Frame pointer linkage: Chained call frames allocate local variables and execution stacks contiguously in memory.
"""),
    ("py_multiprocessing_shared_memory.md", "python_and_concurrency", """# Shared Memory and Inter-Process Communication

## 1. Process Coordination
- Posix shared memory segments: Named memory regions mapped into distinct process address spaces via shm_open calls.
- Raw byte buffers: SharedMemory abstractions wrap memory-mapped files supporting cross-process NumPy array buffers.
- Synchronization primitives: Inter-process Semaphores and Locks backed by operating system futex implementations.
- Zero-copy deserialization: Reading shared memory buffers avoids pickling overhead during high-volume data exchange.
- Process pool management: Forkserver and spawn start methods prevent resource leaks and thread deadlocks during initialization.
"""),
    ("py_garbage_collection_generational.md", "python_and_concurrency", """# Generational Cyclic Garbage Collection Subsystems

## 1. Collector Architecture
- Three collection generations: Objects segregated into Generation 0, Generation 1, and Generation 2 by survival age.
- Collection thresholds: Generational triggers execute automatic sweeps when allocation counts exceed 700 new objects.
- Double-linked object lists: Tracked containers maintain prev and next pointers within generational linked list nodes.
- Reference cycle resolution: Trial deletion algorithm decrements internal reference counts to isolate orphaned cycle clusters.
- Finalizer execution order: Objects with custom __del__ methods collected safely without breaking dependency graphs.
"""),

    # Category 3: distributed_and_consensus
    ("dist_raft_log_replication.md", "distributed_and_consensus", """# Raft Consensus Protocol Log Replication Invariants

## 1. Protocol Specifications
- Leader election timeout: Randomized election timers span 150 to 300 milliseconds preventing split-vote deadlocks.
- AppendEntries RPC protocol: Heartbeat messages transmit preceding log index and term to enforce log consistency.
- Quorum commit requirement: Entries committed once persisted on a strict majority of active cluster voter nodes.
- Log matching property: Identical index and term entries guarantee identical preceding log sequences across all nodes.
- State machine application: Committed log entries apply deterministically to local state machines in index sequence.
"""),
    ("dist_vector_clocks_causality.md", "distributed_and_consensus", """# Vector Clocks and Distributed Causal Ordering

## 1. Clock Mechanics
- Logical vector arrays: Monotonically increasing clock vectors track causal event dependencies across distributed nodes.
- Partial order comparison: Vector A strictly precedes Vector B if all components are less than or equal to Vector B.
- Concurrent write detection: Incomparable clock vectors indicate simultaneous modifications requiring conflict resolution.
- Clock pruning mechanisms: Stale node identifiers purged from vector maps after predefined time-to-live thresholds.
- Anti-entropy sync: Background gossip exchanges exchange vector state to reconcile diverged distributed data stores.
"""),
    ("dist_paxos_two_phase_commit.md", "distributed_and_consensus", """# Multi-Paxos Coordination and Phase Consensus

## 1. Paxos Phases
- Phase 1 Prepare requests: Proposers broadcast unique proposal numbers to acquire promise quorums from acceptors.
- Phase 2 Accept requests: Acceptors commit proposed values provided no higher proposal number was acknowledged.
- Leader lease guarantees: Monotonic epoch leases grant stable leaders master authority without per-query consensus.
- Fast Paxos optimization: Direct client broadcasts achieve consensus within two network round trips under zero contention.
- Catch-up streaming: Trailing nodes stream missing consensus entries from snapshot archives via dedicated TCP sessions.
"""),
    ("dist_distributed_transactions_2pc.md", "distributed_and_consensus", """# Two-Phase Commit Distributed Transaction Engine

## 1. Transaction Protocol
- Phase 1 Prepare voting: Transaction coordinator requests commit votes from all participating database resource managers.
- Phase 2 Commit execution: Coordinator broadcasts global commit decisions only if all participants return affirmative votes.
- Write-ahead logging: Coordinators and participants log state transitions to non-volatile disk before message dispatch.
- Coordinator failure recovery: Presumed abort protocols abort in-flight transactions when coordinator status is ambiguous.
- Timeout resolution: Participant locks remain held during coordinator partitions awaiting resolution protocols.
"""),
    ("dist_gossip_swim_membership.md", "distributed_and_consensus", """# SWIM Gossip Protocol and Node Failure Detection

## 1. Protocol Dynamics
- Periodical ping probes: Random peer selection pings target nodes every 200 milliseconds to verify cluster presence.
- Indirect ping fallback: Unresponsive targets audited through 3 intermediary peers before declaring node suspicion.
- Suspicion mechanism: Suspicious nodes given 5 second grace periods to refute failure claims before global broadcast.
- Piggybacked metadata: Protocol messages carry membership state updates reducing dedicated network bandwidth consumption.
- Scale-free convergence: Failure notifications propagate across 1000 nodes within logarithmic round bounds.
"""),
    ("dist_consistent_hashing_ring.md", "distributed_and_consensus", """# Consistent Hashing Rings and Virtual Node Topologies

## 1. Partitioning Strategy
- Ring identifier space: 64-bit integer ring space coordinates partition assignment across distributed storage clusters.
- Virtual node mapping: Physical servers allocated 128 virtual nodes across ring positions to balance key distribution.
- Key lookup complexity: Binary search across sorted token arrays locates owning partition nodes in logarithmic time.
- Node join rebalancing: New participant additions migrate only fractional key ranges from adjacent ring neighbors.
- Replication factor: Successive distinct physical nodes along the ring store redundant data replicas for fault tolerance.
"""),

    # Category 4: mathematical_and_numerical
    ("math_matrix_multiplication_gemm.md", "mathematical_and_numerical", """# General Matrix Multiplication Algorithms and Cache Blocking

## 1. Matrix Algorithms
- Strassen algorithm complexity: Recursive sub-matrix partitioning reduces matrix multiplication complexity to O(N^2.807).
- Cache blocking tiles: Inner loops tiled into L1 and L2 cache resident blocks maximizing register reuse efficiency.
- Vectorized SIMD register packing: AVX-512 FMA instructions compute 32 floating point multiply-accumulate ops per cycle.
- Memory access strides: Transposed matrix layouts ensure unit stride contiguous memory access along inner reduction loops.
- Numerical precision modes: Mixed precision FP16 accumulation into FP32 registers preserves dynamic range without overflow.
"""),
    ("math_svd_principal_components.md", "mathematical_and_numerical", """# Singular Value Decomposition and Low-Rank Approximation

## 1. Decomposition Formula
- Matrix factorization: Any real matrix decomposes into orthogonal matrices U and V with diagonal singular matrix Sigma.
- Eckart-Young theorem: Truncated SVD yields the optimal rank-k approximation minimizing Frobenius matrix reconstruction error.
- Lanczos iteration: Krylov subspace projections extract leading singular values for massive sparse coordinate matrices.
- Explained variance ratio: Cumulative sum of squared singular values quantifies information retention across principal components.
- Condition number assessment: Ratio of maximum to minimum singular value determines matrix inversion numerical sensitivity.
"""),
    ("math_gradient_descent_optimizers.md", "mathematical_and_numerical", """# Gradient Descent Convergence and Momentum Mechanics

## 1. Optimization Dynamics
- Nesterov accelerated gradient: Look-ahead momentum updates prevent overshooting along high-curvature ravines.
- Adam optimizer formulation: Exponential moving averages of first and second gradient moments guide parameter updates.
- Learning rate warmup: Linear step increases stabilize early optimization trajectory before cosine learning rate decay.
- Weight decay decoupling: True AdamW separates L2 regularization penalties from adaptive gradient moving averages.
- Polyak averaging: Exponential moving average of model checkpoints yields smoother optimization loss landscapes.
"""),
    ("math_fourier_transform_fft.md", "mathematical_and_numerical", """# Fast Fourier Transform and Cooley-Tukey Radix

## 1. Frequency Analysis
- Algorithmic complexity: Divide-and-conquer decomposition reduces Discrete Fourier Transform from O(N^2) to O(N log N).
- Twiddle factor calculation: Complex roots of unity precomputed and cached in lookup tables to accelerate butterfly stages.
- Bit-reversal permutation: Input arrays reordered via bit-reversal sorting before in-place butterfly computation steps.
- Cooley-Tukey radix-2: Decimation-in-time splits even and odd index points into recursive half-size transform kernels.
- Circular convolution theorem: Pointwise frequency domain multiplication performs exact cyclic time domain convolutions.
"""),
    ("math_monte_carlo_markov_chain.md", "mathematical_and_numerical", """# Markov Chain Monte Carlo and Metropolis-Hastings

## 1. Sampling Theory
- Target distribution convergence: Ergodic Markov chains guarantee stationary distribution convergence as iterations approach infinity.
- Acceptance probability: Ratio of proposal probabilities determines sample acceptance preserving detailed balance conditions.
- Burn-in period discarding: Initial transition samples discarded to eliminate starting point state bias from distribution metrics.
- Hamiltonian Monte Carlo: Continuous vector field simulation guides efficient proposal trajectories through complex posterior spaces.
- Effective sample size: Autocorrelation time calculations adjust raw sample counts to quantify statistical independence.
"""),
    ("math_eigenvalues_power_iteration.md", "mathematical_and_numerical", """# Power Iteration and Rayleigh Quotient Formulations

## 1. Spectral Analysis
- Dominant eigenvalue convergence: Repeated matrix-vector multiplications project arbitrary non-zero vectors onto principal eigenvectors.
- Convergence rate: Ratio of second to first eigenvalue determines geometric convergence speed per iteration step.
- Rayleigh quotient acceleration: Optimal scalar approximations computed at each iteration minimizing Euclidean vector residuals.
- Deflation techniques: Rank-one updates subtract dominant eigenspaces enabling iterative discovery of sub-dominant eigenvalues.
- Shifted power method: Spectral shift constants accelerate eigenvalue separation for clusters of near-identical magnitudes.
"""),

    # Category 5: explanatory_technical_prose
    ("prose_fault_tolerant_system_design.md", "explanatory_technical_prose", """# Principles of Fault Tolerant Systems Engineering

## 1. Architectural Resilience
- Redundancy strategies: Dual modular and triple modular redundancy isolate localized hardware faults without service degradation.
- Graceful degradation: Systems shed optional processing stages under severe load to protect primary transaction guarantees.
- Circuit breaker patterns: Automated trip thresholds isolate failing downstream dependencies preventing cascading network failures.
- Bulkhead isolation: Critical compute resources partitioned into distinct pools preventing single component memory exhaustion.
- Self-healing orchestration: Health checks restart failing application containers automatically within seconds of detected crashes.
"""),
    ("prose_zero_trust_network_architecture.md", "explanatory_technical_prose", """# Zero Trust Security Architecture Fundamentals

## 1. Security Tenets
- Explicit verification: Every access request authenticated, authorized, and encrypted regardless of physical network origin.
- Principle of least privilege: Access rights restricted to the minimal resource scope required for immediate task execution.
- Continuous identity validation: Short-lived cryptographic tokens evaluated alongside device posture telemetry continuously.
- Micro-segmentation: Internal networks divided into granular secure zones preventing lateral threat actor movements.
- Comprehensive telemetry: Centralized log aggregation records every network connection and access event for forensic auditing.
"""),
    ("prose_observability_distributed_tracing.md", "explanatory_technical_prose", """# Observability and Distributed Tracing Paradigms

## 1. Telemetry Pillars
- Metrics collection: Time-series numeric counters, gauges, and histograms aggregate infrastructure operational health metrics.
- Structured logging: JSON formatted log events carry contextual trace identifiers enabling cross-service log correlation.
- Distributed trace propagation: W3C Trace Context headers carry trace and span identifiers across heterogeneous network boundaries.
- Contextual span trees: Hierarchical parent and child span records capture microservice execution latency and error events.
- Anomaly detection algorithms: Statistical deviation models identify latency regressions before catastrophic service failure.
"""),
    ("prose_immutable_infrastructure_devops.md", "explanatory_technical_prose", """# Immutable Infrastructure and Continuous Deployment

## 1. Deployment Paradigms
- Declarative infrastructure: Version-controlled configuration files specify desired cluster state without manual intervention.
- Immutable server images: Virtual machine base images compiled once and replaced entirely rather than updated in-place.
- Blue-green deployment: Parallel identical production environments enable instantaneous zero-downtime traffic switching.
- Automated rollback thresholds: Telemetry monitors trigger automated deployment reverts upon detected error rate spikes.
- Ephemeral test environments: Short-lived testing infrastructure instantiated on-demand and destroyed upon test suite completion.
"""),
    ("prose_data_pipeline_streaming_etl.md", "explanatory_technical_prose", """# Streaming Data Pipelines and Event-Driven Processing

## 1. Pipeline Dynamics
- Real-time ingestion: Distributed append-only message logs decouple data producers from downstream streaming consumers.
- Stream processing semantics: Exactly-once processing guarantees achieved through transactional log checkpoints and idempotent sinks.
- Windowing mechanisms: Tumbling, hopping, and sliding windows aggregate event streams across bounded temporal intervals.
- Schema evolution rules: Forward and backward compatible protocol buffer definitions prevent serialization decoding failures.
- Backpressure propagation: Reactive stream flows throttle upstream message producers during downstream processing slowdowns.
"""),
    ("prose_database_storage_engines.md", "explanatory_technical_prose", """# Database Storage Engines and Log Structured Merge Trees

## 1. Engine Architectures
- Write optimization: Log-Structured Merge (LSM) trees convert random disk writes into high-throughput sequential disk writes.
- In-memory MemTable: Recent writes buffered in memory-resident skip lists before flushing immutable SSTables to disk.
- Compaction strategies: Leveled and tiered compaction consolidate redundant key-value updates to reclaim storage space.
- Bloom filter acceleration: Probabilistic bit arrays determine whether requested keys exist within SSTables without disk reads.
- B-Tree read performance: Balanced multi-way search trees provide deterministic read latencies for point query access patterns.
""")
]

def main():
    manifest_entries = []
    for filename, category, content in DOCUMENTS:
        filepath = OUT_DIR / filename
        filepath.write_text(content.strip() + "\n", encoding="utf-8")
        h = hashlib.sha256(filepath.read_bytes()).hexdigest()
        manifest_entries.append({
            "filename": filename,
            "category": category,
            "sha256": h,
            "size_bytes": filepath.stat().st_size
        })
        print(f"Generated {filename} ({category}) - {filepath.stat().st_size} bytes - SHA: {h[:12]}")

    manifest = {
        "description": "Independent Multi-Domain Evaluation Corpus for Batch 1421-1460",
        "total_documents": len(manifest_entries),
        "categories": sorted(list(set(d["category"] for d in manifest_entries))),
        "documents": manifest_entries
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nManifest successfully written to {MANIFEST_PATH}")

if __name__ == "__main__":
    main()
