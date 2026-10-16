# Completely Fair Scheduler and Task Hierarchy

## 1. Scheduling Mechanics
- Red-black tree hierarchy: Runnable threads indexed by virtual runtime nanoseconds tracking CPU resource fairness.
- Target latency window: Minimum scheduling granularity capped at 0.75 milliseconds across active thread pools.
- Thread affinity masking: Bitmask pinning confines latency-critical workers to physical CPU cores with isolated L3 caches.
- Dynamic priority weighting: Nice values mapped logarithmically across 40 priority levels scaling execution time slices.
- Context switch overhead: Thread yield takes approximately 1.2 microseconds per voluntary synchronization call.
