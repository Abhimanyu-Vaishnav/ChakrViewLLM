# MESI Cache Coherence Protocol and Snoop Filters

## 1. Coherence States
- Four discrete states: Modified, Exclusive, Shared, and Invalid states govern multicore cache line transactions.
- Snoop bus broadcast: Cache write invalidations propagate across bidirectional ring interconnects within 8 clock cycles.
- False sharing mitigation: Struct members padded to 64 bytes prevent concurrent invalidation of adjacent fields.
- Write-back buffer allocation: Dirty lines persist in non-volatile write caches before commit to memory controllers.
- Memory barrier primitives: Full fence instructions prevent out-of-order store reordering across speculative pipelines.
