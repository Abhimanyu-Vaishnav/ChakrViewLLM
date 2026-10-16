# Database Storage Engines and Log Structured Merge Trees

## 1. Engine Architectures
- Write optimization: Log-Structured Merge (LSM) trees convert random disk writes into high-throughput sequential disk writes.
- In-memory MemTable: Recent writes buffered in memory-resident skip lists before flushing immutable SSTables to disk.
- Compaction strategies: Leveled and tiered compaction consolidate redundant key-value updates to reclaim storage space.
- Bloom filter acceleration: Probabilistic bit arrays determine whether requested keys exist within SSTables without disk reads.
- B-Tree read performance: Balanced multi-way search trees provide deterministic read latencies for point query access patterns.
