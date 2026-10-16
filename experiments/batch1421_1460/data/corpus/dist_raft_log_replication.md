# Raft Consensus Protocol Log Replication Invariants

## 1. Protocol Specifications
- Leader election timeout: Randomized election timers span 150 to 300 milliseconds preventing split-vote deadlocks.
- AppendEntries RPC protocol: Heartbeat messages transmit preceding log index and term to enforce log consistency.
- Quorum commit requirement: Entries committed once persisted on a strict majority of active cluster voter nodes.
- Log matching property: Identical index and term entries guarantee identical preceding log sequences across all nodes.
- State machine application: Committed log entries apply deterministically to local state machines in index sequence.
