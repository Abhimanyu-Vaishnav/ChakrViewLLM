# Raft Consensus Implementation

## 1. Consensus Parameters
- Leader election timeout: Bounded randomly between 150ms and 300ms.
- Heartbeat interval: Exactly 50 milliseconds.
- Quorum requirement: Strict majority formula floor(N/2) + 1 active nodes.
