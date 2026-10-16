# Multi-Paxos Coordination and Phase Consensus

## 1. Paxos Phases
- Phase 1 Prepare requests: Proposers broadcast unique proposal numbers to acquire promise quorums from acceptors.
- Phase 2 Accept requests: Acceptors commit proposed values provided no higher proposal number was acknowledged.
- Leader lease guarantees: Monotonic epoch leases grant stable leaders master authority without per-query consensus.
- Fast Paxos optimization: Direct client broadcasts achieve consensus within two network round trips under zero contention.
- Catch-up streaming: Trailing nodes stream missing consensus entries from snapshot archives via dedicated TCP sessions.
