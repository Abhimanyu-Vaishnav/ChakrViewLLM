# Two-Phase Commit Distributed Transaction Engine

## 1. Transaction Protocol
- Phase 1 Prepare voting: Transaction coordinator requests commit votes from all participating database resource managers.
- Phase 2 Commit execution: Coordinator broadcasts global commit decisions only if all participants return affirmative votes.
- Write-ahead logging: Coordinators and participants log state transitions to non-volatile disk before message dispatch.
- Coordinator failure recovery: Presumed abort protocols abort in-flight transactions when coordinator status is ambiguous.
- Timeout resolution: Participant locks remain held during coordinator partitions awaiting resolution protocols.
