# SWIM Gossip Protocol and Node Failure Detection

## 1. Protocol Dynamics
- Periodical ping probes: Random peer selection pings target nodes every 200 milliseconds to verify cluster presence.
- Indirect ping fallback: Unresponsive targets audited through 3 intermediary peers before declaring node suspicion.
- Suspicion mechanism: Suspicious nodes given 5 second grace periods to refute failure claims before global broadcast.
- Piggybacked metadata: Protocol messages carry membership state updates reducing dedicated network bandwidth consumption.
- Scale-free convergence: Failure notifications propagate across 1000 nodes within logarithmic round bounds.
