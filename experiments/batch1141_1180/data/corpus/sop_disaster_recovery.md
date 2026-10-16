# Disaster Recovery & Failover Procedure

## 1. Recovery Time and Point Objectives
- Recovery Time Objective (RTO): Under 15 minutes to restore secondary node cluster.
- Recovery Point Objective (RPO): Under 5 minutes of transactional data loss.
- Failover trigger: Automatic DNS redirection after 3 consecutive missed heartbeat intervals.
