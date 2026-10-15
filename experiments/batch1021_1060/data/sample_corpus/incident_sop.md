# Incident Management Standard Operating Procedure

## 1. Incident Severity Definitions
- Severity 1 (Critical): Total outage of local inference or unrecoverable session state corruption.
- Severity 2 (Major): Retrieval latency degradation exceeding 500ms or partial tool sandbox failures.
- Severity 3 (Minor): Non-blocking UI glitch or metric logging omission.

## 2. Escalation Paths
- For Severity 1 events, the primary on-call engineer must be paged immediately via PagerDuty.
- If no acknowledgment is received within 5 minutes, secondary escalation pages the engineering director.
- Post-incident post-mortems must be published within 48 hours of resolution.

## 3. Disaster Recovery and Snapshots
- System state snapshots are taken every 6 hours and stored in an offsite immutable object store.
- Recovery Time Objective (RTO) is 30 minutes.
- Recovery Point Objective (RPO) is 15 minutes.
