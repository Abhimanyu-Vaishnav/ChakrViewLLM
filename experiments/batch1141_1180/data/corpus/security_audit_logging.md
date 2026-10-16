# Security Audit Trail and Event Logging

## 1. Audit Requirements
- Event log destination: Append-only signed audit log at `var/logs/audit.log`.
- Log retention minimum: 365 calendar days.
- Monitored events: Authentication attempts, privilege elevations, document ingestion, and index pruning.
