# Storage Compaction and Vacuum SOP

## 1. Compaction Triggers
- Fragmentation threshold: 25% dead tuple volume triggers auto-vacuum.
- Scheduled compaction time: Nightly at 02:00 UTC.
- Temporary disk space headroom required: 50% free volume capacity.
