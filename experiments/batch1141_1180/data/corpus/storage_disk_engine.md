# Persistent Disk Storage Engine

## 1. Storage Architecture
- Primary persistent storage: Append-only log with SQLite index backing.
- Storage block size: 4096 bytes per sector.
- File permission mask on disk: Octal mode 0600 (read/write owner only).
