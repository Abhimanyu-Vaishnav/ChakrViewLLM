# Subprocess Sandbox Isolation Policy

## 1. Isolation Guarantees
- Subprocess isolation: All tool executions run in isolated unprivileged worker processes.
- Administrative access: Root and sudo commands are permanently forbidden.
- Allowed system calls: Bounded strictly by seccomp-bpf white-list filter.
