# Threat Modeling & Attack Surface

## 1. Passive Document Containment
- Untrusted content policy: All ingested documents are treated as untrusted passive data.
- Prompt-injection resistance: Instructions inside retrieved documents never override system execution policies.
- Tool invocation barrier: Document Q&A cannot trigger filesystem modifications or network egress.
