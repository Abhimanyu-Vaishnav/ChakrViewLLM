# Evaluation Protocol: Batches 1541–1580

## Multi-Turn Tool Chaining, Sandbox Security & Proposition Verification

### 1. Protocol Objective

To objectively test the multi-turn state management, arithmetic tool chaining, sandboxed execution safety, and structured proposition verification, this protocol establishes frozen test suites, pre-declared pass/fail gates, and explicit scoring functions across five evaluation tracks totaling **300 held-out tasks**.

---

### 2. Pre-Declared Pass/Fail Gates

| Track | Tested Dimension | Minimum Pass Gate | Target Metric |
|---|---|---|---|
| **Track 1** | Multi-Turn Reference Resolution | $\ge 95.0\%$ | Resolution of valid antecedents, clean refusal on missing/reset state, zero cross-tenant leakage. |
| **Track 2** | Arithmetic Tool Chaining | $\ge 95.0\%$ | 3-step arithmetic calculation correctness, graceful division-by-zero trapping. |
| **Track 3** | Sandbox Execution & Isolation | $\ge 95.0\%$ | 100% rejection of forbidden imports, 100% timeout enforcement, buffer clamping. |
| **Track 4** | Structured Proposition Verification | $\ge 90.0\%$ | Zero attribute cross-talk in multi-entity pricing and adjacent parameter lines. |
| **Track 5** | Routing & Recovery | $\ge 95.0\%$ | Canonical route selection and error recovery. |
| **Overall** | Unified Architecture Benchmark | $\ge 95.0\%$ | Macro-average reliability across all 300 tasks. |
