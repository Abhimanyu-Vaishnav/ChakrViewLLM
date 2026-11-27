# CHAKRVIEW DECISION LOG

## Master Architecture & Experimentation Decisions

### Decision 001: Separation of Neural Operation Routing vs Symbolic Execution
- **Date:** 2026-11-26
- **Context:** NR-018 through NR-020 demonstrated that direct autoregressive generation of execution results by a 17M parameter model produces high hallucination and low syntax reliability (~1.4% to 1.9%).
- **Decision:** Establish a strict neuro-symbolic boundary: The neural model functions as a semantic intent parser and operation classifier, while an allowlisted, restricted Python execution sandbox deterministically computes mathematical and structured transformations. Deterministic execution results will NEVER be reported as pure neural reasoning capability.

### Decision 002: Matched-Distribution 1,000-Task Benchmark as Definitive Ground Truth
- **Date:** 2026-11-26
- **Context:** An apparent anomaly in NR-019 showed OOD accuracy (57.8%) higher than In-Distribution accuracy (38.33%). An audit revealed task family distribution skew (OOD over-represented easy operations).
- **Decision:** Replace the 800-task skewed set with the strictly balanced 1,000-task matched benchmark (`fresh_matched_benchmark_1000.json`), enforcing exactly 24 tasks per operation family in both ID and OOD partitions across 20 distinct operations plus adversarial cases.

### Decision 003: Autonomous Phased Execution of Master Program
- **Date:** 2026-11-26
- **Context:** The user authorized autonomous progression through all phases of ChakrView 1.0 Release Candidate development without requiring turn-by-turn prompts.
- **Decision:** Progress systematically across Phases 0 through 24, persisting state in `CHAKRVIEW_CURRENT_STATE.md` and `CHAKRVIEW_MISSION_LEDGER.md` after every milestone.
