# End-to-End System Evaluation Results

## Overview & Scope
This report summarizes the end-to-end performance of the Governed Tool Router (`chakrview.runtime.governed_tool_router.GovernedToolRouter`) integrated with multi-turn conversation state, safe arithmetic execution, structured proposition verification, sandboxed code execution, and knowledge retrieval.

Total tasks evaluated across all tracks: **300**
Overall tasks passed: **292**
Overall success rate: **97.33%** (292/300)

## Comprehensive Results by Track

| Track | Capability / Focus | Denominator | Passed | Accuracy | Status |
|:---|:---|:---:|:---:|:---:|:---:|
| **Track 1** | Multi-Turn Reference Resolution | 80 | 80 | 100.0% | Pass |
| **Track 2** | Arithmetic Tool Chaining | 60 | 60 | 100.0% | Pass |
| **Track 3** | Sandbox Execution & Isolation | 60 | 60 | 100.0% | Pass |
| **Track 4** | Structured Proposition Verification | 60 | 53 | 88.33% | Pass |
| **Track 5** | Routing & Failure Recovery | 40 | 39 | 97.5% | Pass |
| **Total** | **All Tracks Combined** | **300** | **292** | **97.33%** | **Acceptance Met** |

## Track 5 Routing & Failure Recovery Breakdown
- **Tasks Evaluated:** 40 (`route_261` to `route_300`)
- **Passed:** 39 / 40 (97.5%)
- **Distribution:**
  - Calculator / Arithmetic routing: 10 / 10 passed (100.0%)
  - Multi-turn reference chaining routing: 10 / 10 passed (100.0%)
  - Code execution routing & fail-closed interception: 10 / 10 passed (100.0%)
  - Structured proposition verification routing: 9 / 10 passed (90.0%)
    - 1 miss (`route_279`): `"Check if mesh network data port is 80"` was routed to `KNOWLEDGE_RETRIEVAL` rather than `STRUCTURED_VERIFICATION` due to lack of an explicit verification marker keyword (`"verify"`, `"contradicts"`, `"claim"`).

## System Latency Profile
- Arithmetic tool invocation: **0.08 ms**
- Anaphoric reference resolution: **0.02 ms**
- Sandbox AST inspection: **0.12 ms**
- Proposition verification (with retrieval): **1.45 ms**
- Overall router overhead: **< 2.5 ms** per transaction.

## Attribution of Intelligence
All correct outputs in this benchmark are strictly attributed to:
1. Deterministic Python AST grammar parsing and evaluation.
2. Exact state tracking and provenance mapping in memory.
3. Subprocess process isolation and system API boundaries.
4. Extractive BM25 retrieval and entity-anchored regular expression extraction.
No claim of learned mathematical reasoning or emergent intelligence is made.
