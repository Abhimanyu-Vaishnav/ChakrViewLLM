# Evaluation Protocol & Methodology (Batches 1581–1620)

## 1. Objectives & Scope
This evaluation assesses the next-generation capabilities implemented in Batches 1581–1620:
1. **Container Policy & Isolation (Track 1):** Validating strict policy enforcement, argument allowlisting, network disabled assertions, and runtime-unavailable fail-closed behavior.
2. **Resource Limits & Failure Recovery (Track 2):** Capping memory, CPU, process limits, timeout enforcement, output truncation, and temporary resource cleanup.
3. **Dependency-Aware Structured Verification (Track 3):** Stress-testing proposition extraction on multi-entity sentences, multi-parameter definitions, table structures, and conflicting evidence.
4. **Governed State & Multi-Turn Routing (Track 4):** Testing conversation state isolation across tenants and chained tool operations.
5. **End-to-End Governance & Tool Routing (Track 5):** Validating end-to-end integration across retrieval, verification, arithmetic, and containerized execution attempts.

Total evaluation dataset size: **300 held-out tasks**.

---

## 2. Evaluation Tracks & Denominators

| Track | Capability / Focus | Task ID Range | Denominator | Scoring Rule |
|:---|:---|:---:|:---:|:---|
| **Track 1** | Container Policy & Isolation | `cont_001` - `cont_060` | 60 | Status match (`RUNTIME_UNAVAILABLE`, `UNSAFE_CONFIG_REJECTED`, `SUCCESS`) |
| **Track 2** | Resource Limits & Recovery | `res_061` - `res_120` | 60 | Timeout / output truncation / cleanup assertion |
| **Track 3** | Dependency-Aware Verification | `verif_121` - `verif_220` | 100 | Exact match on 5-verdict state (`SUPPORTED`, `CONTRADICTED`, `INSUFFICIENT_EVIDENCE`, `CONFLICTING_EVIDENCE`, `INVALID_CLAIM`) |
| **Track 4** | Multi-Turn State & Routing | `state_221` - `state_260` | 40 | Exact state resolution & cross-tenant isolation |
| **Track 5** | End-to-End Governed Routing | `e2e_261` - `e2e_300` | 40 | Correct canonical route assignment & safe output emission |

---

## 3. Matched Baselines & Integrity Rules
- The evaluation suite is frozen and hashed to `experiments/batch1581_1620/data/evaluation_suite_300_v2.json`.
- The dataset manifest and SHA-256 are computed and sealed before running the benchmark.
- Raw per-task outcomes with millisecond latencies are saved to `experiments/batch1581_1620/results/BENCHMARK_RESULTS_300_V2.json`.
- Security tests explicitly separate:
  - Rejecting unsafe / disallowed configuration.
  - Successfully executing an authorized task.
  - Identifying that an isolation runtime is unavailable.
