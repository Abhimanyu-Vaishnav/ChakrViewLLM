# Dataset Provenance & Integrity (Batches 1581–1620)

## 1. Frozen Evaluation Suite
- **Dataset Path:** `experiments/batch1581_1620/data/evaluation_suite_300_v2.json`
- **Manifest Path:** `experiments/batch1581_1620/data/EVALUATION_SUITE_MANIFEST_V2.json`
- **Total Tasks:** **300**
- **Dataset SHA-256:** `06cd680241a142e356885f2cfcf269b6764053c210ff09c5d64943c6fac2190c`

---

## 2. Track Distribution & Category Provenance

| Track | Category | Tasks | Task ID Range | Provenance Description |
|:---|:---|:---:|:---:|:---|
| **Track 1** | Runtime Unavailable Discovery | 20 | `cont_001` - `cont_020` | Verifies fail-closed behavior when OCI engine is absent |
| **Track 1** | Unapproved Image Rejection | 20 | `cont_021` - `cont_040` | Tests image allowlist enforcement |
| **Track 1** | Unsafe Mount Path Rejection | 10 | `cont_041` - `cont_050` | Rejects host tampering attempts (`/`, `/etc`, `C:\Windows`) |
| **Track 1** | Network Access Rejection | 10 | `cont_051` - `cont_060` | Enforces `--network none` security constraint |
| **Track 2** | Memory Limit Rejection | 20 | `res_061` - `res_080` | Validates format and ceiling limits for memory |
| **Track 2** | Timeout Enforcement | 20 | `res_081` - `res_100` | Subprocess wall-clock timeout killing loops |
| **Track 2** | Output Buffer Truncation | 20 | `res_101` - `res_120` | Output clamping ceiling at 64 KB |
| **Track 3** | Dependency-Aware Verification | 100 | `verif_121` - `verif_220` | Multi-entity, multi-parameter, version, and duration claims |
| **Track 4** | Multi-Turn Anaphora Chaining | 20 | `state_221` - `state_240` | Chaining numeric operations through `"that"` |
| **Track 4** | Cross-Tenant Isolation | 10 | `state_241` - `state_250` | Zero state bleed across separate tenant IDs |
| **Track 4** | Post-Reset Purge Fail-Closed | 10 | `state_251` - `state_260` | Purges memory after conversation reset |
| **Track 5** | End-to-End Governed Routing | 40 | `e2e_261` - `e2e_300` | Routing across calculator, verification, and container paths |

---

## 3. Data Integrity & Contamination Controls
- No evaluation tasks are used as training data (zero model training performed).
- Distinct task IDs and synthetic variations ensure strictly held-out evaluation.
- Manifest sealed before benchmark execution.
