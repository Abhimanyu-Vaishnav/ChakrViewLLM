# ChakrView Capability Registry
## Status Following CR-EXP-15 Architecture & Binding Investigation

**Registry Version:** 0.9.1-pre (Post CR-EXP-15)  
**Last Updated:** November 2026  
**Auditor:** Autonomous Scientific Auditor & Benchmark Engineer  

---

## Capability Status Summary

| Capability Domain | Sub-Capability | Benchmark / Probe Suite | Verified Performance | Status | Root Cause / Limiting Factor |
|---|---|---|---|---|---|
| **Conversational Flow** | Turn Management | Multi-turn conversational corpora | Normal | **PASS** | Stable dialogue formatting |
| **Conversational Flow** | Polite Salutations | Dialogue openers | Normal | **PASS** | High-frequency conversational priors |
| **Working Memory** | Literal Name Copying | Task A (150 probes) | **0.00%** (0/150) | **FAIL** | Absence of induction copy circuits |
| **Working Memory** | In-Context Entity Selection | Task B (150 probes) | **0.00%** (0/150) | **FAIL** | Attention sink on prompt headers |
| **Entity Binding** | Static Single Binding | Task C (150 probes) | **0.00%** (0/150) | **FAIL** | Parametric prior substitution |
| **Entity Binding** | Single-Entity State Update | Task D (150 probes) | **0.00%** (0/150) | **FAIL** | Lack of dynamic pointer routing |
| **Entity Binding** | Multi-Entity State Tracking | Task E (150 probes) | **0.00%** (0/150) | **FAIL** | Attention crosstalk between entities |
| **Entity Binding** | Interleaved State Tracking | Task F (150 probes) | **0.00%** (0/150) | **FAIL** | Failure to track temporal sequence |
| **Entity Binding** | Delayed Distractor Retrieval | Task G (150 probes) | **0.00%** (0/150) | **FAIL** | Distractor overwrite in residual stream |
| **Calibration** | Unanswerable Abstention | Task H (150 probes) | **0.00%** (0/150) | **FAIL** | Hallucinates memorized facts |

---

## Critical Registry Note on Historical Audits

- **CR-EXP-12 Claim:** Reported high conversational memory; refuted by CR-EXP-13 strict audit due to loose bag-of-words keyword scoring.
- **CR-EXP-14 Claim:** Reported 8.33% subject accuracy across all models; refuted by CR-EXP-15 evaluator audit which proved $8.33\% = 100 / 1200$ (hardcoded true for abstention probes). True non-abstention subject tracking was 0.00%.
- **CR-EXP-15 Truth:** All 7 evaluated models (across 4 curricula and 3 architecture scalings) score 0.00% on strict entity binding. The exact limiting mechanism has been identified as the lack of pretrained induction heads.
