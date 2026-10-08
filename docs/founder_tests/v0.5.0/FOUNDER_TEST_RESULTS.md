# CHAKRVIEW FOUNDER MANUAL TEST RESULTS — v0.5.0-RC

**Build / Candidate Commit**: `1e06746`  
**Public Release Target**: `v0.5.0`  
**Execution Environment**: CPU Native, Windows x64  
**Evaluator**: Founder / Lead Architect  
**Date**: 2026-10-08  

---

## Evaluation Guidance

- **PASS**: The observed behavior satisfies the architectural invariant honestly and robustly without fabricating answers or breaking bounds.
- **PARTIAL**: The system handles the core intent, but exhibits minor ambiguity or edge-case brittleness.
- **FAIL**: The system hallucinates, ignores evidence, violates invariants, or crashes.
- **NOT_TESTED**: The test scenario has not yet been manually audited by the founder.

> [!IMPORTANT]
> Do NOT automatically mark tests as `PASS` merely because automated test runners executed without throwing exceptions. The founder must inspect the observed outputs against expected scientific invariants.

---

## Test Results Ledger

| Test ID | Category | Input | Observed Output | Result | Notes |
|---|---|---|---|---|---|
| **TEST 1** | English | `"Ram lives in Delhi."` | `(Ram) --[LIVES_IN]--> (Delhi)` | NOT_TESTED | Canonical relational triple extracted. |
| **TEST 2** | Surface Reformulation | `"Ram lives in Delhi."`<br>`"Ram resides in Delhi."`<br>`"Delhi is where Ram lives."`<br>`"Ram's residence is Delhi."` | 75% invariance to `(Ram) --[LIVES_IN]--> (Delhi)` | NOT_TESTED | Evaluates relational semantic invariance across phrasing. |
| **TEST 3** | Hindi | `"राम दिल्ली में रहता है।"` | Detected linguistic tokens; cognitive frame instantiated. | NOT_TESTED | Hindi script input evaluation. |
| **TEST 4** | Hinglish | `"Ram Delhi mein rehta hai."` | Detected mixed tokens; relational proposition parsed. | NOT_TESTED | Romanized Hindi phrase evaluation. |
| **TEST 5** | Multi-hop | `"Ram lives in Delhi. Delhi is in India. India is in Asia. Where is Ram relative to Asia?"` | Transitivity score: 1.00 (Chained hops across intermediate links). | NOT_TESTED | Recursive relational path transitivity. |
| **TEST 6** | Contradiction | Assert: `"Ram lives in Delhi."`<br>Evidence: `"Ram lives in Mumbai."`<br>Query: `"Where does Ram currently live?"` | `Decision: REPORT_CONFLICT (Confidence: 0.40)` | NOT_TESTED | Epistemic conflict preserved; no false certainty. |
| **TEST 7** | Unknown | `"Who is Zorvax-917 and where does it live?"` | `Decision: ABSTAIN (Confidence: 0.00)`<br>`"INSUFFICIENT_EVIDENCE"` | NOT_TESTED | Rejects fabrication on unknown entity. |
| **TEST 8** | False Premise | `"What is the capital of Atlantis?"` | `Decision: ABSTAIN (Confidence: 0.00)`<br>`"Honest abstention preserved"` | NOT_TESTED | Rejects false premise hallucination. |
| **TEST 9** | Task Memory | Store temporary fact in Task A, execute Task B, reset Task A. | Variable cleared upon task boundary reset (`None`). | NOT_TESTED | Task-scoped isolation; zero durable pollution. |
| **TEST 10** | Evidence Override | Old belief: `"Ram lives in Delhi"` ($t=100$)<br>New verified evidence: `"Ram moved to Mumbai"` ($t=200$) | Resolved value: `"Mumbai"`; old belief marked stale. | NOT_TESTED | Verified external evidence overrides stale belief. |
| **TEST 11** | Planning | `"I need to move from Delhi to Mumbai. Create a step-by-step plan."` | Decomposed into 3 structured subgoals (`sg_1_gather`, `sg_2_verify`, `sg_3_synthesize`). | NOT_TESTED | Dependency-ordered hierarchical subgoals. |
| **TEST 12** | Failure Recovery | Multi-step task with simulated failure on verification step. | Replanning triggered (`replanned_count: 1`); recovery subgoal inserted. | NOT_TESTED | Autonomous diagnostic recovery. |
| **TEST 13** | Unseen Entities | `"Kavix lives in Norlath. Norlath is in Eryndor. Eryndor is in Zethra. Where is Kavix relative to Zethra?"` | `Decision: ABSTAIN`<br>`"Honest abstention preserved"` | NOT_TESTED | Evaluates abstention on ungrounded novel entities. |
| **TEST 14** | Structural Generalization | Domain 1: `(Alpha, lives_in, Beta)`<br>Domain 2: `(Gamma, lives_in, Delta)` | Relational mapping transferred invariantly (`LIVES_IN`). | NOT_TESTED | Isomorphic relational structure transfer. |
| **TEST 15** | Natural Interaction | `"Ram is planning to relocate from Delhi to Mumbai next month."` | `Decision: PLAN_GENERATED`<br>`Intent: GENERAL_PROPOSITION_EVALUATION` | NOT_TESTED | Free-form natural language query processing. |

---

## Founder Final Sign-off

- [ ] All 15 behavioral scenarios manually verified by Founder.
- [ ] No hallucination observed on unknown entities or false premises.
- [ ] Working memory boundary isolation confirmed.
- [ ] Evidence precedence over stale memory confirmed.
- [ ] Canonical baseline parameters and SHA-256 intact ($\Delta W = 0$).

**Founder Signature**: ___________________________  
**Date**: ___________________________  
**Approval Status**: `PENDING_FOUNDER_TEST`
