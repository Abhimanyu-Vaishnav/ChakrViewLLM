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

---

## 10-Point Founder Manual Acceptance Checklist

This checklist provides a reproducible, auditable framework for the founder to evaluate whether ChakrView qualifies as an **Experimental Research Preview**.

| # | Inspection Item | Scope & Evaluation Method | Result (`PASS` / `FAIL` / `NOT_TESTED`) | Founder Observation |
|---|---|---|---|---|
| **1** | **Startup & Model / Tokenizer Identity** | Run `scripts/verify_v050_release.py`. Verify parameter count ($3,443,136$), canonical SHA-256 (`c5571c...00a282da`), and $\Delta W = 0$. | NOT_TESTED | [To be recorded by Founder] |
| **2** | **Neural Generation vs. Governed Cognitive Responses** | Inspect responses to structured relational queries vs open prompts. Verify system uses deterministic cognitive realizer for verified facts and does not conflate neural text generation with factual grounding. | NOT_TESTED | [To be recorded by Founder] |
| **3** | **Empty & Malformed Input Handling** | Test empty string `""`, whitespace `"   "`, punctuation only `"?!"`, and mixed transliterations. Verify system does not crash and returns controlled fallback. | NOT_TESTED | [To be recorded by Founder] |
| **4** | **Conflicting Evidence & Honest Abstention** | Provide contradictory assertions (e.g. Test 6: Ram in Delhi vs. Mumbai). Verify system returns `REPORT_CONFLICT` or `ABSTAIN` instead of arbitrary hallucination. | NOT_TESTED | [To be recorded by Founder] |
| **5** | **Working Memory & Factual Consistency** | Assert task-scoped fact, query within task, reset task, query again. Confirm timestamped evidence precedence and zero durable memory leakage. | NOT_TESTED | [To be recorded by Founder] |
| **6** | **Task Failure & Honest Error Reporting** | Trigger step failure in multi-step task execution (e.g. Test 12). Verify honest `FAILED` reporting, bounded retries, and dynamic replanning without fabricated success. | NOT_TESTED | [To be recorded by Founder] |
| **7** | **Durable Restart Recovery** | Simulate process interruption during chunked task or journaled execution. Verify atomic journal reload, SHA-256 integrity check, and clean resumption from checkpoint. | NOT_TESTED | [To be recorded by Founder] |
| **8** | **CPU & Memory Behavior** | Monitor resource footprint under load. Verify host resource discovery, strict memory upper bounds, and absence of memory leaks during multi-hop reasoning. | NOT_TESTED | [To be recorded by Founder] |
| **9** | **Known Neural Reasoning Limitations** | Evaluate ungrounded open queries on 3.44M ChakrMicro. Verify that limitations (repetition, small vocab $V=4096$, lack of general reasoning) are honestly acknowledged in output/manifest. | NOT_TESTED | [To be recorded by Founder] |
| **10** | **Suitability for Experimental Research Preview** | Synthesize all criteria: evaluate whether v0.5.0 meets the documented scope of an Experimental Research Preview without marketing overclaims. | NOT_TESTED | [To be recorded by Founder] |

---

## Founder Final Sign-off

- [ ] All 10 checklist items inspected and recorded by Founder.
- [ ] Behavioral scenarios audited against experimental preview standards.
- [ ] No fabricated results or automated substitutions for Founder judgment.
- [ ] Canonical baseline parameters and SHA-256 intact ($\Delta W = 0$).
- [ ] Official QA review queue (30 records) remains preserved.

**Founder Signature**: ___________________________  
**Date**: ___________________________  
**Approval Status**: `PENDING_FOUNDER_TEST`

