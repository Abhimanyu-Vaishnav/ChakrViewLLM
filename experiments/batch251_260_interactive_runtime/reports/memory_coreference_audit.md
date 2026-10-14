# Batch 255: Multi-Turn Memory & Discourse Coreference Audit

**Milestone**: Batches 251–260  
**Date**: 2026-10-14  

---

## 1. Discourse Entity Stack & Epistemic Boundaries

In Batches 251–255, we elevated the conversational context mechanism from simple isolated user-name memory to an inspectable, bounded **Discourse Entity Stack**:

1. **Discourse Entity Record (`DiscourseEntity`)**:
   - `entity`: Normalized string identifier (e.g. `Earth`, `Ganymede`, `ALU`).
   - `entity_type`: Category classification (`person`, `place`, `concept`, `object`).
   - `turn_index`: Exact turn index in the active session history where the entity was activated or referenced.
   - `provenance`: Provenance tag (`direct_input`, `turn_subject`, `evidence_grounded`).
   - `timestamp`: Monotonic epoch timestamp.

2. **Stack Dynamics**:
   - Bounded to the most recent 10 activated entities.
   - Re-mentioning an existing entity moves it to the top of the stack (activation refresh).
   - Epistemic Isolation: Pronouns (`he`, `she`, `it`, `vo`, `woh`) resolve strictly to the most recent active entity on the stack.
   - Stale Entity Guard: If user input mentions an explicit new subject, subsequent queries do not erroneously bind to stale previous subjects.

---

## 2. Multi-Turn Coreference Evaluation (5-Turn Benchmark)

We tested a 5-turn session with topic switches and pronoun resolutions:

| Turn | User Query | Expected Resolution / Action | Runtime Observed Resolution | Status |
|---|---|---|---|---|
| Turn 1 | `Ram lives in Delhi.` | Store belief: `(Ram) --[LIVES_IN]--> (Delhi)`. Push `Ram` and `Delhi` to stack. | Stored belief; active subject = `Ram`. | **PASS** |
| Turn 2 | `Where does he live?` | Pronoun `he` resolves to top discourse entity `Ram`. Query answers `Delhi`. | Resolved `he` $\to$ `Ram`; answered `Delhi`. | **PASS** |
| Turn 3 | `What is an ALU?` | Topic shift to computer architecture. `ALU` pushed to stack. | Core concept route invoked; answered verified definition. | **PASS** |
| Turn 4 | `What is 25 + 37?` | Intervening arithmetic computation. | Calculator tool evaluated expression $= 62$. | **PASS** |
| Turn 5 | `Where does Ram live?` | Return to person query. | Query recalls `Delhi` without corruption from `ALU` or math turns. | **PASS** |

---

## 3. Session Isolation & Process Restart Verification

- **Session Isolation**: Tested two distinct session IDs (`session_user_A` vs `session_user_B`). Facts stored in `session_user_A` are invisible to `session_user_B`.
- **Restart Persistence**: Re-instantiated memory objects from disk files (`session_<id>.json`). Facts persisted with atomic flush and were verified identical.
