# CR-EXP-12 DIAGNOSTIC STUDIES & FINDINGS
**Mission:** CR-EXP-12 Phases 3–8  
**Auditor:** Autonomous Cognitive Benchmark Engineer & Neural Research Scientist  
**Date:** 2026-11-07T22:33:00+05:30  

---

## 1. CRITICAL DIAGNOSTIC DISCOVERY: THE CONSTRAINT ILLUSION
In CR-EXP-11, Candidate A reportedly achieved **100.0% (50/50)** on constraint satisfaction.
When tested on a balanced 150-probe suite expanding beyond fruits/colors to animals, vehicles, and furniture:
- **Candidate A scored only 10.0% (15/150)**!
- **Candidate B scored 0.0% (0/150)**!

### Root Cause
Candidate A only memorized a single specific vocabulary pairing (`"fruit only and strictly ignore other items"` with apple/banana/orange/mango/grape). When asked to `"Pick the animal only"`, `"Pick the vehicle only"`, or `"Pick the furniture only"`, the attention heads failed completely to attend to the constraint class.
**Finding**: The reported 100% was an artifact of narrow template vocabulary in CR-11, **NOT** generalizable semantic constraint satisfaction.

---

## 2. MULTI-TURN WORKING MEMORY & SUBJECT SUBSTITUTION
In multi-turn tracking dialogues (`"User: Track this: Pooja is in Jaipur... Quick change: Pooja moved to Bhopal... Where is Pooja now?"`):
- Candidate A generated: `"Aarav is currently in Lucknow."` (Subject substitution: `Pooja` replaced with `Aarav`, city replaced with training default).
- Candidate B generated: `"Kochi is currently in Bhopal."` (Subject omitted, updated city retained).
- **Core Diagnosis**: Both models suffer from **Attention Decoupling between Subject and Predicate**. In multi-turn context, the query head for location successfully finds the latest city token, but the subject query head defaults to a high-frequency training name attractor (`Aarav` or `Kabir`).

---

## 3. ARITHMETIC REGRESSION DIAGNOSIS
On integer word problems (`"Swati had 10 pens, bought 5 more"`):
- Candidate A generated: `"200."` and `"2\nSneI am your context does not state..."`.
- The arithmetic head was completely unanchored during CR-11 training, resulting in severe output collapse and bleeding from the abstention heuristic.
- **Intervention Mandate**: A dedicated synthetic state-arithmetic stream must be reintroduced into CR-12 training.

