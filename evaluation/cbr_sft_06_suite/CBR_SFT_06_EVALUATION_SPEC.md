# CBR-SFT-06 Evaluation Framework Specification

**Date:** 2026-11-04  
**Mission:** CBR-SFT-06 Evaluation Engineering  

---

## 1. Dual Evaluation Architecture

To ensure both strict exact copy evaluation and robust conversational flexibility, the CBR-SFT-06 framework establishes two distinct evaluation tiers:

### Tier 1: Exact Character Retrieval (Strict Mechanical)
- Evaluates: Technical entities, single-token copy, port numbers, IP addresses, variables.
- Scoring Rules:
  - **Exact Match:** Strict character-by-character match after boundary whitespace strip (`clean(out) == clean(target)`).
  - **Exact Prefix Match:** Output starts with exact target followed by delimiter/EOS (`out.startswith(target) and out[len(target)] in " \n.,:"`).
  - **False Abstention:** Model emitted abstention when value was present in context (Strict Penalty: 0.0 score).
  - **Hallucination:** Model emitted an entity not present in context or incorrect value (Strict Penalty: 0.0 score).

### Tier 2: Conversational & Epistemic Evaluation
- Evaluates: Time-conditioned greetings, identity, reasoning, abstentions.
- Scoring Rules:
  - **Time Match:** Verifies greeting corresponds to timestamp (e.g. `[08:00:00]` requires morning tokens).
  - **Identity Match:** Must state ChakrMicro / ChakrView / indigenous lineage without external model claims.
  - **Abstention Classification:** Correctly identifies when information is unstated using canonical semantic regex.

---

## 2. Benchmark Suite Harmonization

All future evaluations will run deterministically across:
1. `CBR-SFT-06 Held-Out Suite` (100 tasks: 35% conversational replay, 35% single-token copy/KV, 15% multi-turn, 15% abstention).
2. `CBR-SFT-04 Held-Out Suite` (80 tasks: Regression anchor).
3. `CBR-SFT-03B Greeting/Identity Suite` (70 tasks).
4. `Canonical M3 Conversational Suite` (60 tasks).
5. `Native Intelligence Suite` (32 tasks).
6. `Coding Benchmark` (12 tasks).
