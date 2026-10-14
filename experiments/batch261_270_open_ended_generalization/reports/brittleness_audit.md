# Brittleness Audit & Hardcoded Shortcut Analysis (Batch 264)
**Location:** `experiments/batch261_270_open_ended_generalization/reports/brittleness_audit.md`  
**Milestone:** Batches 261–270 — Open-Ended Question Generalization  
**Date:** 2026-10-14  

---

## 1. Executive Summary: The Vulnerability of Exact-Match Shortcuts

An empirical audit of the router, adapter, and real console paths against the newly generated 120-item sealed held-out question set revealed critical brittleness in ChakrView's runtime routing:

| Architectural Component | Brittle Mechanism Identified | Impact on Unseen Generalization | Generalization Score |
|---|---|---|---|
| **Arithmetic Parser** | Fixed regex requiring `"What is X + Y"` or `fullmatch([\\d\\s\\+\\-\\*/...])` | Failed on verbal word queries (`"Subtract 36 from 78"`, `"Compute sum of 62 and 14"`) | **50.0%** (7/14 passed) |
| **Concept Knowledge Base** | Small in-memory dictionary mapping 7 hardcoded keys (`pi`, `e`, `alu`, `cpu`, `ram`, `mars`, `photosynthesis`) | Failed on all unseen scientific/technical entities (DNA, Mitochondria, GPU, Transistor, Quicksort) | **15.4%** (2/13 passed) |
| **False Premise Detection** | Regex looking specifically for `capital of [planet]` | Failed on political/governance false premises (`"Name the prime minister of Mars"`) | **65.0%** (13/20 passed) |
| **Unanswerable Open-World** | Fallback triggered only if question ends with `?` or specific keywords (`predict`, `lottery`, `secret`) | Failed on assertion-style unanswerable queries (`"Provide the exact verifiable formula for..."`) | **69.2%** (9/13 passed) |
| **Coreference Follow-up** | Regex matching only `"and what about [concept]"` against hardcoded concepts | Failed on general conversational follow-ups (`"What does it mean for [X]?"`) | **0.0%** (0/11 passed) |
| **Neural Core Unassisted** | Canonical 3.44M model without pointer head or retrieval tools | Incapable of factual recall on open-ended prose questions | **3.3%** (4/120 passed) |

---

## 2. In-Depth Root Cause Diagnoses

### 2.1 Arithmetic Verbal Formulation Failure
In `chakrview/runtime/cognitive_router.py`:
```python
prefix_pattern = r"^(?:what\s+is\s+(?:the\s+(?:result|value|sum|product)\s+of\s+)?|calculate\s+|compute\s+|evaluate\s+|solve\s+)"
cleaned = re.sub(prefix_pattern, "", q_strip, flags=re.IGNORECASE).strip()
if re.fullmatch(r"[\d\s\+\-\*\/\(\)\.\%]+", cleaned): ...
```
- When the user asks: *"Subtract 36 from 78"*, `re.sub` does not strip `"Subtract ... from"`. `cleaned` contains alphabetic characters (`"subtract ..."`), which fails `re.fullmatch`. The query falls through to unstructured fallback: `"Acknowledged input: 'Subtract 36 from 78'."`
- **Root Cause:** Inflexible lexical regex rather than syntactic operator/operand extraction.

### 2.2 Concept Dictionary vs. General Semantic Knowledge
In `chakrview/runtime/session_adapter.py`:
```python
knowledge_map = {
    "pi": "The value of pi...",
    "alu": "An ALU (Arithmetic Logic Unit)...",
    "cpu": "A CPU (Central Processing Unit)...",
}
```
- Any technical term outside this 7-item set cannot be answered by the system knowledge adapter.
- The neural core (3.44M parameters) was pretrained on synthetic relational corpora and cannot retrieve open-world factual definitions without external evidence or document context.
- **Root Cause:** Conflating a small built-in concept dictionary with a general question-answering engine.

### 2.3 Celestial False Premise Narrowness
In `chakrview/runtime/cognitive_router.py`:
- Recognizes: `"capital of Mars"`, `"capital of the Sun"`.
- Does NOT recognize: `"Name the prime minister of Mars"`, `"Who is the king of Jupiter?"`.
- Consequently, it returns a generic acknowledgement rather than explaining that Mars/Jupiter are uninhabited astronomical bodies without human governments.

---

## 3. Engineering Remedy Plan (Batch 266)

Rather than adding endless special-case keywords, we will address shared root causes:
1. **Verbal Arithmetic Normalization:** Implement semantic operator parsing in `CognitiveRouter._extract_math_expression()` to handle verbal relations (`"sum of X and Y"`, `"subtract Y from X"`, `"multiply X by Y"`).
2. **Generalized False Premise Detector:** Detect fictional leadership, human municipal attributes, or governance queries applied to known non-human / astronomical entities.
3. **Epistemic Abstention on Open Directives:** Expand open-world abstention to handle declarative commands requesting unverifiable secrets or formulas (`"Provide the exact verifiable formula for..."`).
4. **General Discourse Follow-Up Resolution:** Connect `coreference_followup` to the inspectable `DiscourseEntity` stack in `ConversationalContextTracker` rather than a fixed concept whitelist.
