# Batch 243: Intent-Routing Root-Cause Analysis

## 1. Problem Statement
In interactive evaluation, 10 fundamental query archetypes failed despite specialized components existing in the codebase:
1. `What is 25 + 37?`
2. `What is the value of pi?`
3. `What is an ALU?`
4. `Explain CPU in simple words.`
5. `Write a Python program to add two numbers entered by the user.`
6. `What is the capital of Mars?`
7. `Mujhe simple Hindi mein photosynthesis samjhao.`
8. `and What about ALU`
9. `My name is Ramesh`
10. `What is my name?`

---

## 2. Root Cause Breakdown by Subsystem

### A. Arithmetic Routed to Knowledge Retrieval
- **Mechanism**: `CognitiveRouter._is_knowledge_seeking()` matches `what is`. Because `what is 25 + 37?` contains `what is`, it was classified as `KNOWLEDGE_RETRIEVAL`.
- **Root Cause**: The router had no arithmetic expression detection before the general `what is` pattern.
- **Remedy**: Introduce `RouteType.CALCULATOR_TOOL` checked *prior* to knowledge retrieval when queries contain numeric operands and operators (`+`, `-`, `*`, `/`, `plus`, `minus`, `times`, `divided by`).

### B. Definition & Acronym Mis-Extraction
- **Mechanism**: `What is an ALU?` or `What is the value of pi?` was parsed by `EvidenceExtractor` using regex `What is ([A-Za-z0-9_-]+)\?` or `What is the ([A-Za-z0-9_-]+) of ([A-Za-z0-9_-]+)\?`.
  - In `What is an ALU?`, the pattern captured `an` instead of `ALU` because `an` is a 2-letter word preceding `ALU`.
  - In `What is the value of pi?`, `value` was extracted as the property and `pi` as the entity, but default retrieval indexes had no formula document for `pi`.
- **Remedy**:
  - Filter leading indefinite articles (`a`, `an`, `the`) from definition subject extraction.
  - Implement built-in foundational definitions (`CPU`, `ALU`, `RAM`, `pi`, `Euler's number`) with explicit provenance (`ComponentOrigin.SYSTEM_KNOWLEDGE`).

### C. Coding Instruction Ignored
- **Mechanism**: `Write a Python program...` does not match `_is_knowledge_seeking` or greetings, falling through to default `DIRECT_CHAT`.
- **Root Cause**: ChakrMicro (3.44M parameters) cannot synthesize full Python ASTs autoregressively without a templated code synthesis pipeline or specialized tool.
- **Remedy**: Add `RouteType.CODE_GENERATION` triggered by `write a (python|c|java)? program|code to ...`. Provide verified, isolated runnable code templates with clear capability limitation reporting.

### D. False-Premise Ungrounded Questions
- **Mechanism**: `What is the capital of Mars?` treats `Mars` as a country or state with a capital, triggering retrieval.
- **Root Cause**: No false-premise check or ontological verification.
- **Remedy**: Identify celestial or non-sovereign entities queried for statehood/capitals and produce honest abstention or correction ("Mars is a planet and does not have a capital city.").

### E. Hinglish/Hindi Explanation Requests
- **Mechanism**: `Mujhe simple Hindi mein photosynthesis samjhao.` was routed to `DIRECT_CHAT`, producing dispersed token loops.
- **Remedy**: Detect language intent (`hindi mein`, `samjhao`, `kya hai`) and route to a multilingual explanation handler with structured terminology.

### F. Follow-Up Reference (`and What about ALU`)
- **Mechanism**: Evaluated in isolation without reading preceding turn context from `PersistentSessionMemory`.
- **Remedy**: In `SessionAdapter`, resolve conversational follow-up triggers (`and what about`, `tell me about it`, `how about`) using previous entity context.

### G. User Memory Profile (`My name is Ramesh`)
- **Mechanism**: Treated as generic chat; the user name was not stored in structured memory profile slots.
- **Remedy**: Add user profile slot extractor (`my name is <X>`, `call me <X>`) stored in session memory metadata and recalled on `what is my name`.
