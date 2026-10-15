# WP 898–900: Next Milestone Recommendation — Batches 901–940

## 1. Recommendation: Unified Abstract Slot-Filler Synthesis

### Context
Batches 861–900 proved that:
1. **Token-level pointer copying fails** because subword BPE splits multi-digit operands across steps where grammar constraints permit operators prematurely.
2. **Span-level grounding succeeds** in capturing complete operand intervals (boosting slot grounding accuracy from 8.6% to 32.9% and passing 70% of causal interventions).
3. **The Oracle experiment proved 100% exact plan match** is attainable if structural planning emits slot tokens (`@slot0`) that are dynamically substituted with grounded spans.

### Proposed Architecture for Batches 901–940:
1. **Abstract Plan DSL Generation**:
   Train the core transformer to autoregressively predict tool calls using abstract slot identifiers:
   ```dsl
   GOAL: compound_arithmetic
   STEP 0: tool_call | calculator | args={expr=@slot0*@slot1} | deps=[]
   STEP 1: tool_call | calculator | args={expr=$0+@slot2} | deps=[0]
   CLARIFICATION: NONE
   <END_PLAN>
   ```
2. **Integrated Slot Grounding**:
   Condition the `SpanGroundingHead` on the generated step and argument name (e.g. `step0.expr.slot0`) to directly bind extracted prompt text into the execution graph.
3. **Compiled Prefix Trie for Grammar Decoding**:
   Compile Plan DSL states into a CUDA-accelerated prefix trie to eliminate CPU-GPU roundtrips during token masking.
