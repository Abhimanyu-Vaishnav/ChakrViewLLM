# WP 979–980: Next Milestone Recommendation — Batches 981–1020

## 1. Recommendation: Neural Step Classification & Direct Slot Grammar Tokens

### Key Evidence from Batches 941–980:
1. Exact plan match on simple arithmetic increased from **0% to 36.4%** by enforcing left-to-right positional slot binding.
2. The remaining failures on 3-step and 4-step tasks are caused by the lack of an autoregressive mechanism that determines when a plan requires an additional dependency step.

### Next Milestone Objectives:
1. **Extend Transformer Vocabulary with Dedicated Slot Tokens**:
   Directly emit `<SLOT_0>`, `<SLOT_1>`, `<SLOT_2>` within the autoregressive Plan DSL stream.
2. **Joint Topology-Span Pretraining**:
   Train the core causal model on multi-step arithmetic curricula (1-step through 4-step) using explicit slot tokens.
3. **Controlled Runtime Routing**:
   Expose the compound arithmetic planner as an experimental capability flag while keeping default routing safe and intact.
