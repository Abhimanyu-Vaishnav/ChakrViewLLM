# Step 76: Research -> Cognitive Context Specification

## Objective
Safely inject verified external evidence into the existing `StructuredReasoningEngine` and cognitive context without conflating retrieved claims with absolute facts.

## Design
The `CognitiveResearchIntegrator` bridges the `InvestigationResult` into the neural reasoning layer.

### Formatting for Neural Reasoning
External knowledge is always prefaced with its verification state. For example:
```
RESEARCH OUTCOME: VERIFIED
EVIDENCE ITEMS:
- [VERIFIED] (Source: src_1): The capital of France is Paris.
  Corroborated by: ev_9a8b7c, ev_1b2c3d
```

### Context Isolation
External knowledge is stored in the structured execution state under a distinct namespace (`external_evidence`), completely isolated from `established_facts` or `personal_memory`.

## Guiding Principles
- **Distinguishability**: The neural core must always be able to tell the difference between what it inferred, what it remembers, and what it just researched.
- **Epistemic Modesty**: If the investigation yields `CONTESTED` or `INSUFFICIENT` data, the system propagates that uncertainty rather than forcing a conclusion.
- **Traceability**: All formatted text maintains `source_id` references, allowing the model to cite its sources explicitly during final response generation.
