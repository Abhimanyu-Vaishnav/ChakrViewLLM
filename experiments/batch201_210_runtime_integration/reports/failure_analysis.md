# Failure Analysis: Interactive Runtime Integration & Multilingual Relation Binding
**Milestone:** Batches 201–210  
**Repository:** ChakrView  
**Status:** Completed & Evaluated  

---

## 1. Executive Summary

This diagnostic analysis records the failure modes, root causes, mitigations, and residual limitations observed across Batches 201 to 210 during runtime integration and multilingual relation-binding experiments.

---

## 2. Failure Mode Analysis

### Failure Mode 1: Query-Entity Overlap Bias on Novel Templates
- **Observed Behavior:** On held-out entity test tuples where question phrasing differed from training templates (e.g., *Euler number e*), pointer attention without query dampening tended to copy early context tokens or query entity numbers rather than the target attribute span.
- **Root Cause:** In small 3.4M parameter models, self-attention maps have high dot-product affinity between token representations of repeated entity words, drawing the pointer towards the query subject rather than moving across the causal relation to the target object.
- **Mitigation:** Applied query-token mask dampening (`scores - 3.5 * overlap_expanded.float()`), suppressing attention to prompt query tokens.
- **Residual Limitation:** On unseen relations where the target value is adjacent to other numeric entities in context, pointer precision remains bounded (25.0% EM on held-out test clusters).

---

### Failure Mode 2: Multi-Byte Devanagari UTF-8 Boundary Truncation
- **Observed Behavior:** Unconstrained vocabulary generation produces fragmented bytes or generic English table tokens when prompted with Devanagari script (e.g. `इन्` instead of `विद्युत सुचालक`).
- **Root Cause:** Small byte-level BPE vocabularies require precise autoregressive byte continuation sequences to form complete multi-byte Unicode characters. Without adequate training examples, token transitions break.
- **Mitigation:** `TokenTrie` constrained decoding enforces transitions along valid token ID sequences extracted from candidate context spans.
- **Residual Limitation:** TokenTrie decoding requires that candidate spans exist in the supplied or retrieved context. It is a constrained selection mechanism, not open-ended Hindi generation.

---

### Failure Mode 3: Conflicting Context Information
- **Observed Behavior:** When context contains opposing factual claims (e.g., two distinct boiling points for liquid nitrogen), neural generation and unguided extraction previously selected whichever number appeared first.
- **Mitigation:** `EvidenceExtractor` extracts all matching attribute instances and sets `has_conflicts=True` if multiple disparate values are discovered, prompting `GroundedSessionAdapter` to abort the query cleanly with an explicit explanation.

---

### Failure Mode 4: False Positive Pointer Hallucination on Absent Entities
- **Observed Behavior:** When a query targets an entity absent from context (e.g., *"What is the capital of Mars?"* over Carbon context), unguided pointer attention would copy irrelevant numbers from the context (e.g. `3915 K`).
- **Mitigation:** `GroundedSessionAdapter` parses the subject entity from the query and verifies entity presence in the context prior to permitting pointer fallback. If the entity is absent, the system safely returns *"Not mentioned in context."*
