# Wave 520: v0.3 Adaptive Intelligence Contract

## 1. Executive Summary
Wave 520 establishes the foundational **Adaptive Intelligence Contract** for ChakrView v0.3.0.
This contract formalizes the taxonomy, lifecycle, and strict isolation rules governing how runtime experiences can be captured, abstracted, and reused without catastrophic forgetting, baseline weight corruption, or category conflation.

## 2. Core Taxonomy Boundaries

| Category | Definition | Reusability Scope | Parametric Changes |
|---|---|---|---|
| **Episodic Experience** | Raw single-task execution trajectory containing tokens, intermediate hops, confidence, and outcome. | Ephemeral or archived for consolidation; not a direct rule. | None ($\Delta W = 0$). |
| **Abstract Experience** | Generalized causal lesson derived across $\ge 2$ structurally related episodes. | Guides future dispatch and heuristic prioritization. | None ($\Delta W = 0$). |
| **Strategy** | Action policy defining path selection (Direct vs. Recurrent vs. Peer) and retry budgets. | Dynamic runtime optimization across task families. | None ($\Delta W = 0$). |
| **Knowledge** | Formally grounded entity-relation facts verified against contradictions. | Query answering and memory lookup. | None ($\Delta W = 0$). |
| **Parametric Update** | Governed neural weight modifications applied exclusively to isolated candidate adapters. | Structural representation refinement. | Candidate only; Baseline $\Delta W = 0$. |

## 3. Strict Non-Negotiable Rules
1. **No Memory Lookup as Neural Learning**: Retrieving a stored answer from episodic memory is classified as retrieval, never claimed as neural learning or generalization.
2. **No Single-Shot Over-Abstraction**: An `AbstractExperience` cannot be synthesized from a single episode; it requires at least two distinct corroborating episodes.
3. **Canonical Baseline Immutability**: The 3.44M-parameter ChakrMicro baseline remains frozen with $\Delta W = 0$. Any neural learning must occur in isolated candidate modules.
4. **Provenance Tracking**: Every contract artifact carries full cryptographic provenance (runtime version, benchmark version, seed, configuration hash).

**Wave 520 Status: PASS**
