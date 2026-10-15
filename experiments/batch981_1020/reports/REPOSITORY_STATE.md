# WP 981–986: Master Repository State Audit

## 1. Verified Working Tree and Git Provenance
- **Current Branch**: `dev/v0.6-v0.7-validation`
- **HEAD Commit**: `338d773a629e0fd32610647984846b8859f0a777`
- **Milestone Message**: `feat(batch941-980): diagnose topology collapse, build focused arithmetic family, and eliminate recency bias`
- **Remote Synchronization**: `dev/v0.6-v0.7-validation` is synced with `origin/dev/v0.6-v0.7-validation` on GitHub.
- **Canonical Baseline Checkpoint**:
  - Path: `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
  - Expected SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
  - Measured SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (Verified Immutable)
- **User Protection**: `chakrview/runtime/session.py` has active user modifications that remain strictly uncommitted, untracked, and preserved.

---

## 2. Default Runtime vs Experimental Research Routes

| System Pathway | Primary Modules | Models & Weights Loaded | Production / Default Status |
| :--- | :--- | :--- | :--- |
| **Default Conversational Session** | `chakrview/runtime/session.py`, `chakrview/runtime/conversation.py` | ChakrMicro (`training_checkpoint_step_10000.pt`) | **PRODUCTION DEFAULT** |
| **Cognitive Memory & Retrieval** | `chakrview/cognition/federation/cognitive/models.py` | Vector / embedding memory adapters | **SUPPORTING RUNTIME** |
| **Structured Intent Router (Batches 541–580)** | `chakrview/runtime/neural_intent_router.py` | `candidate_step250_intent_understanding.pt` | **EXPERIMENTAL ROUTE** |
| **Hierarchical Task Planner (Batches 661–740)** | `chakrview/cognition/hierarchical_dialogue_planner.py` | `candidate_step150_neural_planner.pt` | **RESEARCH ONLY** |
| **Autoregressive Plan DSL (Batches 741–820)** | `chakrview/cognition/grammar_constrained_decoder.py` | `candidate_step300_seed101_grammar_planner.pt` | **RESEARCH ONLY** |
| **Neural Pointer-Generator (Batches 821–860)** | `chakrview/brain/pointer_generator.py` | `candidate_step300_seed101_pointer_planner.pt` | **RESEARCH ONLY** |
| **Neural Span Grounding (Batches 861–900)** | `chakrview/brain/span_grounding.py` | `candidate_step250_seed42_span_grounding.pt` | **RESEARCH ONLY** |
| **Unified Plan Fusion (Batches 901–940)** | `chakrview/cognition/unified_plan_fusion.py` | `candidate_step250_seed42_span_grounding.pt` | **RESEARCH ONLY** |
| **Topology & Positional Planner (Batches 941–980)**| `chakrview/cognition/neural_topology_planner.py` | Span Grounding + Deterministic Ordering | **RESEARCH ONLY** |
