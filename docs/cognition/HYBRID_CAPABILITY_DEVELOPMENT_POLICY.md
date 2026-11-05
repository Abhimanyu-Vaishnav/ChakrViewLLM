# CHAKRVIEW HYBRID CAPABILITY DEVELOPMENT POLICY
## A Rigorous, Evidence-Based Engineering Standard for Neural Intelligence

### 1. The Capability-First Mandate

ChakrView's development objective is not the arbitrary completion of training milestones, incrementing version tags, or reducing training cross-entropy loss. The sole standard of progress is the **verifiable acquisition, measurement, and retention of genuine neural capabilities**.

$$\text{Intelligence} \ne \text{Parameter Count} \ne \text{Token Volume} \ne \text{Low Training Loss}$$

A model that generates fluent grammatical text while failing on randomized factual variables or inverted logic is not reasoning. Under this policy, every research cycle must be organized around measurable capabilities.

---

## 2. Experimental Foundation: ChakrMicro v0.1

1. **Immutable Baseline**:
   - The frozen 6-layer, 192-width, 6-head, 512-context ChakrMicro v0.1 architecture remains the primary experimental baseline.
   - All novel training methods, curricula, and loss objectives must first be evaluated on ChakrMicro v0.1 to determine the maximum capability achievable within this footprint before scaling.
2. **Evidence Bar for Architectural Changes**:
   - Architectural scaling (e.g. adding layers, width, recurrence, or auxiliary heads) is strictly forbidden unless:
     a) The target capability has been systematically tested on the baseline with clean, uncorrupted data.
     b) An information-theoretic or mechanistic bottleneck (e.g. attention head saturation or depth insufficiency) is empirically demonstrated.
     c) The scaled architecture is developed in an isolated experimental branch and benchmarked side-by-side against the baseline.

---

## 3. The 5-Phase Scientific Workflow

Every capability initiative must strictly follow this closed-loop progression:

```
[Phase 1: Baseline Measurement]
   Measure frozen parent checkpoint on targeted held-out capability benchmark.
   Record baseline accuracy, error distribution, and perplexity.
         │
         ▼
[Phase 2: Formal Hypothesis & Preflight Audit]
   Define the exact neural mechanism or curriculum expected to teach the capability.
   Verify zero train/val/test data leakage and unambiguous task formulation.
         │
         ▼
[Phase 3: Isolated Controlled Training]
   Execute training on an isolated branch with full checkpointing and RNG logging.
   Preserve parent checkpoints and canonical baselines without overwrite.
         │
         ▼
[Phase 4: Comprehensive Multi-Condition Evaluation]
   Evaluate Candidate on:
   - Condition A: Familiar/memorized associations.
   - Condition B: Out-of-distribution novel associations.
   - Condition C: Distractor and position permutations.
   - Condition D: Regression suite on previously acquired skills.
         │
         ▼
[Phase 5: Decision Gate & Integration]
   Candidate is promoted ONLY if it meets all numeric acceptance gates without
   catastrophic forgetting. Otherwise, report failure and iterate.
```

---

## 4. Separation of Internal Cognition and External Augmentation

ChakrView maintains a non-negotiable boundary between internal neural intelligence and external software crutches:

- **Neural Computation**: What the weights $\theta$ compute in a forward pass without external assistance.
- **External Tools / Augmentations**: Local Python interpreters, calculators, RAG search engines, SQLite state stores, and hardcoded heuristic regexes.
- **Evaluation Mandate**: A capability acquired via tool execution (e.g. calculator tool) must **never** be reported as internal neural reasoning. Benchmark scores must explicitly state the execution mode:
  1. *Neural-Only*
  2. *Neural + Session Memory*
  3. *Neural + Grounded Retrieval*
  4. *Neural + Tool Execution*

---

## 5. Lineage, Reproducibility & Regression Invariants

1. **Bit-Exact Resumption**: Every experiment must record its exact parent checkpoint SHA-256, model weight hash, tokenizer checksum, dataset manifest hash, and random seeds.
2. **Regression Prevention**:
   - No experiment may be declared successful if it damages previously established core capabilities (syntax, conversational identity, basic language perplexity) beyond defined tolerance bounds ($\le 5\%$ relative regression).
3. **Artifact Immutability**: Canonical files (`chakrview/brain/model.py`, `chakrview/runtime/session.py`, canonical checkpoints) must never be overwritten, renamed, or modified.

