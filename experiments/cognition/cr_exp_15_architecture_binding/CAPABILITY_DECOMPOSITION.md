# CR-EXP-15 Phase 2: Capability Decomposition

## 1. Principle of Atomic Capability Isolation

In CR-EXP-14, evaluating all aspects of conversational working memory in a single complex dialogue turn resulted in 0% accuracy across all models. While mathematically true that none of the models completed the full task, this monolithic score failed to isolate the locus of failure.

To determine scientifically why ChakrView fails strict subject-value binding on unseen entities, we decompose the cognitive working memory pipeline into eight distinct, independently scored capabilities:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        COGNITIVE PIPELINE                             │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Subword Perception & Copying (Reproduction of unfamiliar names)     │
│                                ↓                                       │
│ 2. In-Context Entity Selection (Choosing requested entity among 2-3)   │
│                                ↓                                       │
│ 3. Static Entity-Value Association (Single-turn: Entity X is in City Y) │
│                                ↓                                       │
│ 4. Single-Entity State Update (Temporal recency: X in Y -> X in Z)     │
│                                ↓                                       │
│ 5. Multi-Entity State Maintenance (Disjoint slots: X in Y, W in Z)     │
│                                ↓                                       │
│ 6. Interleaved Cross-Entity Updates (Update W without altering X)      │
│                                ↓                                       │
│ 7. Delayed Interleaved Dialogue Retrieval (Surviving distractor turns) │
│                                ↓                                       │
│ 8. Epistemic Calibration & Abstention (Refusing unanswerable queries) │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Definitions and Formal Scoring Criteria for the 8 Sub-Capabilities

### A. Entity Copying (Task A)
- **Prompt:** `Repeat the name: [Unseen Name]. Name:`
- **Cognitive Operation:** Unconditional autoregressive reproduction of unfamiliar subword sequences.
- **Scoring:** Strict string match: `[Unseen Name].lower() in generated.lower()`.
- **Target Question:** Does the model possess the mechanical capability to emit novel multi-token names without substituting familiar training names?

### B. Entity Identification / Selection (Task B)
- **Prompt:** `Context: [Name1] and [Name2] are present. Question: Who was mentioned first? Answer:`
- **Cognitive Operation:** Selective attention and token routing based on a syntactic role query.
- **Scoring:** Queried entity present; non-queried entity absent.

### C. Static Entity-Value Binding (Task C)
- **Prompt:** `Context: [Name1] is in [City1]. [Name2] is in [City2]. Question: Where is [Name1]? Answer:`
- **Cognitive Operation:** Simultaneous binding of subject token representation to attribute token representation without temporal state updates.
- **Scoring:** `Name1` present AND `City1` present; `Name2` and `City2` absent.

### D. Single-Entity State Updates (Task D)
- **Prompt:** `Turn 1: [Name1] is in [City1]. Turn 2: [Name1] moved to [City2]. Question: Where is [Name1] now? Answer:`
- **Cognitive Operation:** Temporal state overwriting for a single entity representation.
- **Scoring:** `Name1` present AND `City2` present; `City1` absent.

### E. Multi-Entity State Tracking (Task E)
- **Prompt:** `Turn 1: [Name1] in [City1]. Turn 2: [Name2] in [City2]. Turn 3: [Name1] moved to [City3]. Question: Where is [Name2]? Answer:`
- **Cognitive Operation:** Maintaining independent state representations for multiple entities and retrieving un-updated entity state.
- **Scoring:** `Name2` present AND `City2` present; `City1` and `City3` absent.

### F. Interleaved Updates (Task F)
- **Prompt:** `Turn 1: [Name1] in [City1]. Turn 2: [Name2] in [City2]. Turn 3: [Name1] moved to [City3]. Turn 4: [Name2] moved to [City4]. Question: Where is [Name1]? Answer:`
- **Cognitive Operation:** Non-interfering cross-entity updates across multi-turn dialogue.
- **Scoring:** `Name1` present AND `City3` present; stale value (`City1`) and cross-entity values (`City2`, `City4`) absent.

### G. Delayed Retrieval with Distractors (Task G)
- **Prompt:** Same as F with intervening conversational distractor turns between last update and query.
- **Cognitive Operation:** Long-horizon attention routing across conversational noise.
- **Scoring:** Strict subject-value match; stale, distractor, and other entity values absent.

### H. Calibrated Abstention (Task H)
- **Prompt:** Dialogue with entities and locations, followed by an unanswerable query (e.g. querying an unmentioned entity or unmentioned attribute).
- **Cognitive Operation:** Recognizing lack of ground truth context and emitting conservative abstention cues.
- **Scoring:** Presence of calibrated abstention cues (`"does not specify"`, `"unknown"`, `"cannot"`, etc.).
- **Metrics:** Precision, Recall, and F1-score reported with explicit denominators, strictly separated from Tasks A–G.

