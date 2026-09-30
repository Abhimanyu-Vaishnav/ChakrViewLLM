# ChakrView Step 57: Experiment Protocol

- **Document Version**: 1.0.0
- **Status**: Ratified Protocol
- **Subject**: Controlled 3-Condition Cognitive Learning & Transfer Benchmark

---

## 1. Experimental Conditions

To cleanly distinguish **contextual memory assistance** from **parameter adaptation** without confusing the two, three conditions are evaluated on identical tasks:

### CONDITION A: Baseline Core (No Memory)
- Input: Task prompt only.
- Weights: Frozen baseline ChakrMicro (`c5571c...`).
- Memory: None.

### CONDITION B: Memory-Conditioned (Retrieved Experience)
- Input: Task prompt prepended with `<CORTEX_CONTEXT>` containing `reusable_pattern` from previous verified episode.
- Weights: Frozen baseline ChakrMicro (`c5571c...`).
- Memory: Active episodic retrieval.

### CONDITION C: Learned Adapter + Memory
- Input: Task prompt + `<CORTEX_CONTEXT>`.
- Weights: Frozen baseline ChakrMicro + mounted `NativeTaskAdapter` adapted from verified experiences.
- Memory: Active episodic retrieval.

---

## 2. Test Task Triplet Structure (Preventing Data Leakage)

To test true transfer vs mere prompt copying vs unrelated tasks:
- **Task A (Anchor Task)**: Simple function repair (`add` returning `a - b` corrected to `a + b`).
- **Task A (Repeat)**: Exact same task presented again after experience is stored.
- **Task A' (Related Transfer Task)**: Same structural error pattern with new identifiers (`sub` returning `a + b` corrected to `a - b`).
- **Task B (Unrelated Control Task)**: State transition problem (`State: DOOR is LOCKED` -> `OPEN`).

---

## 3. Metrics Tracked

1. **Success Rate**: Pass rate across Task A, A', and B.
2. **Attempts to Converge**: Initial attempt vs subsequent attempts.
3. **Delta Experience ($\Delta_{\text{exp}}$)**: $Y_{\text{after}} - X_{\text{before}}$.
4. **Structural Transfer**: Performance on Task A' compared to Task A.
5. **Baseline Immutability**: Verification that $\Delta W_{\text{baseline}} \equiv 0$ across all conditions.
6. **Rollback Verification**: Proving that unmounting or rejecting an adapter restores base performance with zero residue.
