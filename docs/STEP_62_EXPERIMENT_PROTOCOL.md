# Step 62: Experiment Protocol & Benchmark Suite

## 1. Experimental Conditions

- **Condition A (First Strategy Direct Success)**: Clean execution of the top-priority branch with zero recovery transitions.
- **Condition B (First Strategy Fails -> Rollback -> Recovery)**: Initial branch fails intermediate verification, triggers atomic rollback, restores exact state fingerprint, and reroutes to a secondary robust branch.
- **Condition C (Unsafe Scope Precondition Rejection)**: Candidate branch attempting to modify unauthorized files is rejected prior to execution.
- **Condition D (All Strategies Fail -> Safe Abstention)**: All candidate branches fail verification, triggering fail-closed abstention.
- **Condition E (Rollback Fingerprint Exact Bit-Match)**: Post-rollback state fingerprint matches base repository fingerprint bit-exact.
- **Condition F (Memory-Assisted Strategy Selection)**: Prior validated semantic memory prioritizes matching refactoring branch.
- **Condition G (Stale Memory Rejection)**: Live structural changes render prior memory `STALE`, preventing it from satisfying memory-dependent preconditions.
- **Condition H (Deterministic Repeated Decisions)**: Identical inputs produce bit-exact identical branch selection decisions and fingerprints.
- **Condition I (Operational Bounds & Recovery Limits)**: Enforces `max_branches` and `max_recovery_transitions` caps.

## 2. Quantitative Verification Targets
1. Branch selection determinism: $100\%$.
2. Rollback fingerprint restoration rate: $100\%$ ($0$ deviations).
3. Safe abstention on complete failure: $100\%$.
4. Stale memory rejection: $100\%$.
5. Neural baseline hash: Bit-exact invariance (`c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`).
