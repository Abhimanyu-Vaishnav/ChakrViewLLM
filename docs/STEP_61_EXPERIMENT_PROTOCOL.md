# Step 61: Experiment Protocol & Benchmark Suite

## 1. Experimental Conditions

- **Condition A (Cold Start)**: Multi-step refactoring executed without prior memory. Validates sequential execution and initial experience consolidation.
- **Condition B (Memory Assisted)**: Re-encounter of multi-step refactoring task with memory active. Validates memory retrieval, revalidation, and guidance.
- **Condition C (Live Change Detection)**: Modifies live repository state. Validates diff extraction and classification (`COSMETIC`, `LOCAL`, `DEPENDENCY`, `BEHAVIORAL`, `ARCHITECTURAL`, `TEST_ONLY`).
- **Condition D (Memory Revalidation)**: Revalidates stored memory after upstream structural/dependency changes. Validates `STALE` status assignment.
- **Condition E (Unrelated Change Non-Interference)**: Introduces unrelated new modules. Validates that unrelated memories retain `VALID` status.
- **Condition F (Memory Ablation)**: Disables memory on multi-step task to establish causal delta.
- **Condition G (Negative Transfer Safety)**: Submits cross-domain security query against billing calculation memory. Validates safe abstention (`NO_MATCH` / `REJECTED`).
- **Condition H (Intermediate Regression Protection)**: Injects deliberately broken intermediate patch. Validates fail-closed stop, intermediate rollback, and pre-step state restoration.

## 2. Quantitative Verification Targets
1. State fingerprint determinism: $100\%$.
2. Change classification accuracy: $100\%$.
3. Stale memory detection rate: $100\%$.
4. Negative transfer rate: $0.00\%$.
5. Intermediate rollback integrity: Exact bit-match on pre-step content.
6. Baseline neural hash: Bit-exact invariance (`c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`).
