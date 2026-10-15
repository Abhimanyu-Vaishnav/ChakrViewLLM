# Next Milestone Recommendation & Decision: Batches 821–860
## Engineering Decision & Technical Roadmap for Batches 861–900

**Date:** 2026-10-15  
**Milestone:** Batches 821–860  
**Repository:** `ChakrView`  
**Location:** `experiments/batch821_860/reports/NEXT_MILESTONE_RECOMMENDATION.md`

---

### 1. Explicit Answers to the Ten Mission Questions

1. **Was the previous 0/42 argument-fidelity result reproduced?**  
   **Yes, exactly reproduced (0/42).** On disjoint validation operands, the pure generative model emitted memorized training constants 100% of the time.
2. **Did the evaluator require corrections?**  
   **No.** An adversarial audit confirmed that the evaluator accurately parses AST argument dictionaries and isolates argument mismatches from syntax failures.
3. **Did learned pointer selection improve exact argument grounding?**  
   **Partially.** The pointer head successfully activated prompt tokens (e.g. learning to attend to `'6'` from `'600'`), but single-token pointing failed to copy multi-subword sequences.
4. **Does the model correctly handle repeated and ambiguous values?**  
   **Partially.** It differentiates positional order (e.g., in causal operand swap), but tends to prioritize earlier prompt tokens.
5. **Did the pointer mechanism improve exact-plan and semantic-plan accuracy?**  
   **No.** Exact plan match remained at **8.6%** (6/70) due to subword truncation in numeric argument strings.
6. **How much latency did the optimized grammar decoder save?**  
   Per-step masking time decreased from **8.888 ms to 6.210 ms** (a **30.1% speedup**).
7. **Did the optimization preserve grammar equivalence and safety?**  
   **Yes.** 100% of admitted candidates were mathematically proven to be valid grammar transitions in unit tests.
8. **How stable were results across independent seeds?**  
   **High stability.** Both Seed 42 and Seed 101 converged to low training loss (0.2787 and 0.1073) and produced identical validation exact-plan scores.
9. **What capabilities remain deterministic runtime features rather than neural capabilities?**  
   DAG acyclicity validation, tool whitelisting, and permission gating remain deterministic runtime guarantees. Goal prediction, action selection, and pointer attention are genuinely neural.
10. **Is the combined planner still research-only?**  
    **Yes, retain Research-Only status.** In accordance with promotion criteria, because exact plan match did not improve beyond 8.6%, the candidate must not be deployed to production.

---

### 2. Engineering Roadmap for Batches 861–900

To conquer the subword copy bottleneck:
1. **Multi-Span Continuous Pointer Mechanism:**
   Transition from single-token pointing to contiguous multi-token span copying (e.g., copying subword slices $[i, i+k]$ as atomic argument values).
2. **Compiled CUDA Prefix Trie:**
   Replace Python string validation with a tensorized C++/CUDA prefix trie to achieve $<50$ ms latency.
3. **Hybrid AST Extraction Route:**
   Allow the runtime to resolve argument slot references via neural span pointers before executing sandboxed tool calls.
