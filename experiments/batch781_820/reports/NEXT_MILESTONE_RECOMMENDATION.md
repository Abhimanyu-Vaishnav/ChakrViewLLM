# Next Milestone Recommendation & Decision: Batches 781–820
## Engineering Decision & Technical Roadmap for Batches 821–860

**Date:** 2026-10-15  
**Milestone:** Batches 781–820  
**Repository:** `ChakrView`  
**Location:** `experiments/batch781_820/reports/NEXT_MILESTONE_RECOMMENDATION.md`

---

### 1. Explicit Answers to the Nine Mission Questions

1. **How much does grammar-constrained decoding improve syntax validity over unconstrained decoding?**  
   Grammar-constrained decoding improved overall syntax validity on the $N=70$ frozen benchmark from **52.9% to 60.0%**. Crucially, it completely unlocked 4-step plans (improving from **0.0% to 41.7%** valid 4-step plans).
2. **Does that improvement also increase exact-plan and semantic-plan accuracy?**  
   **No.** While syntax validity improved, exact plan accuracy remained at **8.6%** across both systems. The limiting bottleneck was not grammar syntax, but argument value hallucination.
3. **Does performance improve for three-step and four-or-more-step plans?**  
   **Yes, for syntax and structure.** Constrained decoding enabled the model to emit valid 4-step DAGs with correct intermediate dependencies ($0, $1, $2), which previously suffered 100% truncation in unconstrained generation.
4. **Does the model preserve exact prompt arguments?**  
   **No (0.0%).** Because held-out test operands (e.g. 600, 705, 800) were strictly disjoint from training distributions, pure generative softmax hallucinated high-frequency training numbers (e.g., 43, 13, 2).
5. **What is the effect on latency and generation failures?**  
   p50 latency increased from **664.18 ms to 2329.00 ms** (~3.5x overhead) due to token-by-token regex state validation. Generation failures due to delimiter malformations were eliminated.
6. **How stable are results across seeds?**  
   **High stability.** Both Seed 42 and Seed 101 demonstrated near-identical loss trajectories and identical validation exact-plan behavior (variance = 0.0%).
7. **Which capabilities come from neural prediction, which from the grammar parser, and which from the execution runtime?**  
   - **Neural Model:** Task goal prediction, step count estimation, tool selection (`calculator` vs `text_utility`), and causal clarification triggering.
   - **Grammar Parser:** EBNF syntax enforcement, quote closure, delimiter formatting, and preventing premature EOS.
   - **Execution Runtime:** DAG dependency resolution, sandbox isolation, and authorized tool dispatch.
8. **What remains unsafe or unreliable?**  
   Argument value hallucination on novel numerical/textual inputs. Unchecked execution of hallucinated numbers would produce incorrect business calculations.
9. **Should the new planner remain research-only, enter a restricted experimental trial, or be rejected?**  
   **Retain research-only status.** In accordance with Section C promotion criteria, because exact argument copy accuracy is 0.0%, the planner is not ready for an experimental trial or production runtime.

---

### 2. Engineering Roadmap for Batches 821–860

To resolve the argument copying bottleneck and achieve true zero-shot plan generalization:
1. **Pointer-Generator Attention Head:**
   Implement a dual-route generation layer where the model computes a copy probability $p_{\text{copy}}$ and points directly to prompt token representations for operand slots.
2. **Compiled Grammar State Trie:**
   Replace runtime regex evaluations with a pre-compiled token transition trie, reducing grammar constraint overhead from 2329 ms to $<100$ ms.
3. **Hybrid Verification Pipeline:**
   Before executing any planned tool step, verify that all non-dependent operands map to valid extracted spans in the user's prompt.
