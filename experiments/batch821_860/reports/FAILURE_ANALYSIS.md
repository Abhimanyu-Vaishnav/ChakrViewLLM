# Failure Analysis & Architectural Diagnostics: Batches 821–860
## The Subword BPE Copy Bottleneck & Causal Plan Execution Boundaries

**Date:** 2026-10-15  
**Milestone:** Batches 821–860  
**Repository:** `ChakrView`  
**Location:** `experiments/batch821_860/reports/FAILURE_ANALYSIS.md`

---

### 1. Empirical Findings & The Subword Tokenization Bottleneck

While `ChakrPointerMicro` successfully implemented dual-distribution pointing and reduced NLL loss from 0.5360 down to 0.1073 during training, exact plan match on the $N=70$ held-out benchmark remained at **8.6%**.

Forensic inspection of generated token sequences revealed the exact failure mechanism:

#### The Subword BPE Misalignment
1. **Prompt Tokenization:**
   In prompt `Calculate 600 * 5`, the number `600` is split by the 4,096-token BPE tokenizer into two tokens:
   - Token ID `84`: `'6'`
   - Token ID `1202`: `'00'`
2. **Attention Pointing Artifact:**
   The pointer head attends to the first subword token (`'6'`). It correctly copies `'6'` with high probability, but at the next generation step, the grammar state allows operators or digits, causing the model to emit `+3` instead of the trailing subword `'00'`.
   - Result: `expr=6+3` instead of `expr=600*5`.
3. **Core Conclusion:**
   A standard single-token pointer attention mechanism is designed for character-level or whole-word vocabularies. In a subword BPE model, **multi-token spans** must be copied as continuous multi-step pointer sequences rather than single-step token picks.

---

### 2. State-Indexed Optimization Analysis

The optimized decoder reduced per-step masking time from 8.888 ms to 6.210 ms (a **30.1% speedup**), but total generation time still hovered around ~1.7 seconds per plan.
- The Python regex state machine still executes hundreds of substring checks per step.
- Achieving true real-time generation ($<100$ ms) requires a static C++/CUDA prefix trie tensor compiled directly into PyTorch.
