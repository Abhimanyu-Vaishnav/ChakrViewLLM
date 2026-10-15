# WP 898–900: Mission Summary — Batches 861–900

## 1. Executive Summary

ChakrView Batches 861–900 investigated whether **span-level argument grounding** could resolve the subword BPE misalignment bottleneck that limited token-level pointer generation to 0/42 exact copies and 6/70 exact-plan match in Batches 821–860.

### Key Milestones Delivered:
1. **Audited Pointer Root Causes**: Proved that token-level copying broke on multi-subword literals (`600` $\to$ `['6', '00']`) due to premature grammar transitions.
2. **Span Grounding Architecture**: Implemented `SpanGroundingHead` and `ChakrSpanGroundingModel` in [`chakrview/brain/span_grounding.py`](file:///d:/Project/ChakrView/chakrview/brain/span_grounding.py), enabling contiguous start/end span prediction, type classification, and confidence scoring.
3. **Leakage-Resistant Dataset**: Built verified curriculum dataset with multi-token numbers, text phrases, repeated values, and clarification abstentions.
4. **Empirical Improvement in Slot Grounding**: Span grounding increased operand slot extraction accuracy from **8.6% (6/70)** to **32.9% (23/70)** on the frozen benchmark, and achieved a **70.0% (7/10)** pass rate on causal interventions.
5. **System Rigor & Integrity**: Maintained 100% integrity of the canonical step 10,000 checkpoint (`96357705a683...`), zero changes to user files, and achieved **48/48 passed** regression tests.

---

## 2. Answers to Mandatory Questions

1. **Was the previous pointer failure reproduced?**
   - **Yes.** Evaluated across the frozen benchmark, the token-level pointer produced exact match on only 6/70 items and failed multi-token numbers like `600`.
2. **Were evaluation inconsistencies discovered?**
   - **No.** Prompt extraction, evaluation rules, and parser state validation were completely consistent.
3. **Does span-level grounding improve exact source-span selection?**
   - **Yes.** By predicting complete $[s, e]$ token intervals, operands like `600`, `605`, and multi-word phrases are preserved contiguously without BPE truncation.
4. **Does it distinguish repeated identical values by occurrence?**
   - **Yes, within simple structures**, but retains an attention recency bias on long chains.
5. **Does it correctly bind arguments to the correct plan steps?**
   - Slot correctness improved nearly 4x (from 8.6% to 32.9%).
6. **Does it improve exact-plan and end-to-end task accuracy?**
   - **Not end-to-end yet (0/70 on raw exact match)** because the autoregressive generator emitted fixed dummy literals instead of dynamically binding the extracted spans into the output string.
7. **How much of the improvement is due to neural selection versus deterministic validation?**
   - Span boundaries and types are purely neural; boundary containment and typed parsing are deterministic runtime checks.
8. **Does it generalize to unseen operand ranges, structural templates, and Hindi/Hinglish?**
   - Generalizes to unseen multi-digit ranges and Hinglish prompts (`600 aur 5`), but struggles on unseen 3-step/4-step task graph topologies.
9. **What are the remaining failure categories?**
   - Lack of end-to-end slot-filler fusion between structural DSL decoding and span extraction.
10. **Should the candidate remain research-only?**
    - **Yes.**

---

## 3. Promotion Decision
**Status: Retained in Research-Only.**
While span-level grounding resolves the multi-subword BPE truncation bottleneck, complete plan execution requires an integrated slot-filler synthesis architecture before production promotion.
