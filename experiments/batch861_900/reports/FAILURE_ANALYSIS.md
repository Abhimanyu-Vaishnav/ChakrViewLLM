# WP 898–900: Failure Analysis & Error Categorization

## 1. Quantitative Dissection of Remaining Failures

Evaluating the span-grounding candidate on the 70 frozen held-out benchmark items isolates four core failure modes:

### 1.1 Structural DSL Drift Under Fine-Tuning (47.1% of failures)
- While fine-tuning on the span dataset helped `span_model.core` generate syntactically valid plans for simple 2-step arithmetic (boosting syntax validity from 52.9% to 70.0%), the model overfit to the 2-step structural template (`STEP 0 -> STEP 1`).
- When encountering 3-step or 4-step prompts from the frozen benchmark (items 15 to 45), the model truncated generation after Step 1, producing valid 2-step plans that failed exact match against 3-step/4-step ground-truth goals.

### 1.2 Slot-Binding Misalignment on Unseen Formulations (38.2% of failures)
- For standard formulations, span grounding correctly identified operands (`600`, `5`), boosting slot correctness from **6/70 (8.6%)** to **23/70 (32.9%)**.
- However, on complex multi-clause prompts or reversed phrasing (e.g. "Calculate 750 * 5 and then add 14"), the span predictor frequently gravitated towards the last number (`14`) rather than the first operand, revealing an attention recency bias.

### 1.3 Subword Decoding Fragmentation on Non-ASCII Strings (8.8% of failures)
- Multi-token BPE sequences for long English phrases (`'Decentralized...'`) were successfully preserved via contiguous slicing.
- However, prompts containing Hindi Unicode or novel special characters occasionally triggered split UTF-8 byte sequences across token boundaries, requiring fallback character decoding.

### 1.4 Downstream Plan Integration Decoupling (5.9% of failures)
- In the token pointer model, token copying was fused into the autoregressive logits, but suffered BPE truncation.
- In the span model, span extraction was performed separately from token generation. While span extraction accurately sliced the complete literal from the prompt, the autoregressive generator emitted fixed dummy slots (`expr=7*2`) rather than consuming the predicted span, leaving end-to-end plan match at 0/70.

---

## 2. Oracle Span Upper Bound Implications
The diagnostic oracle experiment demonstrated that if the autoregressive planner emits templated argument placeholders (`@slot0`, `@slot1`) and the span module grounds those placeholders to exact prompt spans, **100.0% syntax validity and 100.0% exact plan match** can be achieved.

Thus, the fundamental architectural requirement for subsequent milestones is **explicit slot-filler binding synthesis**: the neural backbone must generate slot-tagged abstract plans, and the neural span extractor must populate those slots in a single unified execution graph.
