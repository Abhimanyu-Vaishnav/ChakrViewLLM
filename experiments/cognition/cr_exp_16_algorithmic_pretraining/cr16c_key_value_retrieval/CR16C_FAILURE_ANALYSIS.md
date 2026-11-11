# CR-EXP-16C Failure Analysis

## 1. Deep Dive: Why Did CAT5, CAT6, and CAT7 Score 0%?

### A. The Vocabulary-Binding Bottleneck
In CAT5 (familiar keys with unseen values), CAT6 (unseen keys with familiar values), and CAT7 (familiar keys with Greek letter values):
- The prompt structures were identical to CAT1.
- However, when queried with `fabric = Silk` or `weapon = Dagger`:
  - The model outputted familiar vocabulary items (`'Snow'`, `'Chisel'`, `'Hammer'`).
  - It failed to bind the query key to the novel token string.

### B. Mechanistic Explanation
1. During 400 steps of retrieval pretraining, the attention heads learned to map **specific query embeddings** (e.g. `query: city`) to **specific value clusters** (e.g. `Delhi`, `London`, `Paris`) within context.
2. The attention circuit did NOT learn a general, content-agnostic pointer operation (like `"copy token at position after '='"`). Instead, it learned a **key-conditioned associative lookup** over familiar embedding spaces.
3. When novel keys or novel values appear, the associative key-query projection fails to produce high attention weights, and the feedforward layers fall back to high-probability frequent values.

## 2. Regression Analysis: Catastrophic Forgetting of Copying
When evaluated on the frozen CR-16 exact sequence copying benchmark (`Echo sequence: delta beta omicron zeta\nOutput: `):
- Retrieval Arm A scored **0 / 100 (0.0%)**.
- Instead of copying the sequence, the model generated single words (`'Delhi'`, `'Red'`).
- This confirms that fine-tuning purely on single-value retrieval overwrote the induction-head circuits acquired in CR-16A.
- **Future Implication:** Stage D curriculum must co-train retrieval and sequence copying in a unified multi-task mixture to prevent capability collapse.

