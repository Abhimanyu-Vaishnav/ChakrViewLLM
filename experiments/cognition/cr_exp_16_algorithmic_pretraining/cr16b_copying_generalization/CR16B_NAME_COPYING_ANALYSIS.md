# CR-EXP-16B Name Copying Analysis

## 1. The Name-Copying Paradox
CR-16A reported high sequence copying on Greek letters (66%), but 0% on benchmark names. CR-16B conducted an exhaustive diagnostic across 220 name copying probes (CAT_G: 110 single-token names, CAT_H: 110 multi-token names).

## 2. Experimental Findings on Sealed Benchmark

### A. Single-Token & Western Names Succeed
When evaluated on single-word Western names in CR-16A Arm D:
- `'John'`: **6 / 6 exact matches (100.0%)** across multiple templates (`Repeat the name: John`, `Echo sequence: John`, `Copy: John -> `).
- `'Roy'`: **1 / 1 exact match (100.0%)** under `Echo sequence: Roy`.
- `'Victoria'`: **5 / 5 exact matches (100.0%)** across `Copy: Victoria -> ` and `Repeat the name: Victoria`.
- `'Elizabeth'`: **2 / 2 exact matches (100.0%)**.

This proves that **ChakrMicro CAN copy names** in context. The capability failure is **not** an architectural inability to process names or entity types.

### B. Why Indian Names Fail in Free Generation
Indian names in the benchmark (`Dev`, `Prem`, `Amit`, `Gautam`, `Mukesh`, `Pankhuri`) suffer from a specific two-stage failure mode:
1. **Initial Pointer Activation (Step 0):**
   - The model reliably routes attention from the prompt to the first token:
     - `Amit`: Target `68` (`'A'`) $\implies$ Model predicts `68` with **98.93% probability**.
     - `Gautam`: Target `74` (`'G'`) $\implies$ Model predicts `74` with **94.55% probability**.
     - `Mukesh`: Target `80` (`'M'`) $\implies$ Model predicts `80` with **89.06% probability**.
     - `Pankhuri`: Target `83` (`'P'`) $\implies$ Model predicts `83` with **94.10% probability**.
   - First-token copying accuracy on names reaches **78.0% to 88.0%**.
2. **Language Prior Hijacking (Step 1+):**
   - At position 1, the model must predict the rare subword piece (`Gautam` $\to$ `4011` `'aut'`).
   - However, in pretraining weights, `'G'` is overwhelmingly associated with high-frequency English completions (`'rac'` for `'Grace'`, `'eeta'` for `'Geeta'`).
   - The language prior ($P_{\text{pretrain}}(\text{'rac'} \mid \text{'G'}) = 0.7666$) completely overwhelms the weak in-context subword signal ($P_{\text{copy}}(\text{'aut'} \mid \text{'G'}) = 0.0203$).
   - The model emits `'rac'`, and autoregressive decoding completes `'Grace'` instead of `'Gautam'`.

## 3. Structural Conclusion
The failure on unfamiliar fragmented names is an **information-theoretic conflict between pretraining bigram language priors and subword BPE tokenization**, not a breakdown of attention pointer routing.

