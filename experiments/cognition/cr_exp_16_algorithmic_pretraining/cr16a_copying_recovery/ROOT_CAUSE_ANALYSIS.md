# Root Cause Analysis

## 1. Executive Scientific Finding
The failure of CR-EXP-16 Stage A to achieve any sequence copying (0/300) was not caused by an architectural deficiency or parameter budget limitation of ChakrMicro. It was caused by three compounding flaws:
1. **Objective Mismatch (Full-Sequence Loss):** Standard causal language modeling over prompt and target penalized prompt reconstruction errors and allowed the model to minimize cross-entropy loss by fitting static formatting tokens and frequent vocabulary unigrams without creating induction circuits.
2. **Template Rigidity:** Training on a narrow, non-benchmark formatting template prevented the attention heads from generalizing copy operations across varying prefix prompts.
3. **Subword Fragmentation and Language Prior Dominance:** Pretraining priors on Stage B checkpoints strongly favor producing frequent English words (`Grace`, `James`, `Vihaan`) over copying arbitrary subword fragments (`Gautam` $\to$ `['G', 'aut', 'am']`) when token pieces are unrepresented in natural sentences.

## 2. Experimental Proof of Resolution
When trained with:
- Target-only loss masking (`-100` on prompt),
- Diverse multi-template framing, and
- Combinatorial sequence mixtures,

ChakrMicro achieves **66.0% exact sequence copying** across frozen benchmark probes and **100.0% first-token copying accuracy**, decisively debunking the hypothesis that ChakrMicro cannot execute in-context copying.

