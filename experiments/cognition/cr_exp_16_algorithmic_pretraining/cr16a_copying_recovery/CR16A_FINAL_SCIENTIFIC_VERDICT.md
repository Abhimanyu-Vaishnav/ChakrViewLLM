# CR-EXP-16A Final Scientific Verdict

## 1. Core Mission Questions Answered

### Q1: Why did CR-16 Stage A achieve 0% copying despite loss reduction?
**Verdict:** The loss reduction was illusory. Computing full-sequence causal language modeling loss over both prompt and target allowed the model to fit static syntax tokens and common unigram priors without activating induction-head circuits.

### Q2: Is ChakrMicro's architecture incapable of sequence copying?
**Verdict: NO.** The architecture ($d_{\text{model}}=192, 6\text{ layers}, 6\text{ heads}$) is fully capable of algorithmic sequence copying. When trained with target-only loss masking, ChakrMicro achieves:
- **66.0% exact match** on unseen Greek sequences (reaching **90.5%** on length-3 sequences).
- **100.0% exact match** on common word copying across diverse prompt templates.
- **100.0% first-token copy induction** on sequences.
- **78–88% first-token copy induction** on unseen fragmented names.

### Q3: What is the primary bottleneck for unseen name copying?
**Verdict:** Tokenizer subword fragmentation and language prior competition. Multi-piece unseen names suffer from bigram language prior hijacking at the second token step, converting `'Gautam'` into `'Grace'` despite correct first-token routing.

## 2. Gate Decision for Subsequent CR-16 Arms
- **Arm B (Key-Value Retrieval) & Arm C (Variable Binding):**
  - **AUTHORIZED TO PROCEED** using the Target-Only Masked Loss framework and Multi-Template serialization validated in CR-16A.
  - Full-sequence training must remain permanently prohibited for algorithmic pretraining.

