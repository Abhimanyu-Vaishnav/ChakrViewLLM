# CR-EXP-16A Failure Analysis

## 1. Deep Dive: Why Did Arm D Score 66% on Greek Sequences but 0% on Names?

### A. The In-Context Pointer vs Pretrained Language Prior Tension
When primed with:
`'Repeat the name: Gautam\nName: '`
1. At step 0, the model executes the induction head correctly:
   - Target token: `74` (`'G'`)
   - Model prediction: `74` (`'G'`) with probability **0.9455**.
   - Pointer routing successfully attended to the name in the prompt!
2. At step 1, the model faces a competing objective:
   - In-context copying requires copying token `4011` (`'aut'`).
   - However, in the Stage B pretraining weights, token `74` (`'G'`) is overwhelmingly followed by `'race'` (`'Grace'`) or `'eeta'` (`'Geeta'`).
   - The language model prior for English name bigrams overpowers the weak multi-step induction signal:
     - `'rac'`: probability 0.7666
     - `'e'`: probability 0.0935
     - `'aut'`: probability 0.0203
3. Once step 1 emits `'rac'`, the autoregressive sequence is irrevocably derailed, finishing as `'Grace'`.

### B. Length Degradation on Sequences
On exact Greek sequences:
- Length 3: **90.5%** exact match.
- Length 4: **81.0%** exact match.
- Length 5: **64.5%** exact match.
- Length 6: **37.0%** exact match.

The primary failure mode at length 5 and 6 is **skip-ahead omission** (e.g., omitting the final token or substituting an adjacent letter). This aligns directly with known attention-head horizon limitations in 6-layer Transformers without explicit recurrence.

## 2. Mitigation Strategy for Stage B/C Pretraining
To achieve robust multi-token entity copying:
1. Include explicit subword chunking perturbations during pretraining.
2. Employ higher position-encoding resolution or relative position gating for long sequence maintenance.

