# STEP 243: Identity Diversity Curriculum

## Mission
Investigate whether increasing the diversity of training identities without changing the underlying semantic operation ($K \to V$) enables the compact associative circuit to generalize to unseen identities.

## Methodology & Identity Pools
Implemented in [`chakrview/cognition/identity_diversity_curriculum.py`](file:///d:/Project/ChakrView/chakrview/cognition/identity_diversity_curriculum.py):
- **Pool A**: Canonical alphanumeric characters (`A`-`H`, `1`-`8`).
- **Pool B**: Alphanumeric + arithmetic ASCII symbols (`A`-`L`, `1`-`9`, `+`, `-`, `=`).
- **Pool C**: Wide single-token vocabulary symbols (`A`-`O`, `1`-`9`, `+`, `-`, `=`, `:`, `;`, `.`).
- **Disjoint Evaluation Pool**: Strictly separated symbols (`P`-`Z`, `0`, `#`, `@`, `$`, `%`, `&`, `!`, `?`).
- **Token Integrity Guard**: Verifies that every identity in all pools maps to exactly one single token in `BPETokenizer`, preventing multi-token fragmentation artifacts.

## Results
- **Pool Progression**: Stages Pool A $\to$ Pool B $\to$ Pool C completed.
- **Held-Out Disjoint Accuracy**: Remains $0.0000$ across all 3 diversity pools.
- **Target Token Rank**: Mean target rank on disjoint evaluation remains elevated (~1800-2500 out of 4096).
- **Finding**: Increasing token identity vocabulary diversity during associative training is insufficient on its own to induce out-of-distribution symbol routing in the current cross-attention projection layer without upstream binding support.
