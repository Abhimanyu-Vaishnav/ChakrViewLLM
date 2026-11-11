# CR-EXP-16B Failure Analysis

## 1. Primary Failure Modes Identified

### A. The Horizon Cliff at Length $\ge 7$ (CAT_F: 0/110)
In sequences of 7 to 10 tokens:
- At steps 0 to 4, pointer attention correctly tracks the prompt sequence.
- Beyond step 4, the attention distribution flattens, resulting in **skip-omissions** (dropping tokens 5 and 6) or **premature termination** (emitting `\n` or EOS).
- In a 6-layer Transformer with $d_{\text{model}}=192$, attention heads lack dedicated positional recurrence registers to maintain pointers over long token spans without explicit memory scratchpads.

### B. Pretrained Language Prior Competition on Fragmented Names (CAT_G & CAT_H)
- Single-token Western names (`John`, `Mary`, `Victoria`) succeed because their token pieces form standard, high-frequency lexical entries.
- Unfamiliar Indian names (`Gautam` $\to$ `['G', 'aut', 'am']`) fail because pretraining bigram language priors ($P(\text{'rac'} \mid \text{'G'}) = 0.766$) override the in-context copy signal at step 1.

### C. Alphanumeric Token Fragmentation (CAT_J: 0/100)
- Tokens such as `A1`, `B2`, `C3` tokenize into character + digit pairs (`['A', '1']`).
- Because digits have distinct positional encodings and lack pretraining co-occurrence, the model failed to generalize pointer tracking to arbitrary alphanumeric codes.

### D. Multi-Turn Dialogue Framing (CAT_N: 0/100)
- In CAT_N (`System: ... User: ... Assistant: `), the model attempted conversational reply generation (`"I have noted your sequence."`) rather than verbatim replication, demonstrating that dialogue formatting triggers conversational generation priors rather than algorithmic copying heads.

