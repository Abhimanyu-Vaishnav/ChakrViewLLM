# CBR-SFT-03A — TEACHER FORCING VS. FREE AUTOREGRESSIVE GENERATION DIVERGENCE

---

## 1. Divergence Location Distribution

Across all 100 evaluated held-out prompts, token-level prefix matching was measured at positions $k \in \{1, 2, 4, 8, 16\}$:

| Measurement Metric | Measured Result | Interpretation |
| :--- | :---: | :--- |
| **Prefix Match at 1 Token** | **0 / 100 (0.0%)** | Zero generation paths started on-target |
| **Prefix Match at 2 Tokens** | **0 / 100 (0.0%)** | Zero generation paths preserved two tokens |
| **Prefix Match at 4 Tokens** | **0 / 100 (0.0%)** | Zero generation paths preserved four tokens |
| **Earliest Divergence Position = 0** | **100 / 100 (100.0%)**| **100% of samples diverge at Position 0** |

---

## 2. Token-by-Token Sequence Comparison

In every evaluated sample, divergence occurs immediately at the first generated token:

```
Sample 1 (HELD-ENTITY-0001):
  Expected Concept: 'Aarav' -> Tokens: [1606 (' A'), 280 ('ra'), 2306 ('v')]
  Generated Tokens: [68 ('A'), 113 ('n'), 103 ('d'), 117 ('r'), 279 ('on'), 267 ('e'), 302 (' = '), 53 ('200.')]
  Generated Text:   'Androne = 200.'
  Divergence Point: Position 0 (' A' != 'A')

Sample 2 (HELD-ENTITY-0002):
  Expected Concept: 'Lucknow' -> Tokens: [3456 (' L'), 425 ('uck'), 110 ('n'), 768 ('ow')]
  Generated Tokens: [68 ('A'), 113 ('n'), 103 ('d'), 117 ('r'), 279 ('on'), 267 ('e'), 302 (' = '), 94 ('[http')]
  Generated Text:   'Androne = [http://w'
  Divergence Point: Position 0 (' L' != 'A')

Sample 3 (HELD-ENTITY-0003):
  Expected Concept: 'sensor' -> Tokens: [35 (' '), 118 ('s'), 275 ('en'), 836 ('sor')]
  Generated Tokens: [68 ('A'), 113 ('n'), 103 ('d'), 117 ('r'), 279 ('on'), 267 ('e'), 302 (' = '), 94 ('[http')]
  Generated Text:   'Androne = [http://w'
  Divergence Point: Position 0 (' ' != 'A')
```

---

## 3. Root Cause Attribution

The failure of free autoregressive generation is caused decisively by **Incorrect First-Token Prediction**:
1. **Not context truncation:** Prompt token counts (35 to 55 tokens) are well within the 512 context window.
2. **Not tokenizer boundary mismatch:** Delimiters and whitespace prefixes are bit-exact between dataset encoding and runtime session.
3. **Not EOS misbehavior:** The divergence occurs before EOS is ever sampled.
4. **The Mechanism:** Because Position 0 fails, the model emits `'A'` or `'The '`. The conditioned sequence now becomes `[Prompt] + 'A'`. At this point, the attention context matches Stage C pretraining Wikipedia entries (such as `"Androne"` or `"California"`), pulling the subsequent tokens into pretraining attractor loops.
