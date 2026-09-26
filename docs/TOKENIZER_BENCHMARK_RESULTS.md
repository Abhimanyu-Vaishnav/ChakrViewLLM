# ChakrView Tokenizer Empirical Benchmark Results

**Phase**: Step 2.3 — Corpus Engineering & BPE Vocabulary Experiment  
**Status**: RATIFIED EMPIRICAL REPORT  
**Date**: September 26, 2026  
**Hardware Context**: Intel Core i9-13900H (14 cores / 20 threads), 32 GB RAM, Windows 11, Python 3.14.7  

---

> ### SCIENTIFIC DISCLAIMER
> - The dataset used for this benchmark is intentionally small, controlled, and experimental (42,193 UTF-8 bytes across 8 categories).
> - Performance and compression ratios reported herein represent **empirical findings on this specific research corpus**.
> - These results must **NOT** be claimed as "universally optimal" or "best for Hindi globally".
> - All evaluations are governed by the strict invariant:
>   $$\text{Decode}(\text{Encode}(\text{text})) \equiv \text{text} \quad \forall \text{ inputs}$$
>   *"Tokenizer correctness takes precedence over compression."*

---

## 1. Consolidated Results Matrix

All four vocabulary candidates were trained on the identical 80% training split (383 items, 33,648 bytes) and evaluated on the identical 20% validation split (107 items, 8,545 bytes, 6,475 characters, 1,028 words).

| Candidate | Merges Trained | Actual Vocab Size | Validation Tokens | Tokens / Char | Tokens / Word | Bytes / Token (Compression) | Lossless Test | Encode Latency ($\mu\text{s}$) | Decode Latency ($\mu\text{s}$) | Vocab Table RAM |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$V = 2,048$** | 1,789 | 2,048 | 3,243 | 0.5008 | 3.15 | 2.63 | **PASS** (100%) | 402.2 | 3.14 | 435.5 KB |
| **$V = 4,096$** | 3,837 | 4,096 | 2,949 | 0.4554 | 2.87 | 2.90 | **PASS** (100%) | 513.1 | 2.99 | 898.1 KB |
| **$V = 8,192$** | 7,933 | 8,192 | 2,777 | 0.4289 | 2.70 | 3.08 | **PASS** (100%) | 763.0 | 3.39 | 1,852.5 KB |
| **$V = 16,384$** | 9,788* | 10,047* | 2,774 | 0.4284 | 2.70 | 3.08 | **PASS** (100%) | 877.6 | 3.50 | 2,228.1 KB |

*\*Note on $V = 16,384$*: On this research corpus, all adjacent pair combinations were completely exhausted after 9,788 merges (reaching an actual vocabulary of 10,047 tokens). Beyond this point, no further merges were possible.

---

## 2. Category-Specific Efficiency Breakdown (Tokens / Word)

Lower tokens per word indicates higher compression efficiency:

| Category | $V = 2048$ | $V = 4096$ | $V = 8192$ | $V = 16384$ | $\Delta (4096 \to 8192)$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Hindi (Devanagari)** | 2.63 | 2.43 | 2.23 | 2.23 | $-8.2\%$ tokens |
| **English** | 3.10 | 2.78 | 2.57 | 2.56 | $-7.5\%$ tokens |
| **Hinglish** | 2.06 | 1.77 | 1.57 | 1.57 | $-11.3\%$ tokens |
| **Code (Python/C/SQL)** | 3.02 | 2.60 | 2.52 | 2.52 | $-3.1\%$ tokens |
| **Numbers & Currency** | 3.44 | 3.20 | 3.11 | 3.11 | $-2.8\%$ tokens |
| **Mathematics & Logic** | 3.40 | 3.00 | 2.85 | 2.85 | $-5.0\%$ tokens |
| **Extended Unicode & Emojis** | 5.16 | 4.90 | 4.78 | 4.78 | $-2.4\%$ tokens |
| **Mixed / Code-Switched** | 2.25 | 2.00 | 1.84 | 1.84 | $-8.0\%$ tokens |

---

## 3. Numeric Strategy Experiment Results

Conducted across Candidate A (Individual Digits `\d`), Candidate B (Two-Digit Chunks `\d{1,2}`), and Candidate C (Normal unconstrained BPE) on candidates $V = 4096$ and $V = 8192$:

| Sample Test Input | UTF-8 Bytes | Strategy A Tokens (`\d`) | Strategy B Tokens (`\d{1,2}`) | Strategy C Tokens (Normal BPE) | Lossless Verified? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `1234567890` | 10 | 10 | 5 | 5 | **YES** |
| `20260926` | 8 | 8 | 4 | 3 | **YES** |
| `₹50000` | 8 | 6 | 5 | 2 | **YES** |
| `13.14159` | 8 | 8 | 5 | 3 | **YES** |
| `13900H` | 6 | 6 | 4 | 1 | **YES** |
| `2026-09-26` | 10 | 10 | 6 | 1* | **YES** |
| `12:45:30` | 8 | 8 | 6 | 6 | **YES** |
| Code snippet with numbers | 68 | 29 | 26 | 26 | **YES** |
| LaTeX math with numbers | 53 | 20 | 19 | 17 | **YES** |

*\*Critical Observation on Normal BPE (Strategy C)*:  
In normal BPE, frequent numbers such as `2026-09-26` or `13900H` merge into monolithic single tokens because they appeared in the training set. This harms arithmetic generalization because the model cannot reuse digit representations across unseen numbers.  
- **Strategy A** guarantees consistent positional alignment for arithmetic at the expense of sequence length.  
- **Strategy B** provides a practical middle ground (halving digit sequence length while preventing unbounded subword memorization).

---

## 4. Latency and Memory Analysis

```
Encode Latency vs Vocabulary Size:
  V = 2048:   402 µs  [████████]
  V = 4096:   513 µs  [██████████]
  V = 8192:   763 µs  [███████████████]
  V = 16384:  878 µs  [█████████████████]

Vocab RAM Footprint:
  V = 2048:   435 KB  [████]
  V = 4096:   898 KB  [████████]
  V = 8192:  1852 KB  [████████████████]
  V = 16384: 2228 KB  [███████████████████]
```

- **Decode Latency**: Invariant across vocabulary size ($\approx 3.0\text{--}3.5\,\mu\text{s}$ per line) because decoding is a direct byte-buffer concatenation.
- **Encode Latency**: Scales with vocabulary size and merge count. $V = 8192$ takes $48.7\%$ longer to encode than $V = 4096$.
- **Parameter Impact on Neural Core (Chakr-Micro)**:
  - $V = 2048 \implies 12.9\%$ of model in embeddings ($0.39\text{M}$ params)
  - $V = 4096 \implies 22.8\%$ of model in embeddings ($0.79\text{M}$ params)
  - $V = 8192 \implies 37.2\%$ of model in embeddings ($1.57\text{M}$ params)
  - $V = 16384 \implies 54.2\%$ of model in embeddings ($3.15\text{M}$ params)

---

## 5. Architectural Interpretation & Recommendation

1. **Failure of $V = 16,384$**:
   Moving from $V = 8192$ to $V = 16384$ yielded a negligible 3-token improvement ($0.1\%$) on validation data while inflating vocabulary memory and increasing encoding latency by $15\%$. On small-to-medium corpora, $V = 16384$ over-allocates parameters to memorizing rare subword combinations.
2. **Inflection Point at $V = 4,096$**:
   $V = 4096$ reduces validation tokens by $9.1\%$ compared to $V = 2048$ with only a $27\%$ increase in encoding latency. It maintains a healthy parameter ratio ($22.8\%$ in embeddings for Chakr-Micro), leaving $77.2\%$ of parameter capacity for transformer layer reasoning.
3. **Recommendation**:
   **$V = 4096$ is the best-performing candidate on this benchmark corpus.**  
   $V = 8192$ remains a viable candidate if corpus volume is substantially expanded in Step 3, while $V = 16384$ is empirically rejected for the Micro model tier.
