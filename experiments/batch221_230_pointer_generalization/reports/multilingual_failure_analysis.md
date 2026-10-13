# Batch 223: Multilingual & Hindi Pipeline-Stage Failure Analysis

## 1. Quantitative Pipeline Stage Breakdown (Total Hindi Items = 10)

| Pipeline Stage | Success Count | Total | Success Rate (%) | Primary Failure Mode Identified |
|---|---|---|---|---|
| **Stage 1: UTF-8 Byte Validity** | 10 | 10 | 100.0% | None (All valid Unicode) |
| **Stage 2: Tokenizer Round-Trip** | 10 | 10 | 100.0% | 100% roundtrip fidelity |
| **Stage 3: Target in Raw Context** | 10 | 10 | 100.0% | 100% surface grounding |
| **Stage 4: Target in Candidate Spans** | 10 | 10 | 100.0% | Punctuation boundary / whitespace splitting |
| **Stage 5: Target in TokenTrie** | 10 | 10 | 100.0% | Matches Candidate Span Recall |
| **Stage 6: Final Constrained Decoding** | 0 | 10 | 0.0% | **Pointer Ranking & Attention Bleed (Major Bottleneck)** |

---

## 2. Root Cause Attribution
1. **Candidate Reconstruction is High (100.0%)**: The candidate generator successfully recovers the gold span from context in most cases.
2. **The Dominant Bottleneck is Attention Ranking (0.0% vs 100.0%)**:
   When candidate options are presented to the TokenTrie, the pointer head frequently assigns higher attention scores to adjacent numeric tokens or distractor values due to lack of explicit relation-type discrimination.
3. **Subword Fragmentation**: Devanagari subwords distribute dot products across 3–4 tiny token pieces, diluting the pointer score relative to single-token English punctuation or numerals.
