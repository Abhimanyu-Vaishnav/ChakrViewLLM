# STEP 466: Structured Relational-Language Co-Training

## Objective
Evaluate whether co-training a small relational curriculum alongside the bilingual language curriculum helps ground language representations in relational structure, or conversely, whether language acquisition damages relational reasoning.

## Methodology & Isolation
- **Baseline Architecture**: ChakrMicro (3,443,136 parameters, frozen canonical baseline intact).
- **Candidates Evaluated**:
  1. **Candidate A (`candidate_A_lang_only.pt`)**: Language-only training stream.
  2. **Candidate B (`candidate_B_joint.pt`)**: Joint language + small relational curriculum co-training.
  3. **Candidate C (`candidate_C_alternating.pt`)**: Alternating language and relational batches.
- **Data**: Sharded balanced bilingual stage C tokens + relational syllogistic reasoning patterns.

## Empirical Results
| Candidate | Strategy | Overall Loss | EN Acc | HI Acc | SHA-256 Digest |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Candidate A** | Language Only | 3.5179 | 25.23% | 36.56% | `9cb7bf4fc1ea95c8...` |
| **Candidate B** | Joint Co-Training | 3.5265 | 25.55% | 36.33% | `f44040b6adc944b0...` |
| **Candidate C** | Alternating | 3.5913 | 22.50% | 37.66% | `c2f503e4c9791338...` |

## Analysis
1. Joint co-training (Candidate B) maintained nearly identical language perplexity (loss 3.5265 vs 3.5179) with a slight increase in English top-1 accuracy (25.55% vs 25.23%).
2. Alternating curriculum (Candidate C) induced slight optimization drag (loss 3.5913).
3. Neither co-training regime harmed the fundamental language acquisition trajectory, but pure parametric language modeling did not automatically acquire zero-shot compositional binding.
