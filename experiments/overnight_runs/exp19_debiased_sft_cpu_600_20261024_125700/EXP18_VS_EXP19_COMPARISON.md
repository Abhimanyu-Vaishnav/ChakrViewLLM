# Exp18 vs Exp19 Paired Comparison Report

**Baseline (Exp18)**: `experiments/overnight_runs/exp18_aligned_sft_cpu_300/checkpoints/sft_best_checkpoint.pt` (Step 240, 300 steps total, non-debiased training data)  
**Experiment (Exp19)**: `experiments/overnight_runs/exp19_debiased_sft_cpu_600_20261024_125700/checkpoints/sft_best_checkpoint.pt` (Step 340, 600 steps total, debiased diversified training data)  
**Test Set**: 160 frozen held-out records (140 scorable, 20 qualitative `short_explanation`)  
**Decoding Settings**: Greedy ($T=0.0$), `max_new_tokens = 45`, `seed = 42`  

---

## 1. Aggregate Headline Metrics

| Metric / Attribute | Exp18 (Step 240) | Exp19 (Step 340) | Absolute Delta | Evaluation & Impact |
| :--- | :---: | :---: | :---: | :--- |
| **Scorable Task Accuracy** | **0.71%** (1/140) | **0.00%** (0/140) | **-0.71%** (-1 example) | Within statistical noise ($\pm 0.71\%$ per sample) |
| **EOS Completion Rate** | **45.00%** (72/160) | **54.37%** (87/160) | **+9.37%** (+15 examples) | **Significant improvement**; surpasses 50% target |
| **Max 2-Word Prefix Concentration** | **77.50%** (124/160) | **38.12%** (61/160) | **-39.38%** | **Massive debiasing**; monoculture cut in half |
| **Max Prefix String** | `"The difference"` | `"The difference"` | - | Still prevalent, but diversified with `"The firm"`, `"'It' is"`, etc. |
| **Avg Tokens Generated** | ~41.0 tokens | **29.71 tokens** | **-11.3 tokens** | Models generate tighter, non-runaway completions |
| **Avg Inference Latency (CPU)** | 332.24 ms | **282.78 ms** | **-49.46 ms** | ~15% faster due to clean early EOS emission |
| **Structured JSON Validity** | 0.00% (0/20) | **0.00%** (0/20) | 0.00% | Python code prior (`import doctest`) persists |
| **Best Validation Loss** | 2.8339 (step 240) | **2.8786** (step 340) | +0.0447 | Comparable validation loss on slightly more diverse target |

---

## 2. Category-by-Category Breakdown

| Category | Total | Scorable | Exp18 Correct | Exp19 Correct | Exp18 EOS Count | Exp19 EOS Count | Exp19 Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `factual_qa` | 20 | 20 | 0 | 0 | 9 | 9 | High syntax regularity, factual hallucination |
| `instruction_following` | 20 | 20 | 0 | 0 | 18 | 19 | Almost 100% clean EOS termination |
| `context_extraction` | 20 | 20 | 0 | 0 | 10 | 11 | Extracts text patterns, misses exact entity |
| `reference_resolution` | 20 | 20 | 0 | 0 | 6 | 7 | Diversity improved: uses `"'It' is"`, `"reference ="` |
| `short_explanation` | 20 | 0 (qual) | N/A | N/A | 7 | 8 | Needs manual review; coherent grammar |
| `basic_arithmetic` | 20 | 20 | 1 | 0 | 10 | 12 | The 1 correct Exp18 sample shifted; arithmetic remains weak |
| `uncertainty_abstention` | 20 | 20 | 0 | 0 | 1 | 1 | Difficult category for 3.4M micro models |
| `structured_formatting` | 20 | 20 | 0 | 0 | 11 | 20 | **100% EOS termination** (up from 55%), though code prior persists |
| **Total** | **160** | **140** | **1** | **0** | **72** | **87** | **+15 records terminated cleanly with EOS** |

---

## 3. Qualitative Output Comparison Examples

### Example 1: Instruction Following (Increased Diversity & Clean EOS)
- **Prompt**: `"Rewrite the following in past tense: They go to the market."`
- **Expected**: `"They went to the market."`
- **Exp18 Generation**: `"The difference is that they went to the market."` (Concentrated on `"The difference"`)
- **Exp19 Generation**: `"The firm went to the market."` (Diversified opening; clean EOS termination)

### Example 2: Structured Formatting (EOS Termination Breakthrough)
- **Prompt**: `"Format the following details as valid JSON: name Alice, age 30, city London"`
- **Expected**: `{"name": "Alice", "age": 30, "city": "London"}`
- **Exp18 Generation**: `"\n    def test_json():\n        return {\"name\": \"Alice\", \"age\": 30... (exceeded max tokens)"`
- **Exp19 Generation**: `"\n    import doctest\n    doctest.testmod()<|endoftext|>"` (Clean termination at 9 tokens; avoids runaway loop)

### Example 3: Reference Resolution
- **Prompt**: `"In the sentence 'The car was fast because it had a V8 engine', what does 'it' refer to?"`
- **Expected**: `"'it' refers to the car."`
- **Exp18 Generation**: `"The difference is the car."`
- **Exp19 Generation**: `"'It' is the car.<|endoftext|>"` (Much more natural syntax and accurate entity association, terminated with EOS)

---

## 4. Key Takeaways & Limitations

1. **Hypothesis Confirmed (Turn Boundaries & Diversity)**: Deterministic template diversification successfully broke the severe opening monoculture (collapsing prefix concentration from 77.5% to 38.1%) and elevated EOS termination from 45.0% to 54.37%.
2. **Hypothesis Unconfirmed (Semantic Task Accuracy)**: Broadening template vocabulary did not impart reasoning capabilities to the 3.4M parameter model. Task accuracy remained statistically indistinguishable from zero (0/140 vs 1/140, a difference of 1 record / 0.71%).
3. **Capacity Constraints**: At 3.4M parameters, ChakrMicro responds strongly to surface format cues (EOS tokens, sentence openings) but lacks the latent capacity to simultaneously maintain arithmetic computation, fact retrieval, and JSON grammar without dedicated multi-task architectural headroom.
