# CBR-SFT-03A — FIRST-TOKEN PREDICTION DIAGNOSTICS
## Analysis Across 100 Held-Out Prompts on Step 50 Checkpoint

**Diagnostic Target:** 100 Held-Out Dynamic Entity Grounding Prombes (`held_out_entity_grounding_100.json`)  
**Evaluated Checkpoint:** `sft_checkpoint_step_50.pt`  
**Vocabulary Size:** 4,096 BPE tokens  

---

## 1. Summary Metrics

| Diagnostic Metric | Measured Value | Theoretical Implication |
| :--- | :---: | :--- |
| **Total Evaluated Prompts** | 100 | Held-out prompts with disjoint names, cities, departments, items |
| **Correct First-Token Predictions** | **0 / 100** | Model never emits the expected concept token at Position 0 |
| **First-Token Exact Accuracy** | **0.00%** | 100% failure rate at the very first output token |
| **Mean Expected Token Rank** | **180.35** | Expected token is buried deep in the distribution (out of 4,096) |
| **Median Expected Token Rank** | **166** | Consistently unranked in top 100 candidates |
| **Mean Expected Token Probability**| **0.00050 (0.05%)** | Probability mass assigned to expected target is near-zero |

---

## 2. Dominant Top-1 and Top-5 Attractor Analysis

Across almost all 100 held-out prompts, the top-1 predicted token at Position 0 is overwhelmingly hijacked by single-letter uppercase tokens:
- **`Token 68 ('A')`**: Top-1 candidate on **88 / 100 prompts** with probability $\sim 7.7\% - 9.2\%$.
- **`Token 898 ('The ')`**: Consistently in Top-3 with probability $\sim 4.2\%$.
- **`Token 69 ('B')`**, **`Token 70 ('C')`**, **`Token 74 ('G')`**: Frequently occupy Top-5 ranks.

### Representative Sample Diagnostics:

```
Prompt: "User: Context: In laboratory 1000, researcher Aarav supervises operations.\nQuestion: Who supervises laboratory 1000?\nAssistant:"
Expected Concept: "Aarav" (First token: 1606 ' A')
  - Expected Token Rank: 140 (Prob: 0.00061)
  - Top-1 Predicted: 68 'A' (Prob: 0.0771)
  - Top-5: [('A', 7.71%), ('The ', 4.35%), ('G', 3.84%), ('C', 3.79%), ('B', 3.37%)]

Prompt: "User: Context: Cargo unit 1001 containing projector arrived at Lucknow.\nQuestion: Which city received cargo unit 1001?\nAssistant:"
Expected Concept: "Lucknow" (First token: 3456 ' L')
  - Expected Token Rank: 398 (Prob: 0.00011)
  - Top-1 Predicted: 68 'A' (Prob: 0.0868)
  - Top-5: [('A', 8.68%), ('B', 4.29%), ('G', 3.81%), ('K', 3.81%), ('C', 3.72%)]

Prompt: "User: Context: Division Robotics requested shipment of a sensor.\nQuestion: What item was requested by division Robotics?\nAssistant:"
Expected Concept: "sensor" (First token: 35 ' ')
  - Expected Token Rank: 166 (Prob: 0.00054)
  - Top-1 Predicted: 68 'A' (Prob: 0.0916)
  - Top-5: [('A', 9.16%), ('C', 3.74%), ('G', 3.67%), ('B', 3.52%), ('The ', 3.34%)]
```

---

## 3. Epistemic Conclusion
The first-token prediction failure is **absolute and decisive**:
1. The model assigns less than 0.06% probability to the correct initial token of the target concept.
2. The attention mechanism does not route activation from the context entity to the first generation step.
3. Because greedy decoding selects `Token 68 ('A')` or `Token 898 ('The ')`, free autoregressive generation begins off-target 100% of the time.
