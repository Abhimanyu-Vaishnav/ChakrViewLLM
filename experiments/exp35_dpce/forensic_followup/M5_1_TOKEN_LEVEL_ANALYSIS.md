# CHAKRVIEW — M5.1 TOKEN-LEVEL LOSS ANALYSIS
## Disaggregation of Template, Entity, and EOS Loss Across Training Checkpoints

**Experiment:** Exp35-DPCE Forensic Audit  
**Milestone:** M5.1  
**Date:** 2026-10-30  

---

## 1. THE TOKEN ALLOCATION RATIO IN TRAINING LABELS

In the 1,000-example Exp35 training set (`exp35_train.jsonl`, 900 records evaluated), responses were analyzed at the token level:

$$\text{Total Supervised Response Tokens} = 19,932$$

| Token Category | Token Count | Proportion of Loss Tokens | Role in Target Response |
|---|---|---|---|
| **Static Template Tokens** | 15,738 | **78.96%** | Structural syntax (`"At station"`, `"resides in"`, `"belongs to"`) |
| **Dynamic Entity Tokens** | 3,294 | **16.53%** | Target entity names, cities, codes, attributes |
| **EOS Stop Tokens** | 900 | **4.52%** | `<|endoftext|>` sequence terminator |

**Key Finding:** Almost four-fifths (78.96%) of all gradient updates during SFT were directed at predicting standard syntactic scaffolding rather than the dynamic entity values.

---

## 2. CROSS-CHECKPOINT PER-TOKEN LOSS DYNAMICS

Evaluating the full 100-record validation split (`exp35_validation.jsonl`) under teacher forcing revealed how each token category evolved:

| Checkpoint | Template Token Loss | Entity Token Loss | EOS Token Loss | Overall Teacher-Forcing Accuracy |
|---|---|---|---|---|
| **M3 Step 250 (Baseline)** | 5.2927 | 5.7449 | 0.0724 | 11.7% |
| **M2 Arm A Step 50 (Baseline)** | 5.0845 | 5.6707 | 2.9064 | 10.6% |
| **Exp35 Arm A Step 500** | 2.6187 | 4.1297 | 0.0255 | 49.8% |
| **Exp35 Arm B Step 500** | **1.3830** | **2.4352** | **0.0245** | **72.0%** |

### Key Insights:
1. **The Source of Validation Loss Reduction:**  
   In Arm B, validation loss plummeted from 4.9529 to 1.4006. The token breakdown proves that this drop was overwhelmingly driven by:
   - Near-zero EOS loss (**0.0245**, a 99.2% reduction).
   - Massive drop in template token loss (**1.3830**, down from 5.0845).
2. **The Entity Loss Disparity:**  
   Entity token loss remained significantly higher (**2.4352**) than template loss (**1.3830**).
3. **Teacher Forcing vs Free Generation Decoupling:**  
   Under teacher forcing (where the preceding ground-truth response tokens are provided), Arm B correctly predicts the next token **72.0%** of the time. However, in free autoregressive generation, once the model emits a generic template prefix (`"At station 271..."`), it wanders away from the entity span because the attention heads do not attend back to the prompt context.

