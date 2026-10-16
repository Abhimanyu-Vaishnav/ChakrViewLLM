# INTENT CLASSIFICATION RESULTS
## Batches 1701–1740 Detailed Class-by-Class Analysis for Direct Neural Intent Heads

### 1. Per-Class Precision, Recall & F1 Breakdown

Evaluated on 180 frozen held-out test tasks (30 tasks per intent):

```
Intent Class Performance (Variant B: Fine-Tuned Backbone + Linear Head)
Overall Accuracy: 77.78% (140/180) | Macro-F1: 0.7772
  ├── CODE_TASK:       Precision 85.7% | Recall 100.0% | F1: 0.9231 (Top Performer)
  ├── CLARIFY:         Precision 96.0% | Recall  80.0% | F1: 0.8727 (High Specificity)
  ├── KNOWLEDGE_QUERY: Precision 81.5% | Recall  73.3% | F1: 0.7719
  ├── CALCULATE:       Precision 71.0% | Recall  73.3% | F1: 0.7213
  ├── OUT_OF_SCOPE:    Precision 65.7% | Recall  76.7% | F1: 0.7077
  └── VERIFY_CLAIM:    Precision 70.4% | Recall  63.3% | F1: 0.6667
```

---

### 2. Comparative Per-Class Matrix (Variant A vs. Variant B vs. Baselines)

| Intent Class | Keyword Baseline F1 | Governed Deterministic F1 | Variant A (Frozen) F1 | Variant B (Fine-Tuned 100%) F1 |
| :--- | :---: | :---: | :---: | :---: |
| **`CODE_TASK`** | 0.7619 | 0.5238 | **0.9375** | 0.9231 |
| **`CLARIFY`** | 0.0000 | 0.3109 | 0.7246 | **0.8727** |
| **`KNOWLEDGE_QUERY`** | 0.6207 | 0.0000 | 0.6286 | **0.7719** |
| **`CALCULATE`** | 0.6207 | 0.1818 | 0.6786 | **0.7213** |
| **`OUT_OF_SCOPE`** | 0.3736 | 0.0000 | 0.3600 | **0.7077** |
| **`VERIFY_CLAIM`** | 0.6667 | 0.1250 | 0.5098 | **0.6667** |
| **Macro-F1 Average** | **0.5073** | **0.1903** | **0.6398** | **0.7772** |

---

### 3. Detailed Behavioral Analysis by Class

#### 3.1 `CODE_TASK` (Recall: 100.0%, Precision: 85.7%)
The model exhibited near-flawless recognition of programming syntax, code keywords (`print`, `def`, `import`, brackets), and sandbox execution requests. All 30 held-out code tasks were correctly classified.

#### 3.2 `CLARIFY` (Precision: 96.0%, Recall: 80.0%)
Crucially, the direct classification head learned to differentiate complete requests from incomplete requests. When presented with `"Calculate it for me please"` or `"Can you verify this?"`, it routed 24/30 queries to `CLARIFY` with 96.0% precision, avoiding dangerous false execution on underspecified queries.

#### 3.3 `CALCULATE` (Precision: 71.0%, Recall: 73.3%)
Correctly classified complex conversational arithmetic queries where regexes failed, such as `"Could you work out what 96 divided by 8 is?"`, `"Find the quotient of 144 over 12"`, and `"Add 140 and 260 together"`.

#### 3.4 `OUT_OF_SCOPE` & Negative Controls (Precision: 65.7%, Recall: 76.7%)
Successfully rejected conversational tool mentions (e.g. `"The calculator is a useful deterministic tool"`) and malicious injections (`"; DROP TABLE"`), recognizing that no execution was requested.
