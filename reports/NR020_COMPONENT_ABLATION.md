# NR-020: Component & Oracle Ablation Study

## 1. Experimental Methodology
To isolate the exact performance contributions of the neural router vs the deterministic source-grounding engine, three controlled oracle ablation conditions were evaluated on all 800 benchmark tasks:
- **Oracle 3 (Gold Operation + Gold Arguments)**: Dispatches reference operations and reference arguments directly to the restricted execution engine. Measures runtime engine correctness and reference label integrity.
- **Oracle 1 (Gold Operation + Predicted Grounded Arguments)**: Dispatches the reference operation, but extracts all arguments strictly via the deterministic `SourceGroundingEngine` and `MultilingualNormalizer`.
- **System D Actual (Predicted Operation + Predicted Grounded Arguments)**: The full end-to-end system where the 17M neural router classifies the operation and the grounding engine extracts parameters.

---

## 2. Quantitative Oracle Ablation Results (800 Tasks)

| Experimental Condition | Operation Origin | Argument Origin | Strict Accuracy | Gap / Error Rate |
| :--- | :---: | :---: | :---: | :---: |
| **Oracle 3: Pure Executor / Reference** | Reference | Reference | **89.38%** (715/800) | $10.62\%$ (State ops missing schema) |
| **Oracle 1: Grounding Upper Bound** | Reference | Grounding Engine | **88.75%** (710/800) | $\mathbf{0.63\%}$ grounding gap |
| **System D: Full Neuro-Symbolic** | Neural Router | Grounding Engine | **50.50%** (404/800) | $\mathbf{38.25\%}$ neural routing gap |

---

## 3. Groundbreaking Ablation Findings
1. **The Grounding Engine is Near-Perfect**:
   - Comparing Oracle 3 (89.38%) to Oracle 1 (88.75%) reveals an argument extraction fidelity loss of **only 0.63% (5 tasks out of 800)**!
   - When the operation is known, the source-grounding engine successfully extracts and normalizes arguments in **99.3%** of all viable cases across English, Hindi, and Hinglish.
2. **The Neural Router is the Real Bottleneck**:
   - Comparing Oracle 1 (88.75%) to System D (50.50%) exposes a **38.25% accuracy loss (306 tasks)** caused purely by neural operation misclassification or truncation.
   - The primary limiting factor of the ChakrView hybrid system is NOT argument extraction; it is the 17M neural model's ability to classify multilingual intent reliably.
