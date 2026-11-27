# NR-020: Out-of-Distribution (OOD) & Dataset Integrity Audit

## 1. Investigation of the OOD Anomaly
In NR-019, System D achieved **57.80% accuracy on OOD tasks** compared to **38.33% on In-Distribution tasks**.

A detailed operational frequency audit revealed why this occurred:
1. **Compositional Skew in OOD Set**:
   - The OOD set was heavily loaded with three specific task types:
     - `unsupported` adversarial tasks: 100 tasks (20% of OOD set) $\to$ System D scored 63.0%.
     - `conditional_branch`: 69 tasks (13.8% of OOD set) $\to$ System D scored **100.0%**.
     - `state_accumulator`: 70 tasks (14.0% of OOD set) $\to$ System D scored **97.1%**.
   - These three categories alone account for **239 out of 500 OOD tasks** (47.8% of the OOD set), on which System D achieved an average accuracy of **83.7%**!
2. **In-Distribution Operation Distribution**:
   - The In-Distribution set had only 15 `conditional_branch` tasks (5.0%) and 23 `state_accumulator` tasks (7.7%).
   - Meanwhile, In-Distribution had high concentrations of operations where the grounding engine failed due to missing return schemas (`var_set` 21 tasks: 0%, `var_get` 11 tasks: 0%, `list_sum` 23 tasks: 0%, `string_lower` 13 tasks: 0%, `string_length` 17 tasks: 0%). These 5 failed categories constituted **28.3% of the In-Distribution set**!

---

## 2. Forensic Conclusion
The higher OOD score was **NOT** an indication that the neural router generalizes better on harder numerical distributions. It was an artifact of **task-family distributional skew**: the OOD set contained disproportionately more tasks from operation families where the neural router and grounding engine had near-100% accuracy (`conditional_branch`, `state_accumulator`, `unsupported`).

**Action for Phase 6**:
Construct a fresh matched benchmark where ID and OOD splits have **strictly identical operation family distributions and difficulty curves**.
