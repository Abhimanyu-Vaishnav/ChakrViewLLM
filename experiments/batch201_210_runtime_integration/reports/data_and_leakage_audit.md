# Data & Leakage Audit: Multilingual Relation-Binding Dataset
**Milestone:** Batches 201–210: Batch 203 Specification  
**Dataset Artifact:** `experiments/batch201_210_runtime_integration/data/multilingual_relation_splits.json`  
**Status:** Audited & Leakage-Free  

---

## 1. Domain & Fact Distribution

The relation-binding dataset covers three foundational disciplines with multi-lingual (English and Hindi) representations:

| Domain | Entity Clusters Included | Sample Relations | Split Counts |
|---|---|---|---|
| **Mathematics** | Circle, Sphere, Fibonacci sequence, Euler number e | area formula, circumference, volume, recurrence relation, approx value | Train: 3, Val: 2, Test: 1 |
| **Coding** | QuickSort, Binary Search, Python list, Rust | time complexity, worst-case space, amortized append, memory model | Train: 4, Val: 2, Test: 0 |
| **Science** | Carbon, Copper, सोना (Gold), चांदी (Silver), Jupiter, Saturn | atomic number, melting point, sublimation point, परमाणु क्रमांक, गलनांक, largest moon | Train: 6, Val: 2, Test: 3 |
| **Total** | **14 Clusters** | **23 Structured Relational Facts** | **Train: 13, Val: 6, Test: 4** |

---

## 2. Leakage Protection & Cluster Grouping Audit

- **Partition Policy:** Split assignment occurs strictly at the **entity cluster level** (`cluster_id`), never by random shuffling of individual relational facts.
- **Cluster Overlap Checks:**
  - `Train ∩ Val`: 0 clusters (0.0%).
  - `Train ∩ Test`: 0 clusters (0.0%).
  - `Val ∩ Test`: 0 clusters (0.0%).
- **Evaluation Isolation:** The 4 test facts (covering *Euler number e*, *चांदी / Silver*, and *Saturn's largest moon Titan*) feature entities completely absent from both the training and validation splits.
- **Prompt Overlap Distractor Audit:** Queries feature paraphrased question forms (e.g., *"What is the recurrence relation of Fibonacci sequence?"*) paired with multi-fact context passages containing multiple candidate values, ensuring that naive first-token copying fails.
