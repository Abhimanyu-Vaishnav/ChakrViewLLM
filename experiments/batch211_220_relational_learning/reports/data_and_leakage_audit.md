# Data & Leakage Audit: 10,000-Record Relational Corpus
**Milestone:** Batches 211–220: Batch 214 Specification  
**Artifact:** `experiments/batch211_220_relational_learning/data/relational_corpus_manifest.json`  
**Status:** Frozen & Leakage-Free  

---

## 1. Corpus Summary

- **Total Generated Relational Records:** 10,000 records.
- **Domains Covered:**
  - **Mathematics:** Circle, Square, Triangle, Sphere, Cube, Cylinder, Golden Ratio phi, Euler number e, Archimedes constant pi, Fibonacci sequence.
  - **Coding:** QuickSort, MergeSort, BubbleSort, Binary Search, Linear Search, Hash Table, Binary Search Tree, Python list, Rust, Go.
  - **Science:** Hydrogen, Helium, Lithium, Carbon, Nitrogen, Oxygen, Copper, Gold, Jupiter, Saturn, सोना (Gold), चांदी (Silver), तांबा (Copper), लोहा (Iron).
- **Split Distribution:**
  - **Train:** 6,360 records (63.6%).
  - **Validation:** 1,097 records (11.0%).
  - **Held-Out Test:** 2,543 records (25.4%).

---

## 2. Evaluation Slices

The corpus features 4 explicitly labeled evaluation slices across domains:
1. `seen_entity_seen_relation`: In-distribution factual retrieval and extraction.
2. `numeric_distractors`: Multiple competing numeric quantities (constants, temperatures, dimensions, complexities) in the same passage.
3. `held_out_entity`: 9 entity clusters completely held out from training:
   * Mathematics: *Golden Ratio phi*, *Euler number e*
   * Coding: *Hash Table*, *Rust*
   * Science: *Copper*, *Gold*, *Saturn*, *चांदी (Silver)*, *तांबा (Copper)*
4. `multilingual_devanagari`: Native Hindi queries and context sentences with Devanagari script.

---

## 3. Leakage Protection Verification

- **Cluster-Level Partitioning:** Every held-out entity cluster was filtered strictly into the test set prior to train/val splitting.
- **Audit Verification Metric:**
  - Number of held-out entity mentions in `relational_corpus_train.json`: **0 (Zero)**.
  - Leakage risk: **0.0%**.
- **Frozen Contract:** All test and validation manifests are stored as immutable JSON records in `data/` and must not be modified during or after training.
