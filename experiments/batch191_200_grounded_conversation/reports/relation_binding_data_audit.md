# Relation-Binding Dataset Audit: Batch 193
## Entity Isolation, Relation Diversity & Zero-Leakage Verification

### 1. Dataset Provenance & Split Integrity
- **Dataset Name:** `ChakrView_Relation_Binding_Benchmark_v1`
- **Location:** `experiments/batch191_200_grounded_conversation/data/split_manifest.json`
- **Split Partitioning:**
  - `train`: 12 examples (Elements Hydrogen, Helium, Lithium; Planets Mars, Jupiter, Earth, Venus)
  - `val`: 4 examples (Elements Iron, Copper)
  - `test`: 6 examples (Elements Boron, Gold; Planets Neptune, Titan; Multilingual Devanagari चांदी)
- **Entity Disjointness Invariant:** $\text{Entities}(\text{train}) \cap \text{Entities}(\text{test}) = \emptyset$.
- **Relation Diversity:** Tests distinguishing multiple attributes per entity (e.g., `atomic number` vs `atomic mass` vs `boiling point` vs `density`).

### 2. Cryptographic Checksums
- `train_sha256`: Verified in `split_manifest.json`
- `val_sha256`: Verified in `split_manifest.json`
- `test_sha256`: Verified in `split_manifest.json`
