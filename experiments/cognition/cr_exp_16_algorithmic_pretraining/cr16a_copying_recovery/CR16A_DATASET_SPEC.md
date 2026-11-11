# CR-EXP-16A Dataset Specification

## 1. Dataset Generation Methodology
All training datasets were synthetically generated using pure Python scripts (`generate_cr16a_data.py`, `train_arm_c.py`, `train_arm_d.py`) without external models or web queries.

## 2. Vocabulary Pools
1. **Greek Letters (19 tokens):**
   `alpha`, `beta`, `delta`, `epsilon`, `eta`, `gamma`, `iota`, `kappa`, `lambda`, `mu`, `nu`, `omicron`, `pi`, `rho`, `sigma`, `tau`, `theta`, `xi`, `zeta`.
2. **Held-Out Generalization Names (150 names):**
   Disjoint Indian and Western names (e.g., `Aarav`, `Aryan`, `Zara`, `John`, `David`, `Sophia`). Strictly excluded from all 23 benchmark names.
3. **Common Nouns & Basic Lexicon (200 words):**
   Fruits, colors, numbers, animals, natural elements, household objects.

## 3. Formatting Templates
- Template 1: `Echo sequence: {seq}\nOutput: {seq}\n`
- Template 2: `Repeat the name: {seq}\nName: {seq}\n`
- Template 3: `Repeat: {seq}\nOutput: {seq}\n`
- Template 4: `Copy: {seq} -> {seq}\n`
- Template 5: `Sequence: {seq} -> {seq}\n`
- Template 6: `Input: {seq}\nCopy: {seq}\n`

## 4. Dataset Splits
- `cr16a_curriculum_train.json`: 3,000 samples (Arm C)
- `cr16a_curriculum_val.json`: 300 samples (disjoint combinations)
- `cr16a_arm_d_train.json`: 5,000 samples (Arm D)

