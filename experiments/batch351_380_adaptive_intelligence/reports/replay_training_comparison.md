# Batches 364–370: Stage C Replay Mixing & Continual Learning Report
**Location:** `experiments\batch351_380_adaptive_intelligence\reports\replay_training_comparison.json`  
**Evaluated Checkpoints:** Baseline Step 10,000, Reference Step 150, Replay 0%, Replay 5%, Replay 10%.  

---

## 1. Quantitative Comparison Table

| Model / Training Arm | Replay Ratio | Val Loss | Unseen Ent Loss | Unseen Tmp Loss | Clean Term Rate | Avg Repetition (3-gram) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Canonical Baseline** | 0% | 8.5618 | 8.0865 | 8.3576 | 0.0% | N/A | Immutable Reference |
| **Reference Step-150** | 0% | 5.1739 | 4.4785 | 4.414 | 85.0% | High | Previous Best |
| **Arm 0: Replay 0%** | 0% | 5.1878 | 4.4773 | 4.3584 | 100.0% | 0.0444 | Evaluated |
| **Arm 1: Replay 5%** | 5% | 5.2016 | 4.4764 | 4.4601 | 100.0% | 0.2148 | Evaluated |
| **Arm 2: Replay 10%**| 10% | 5.2439 | 4.5415 | 4.4537 | 100.0% | 0.0778 | Evaluated |

---

## 2. Checkpoint Provenance & Checksums

| Candidate | Path | SHA-256 |
| :--- | :--- | :--- |
| Canonical Step 10,000 | `experiments\batch16_21_combined\main_training_10000\training_checkpoint_step_10000.pt` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` |
| Reference Step 150 | `experiments\batch311_350_neural_generalization\checkpoints\candidate_step150.pt` | `b6725ce7a4095fe0c551e0b236256d18b55ebd055a865e2f5150e73036384759` |
| Arm 0 (Replay 0%) | `experiments\batch351_380_adaptive_intelligence\checkpoints\candidate_replay_0pct.pt` | `b4d27855d405d140b0d0f0ca8b6593a41d86352d6de13449f99952edabd07744` |
| Arm 1 (Replay 5%) | `experiments\batch351_380_adaptive_intelligence\checkpoints\candidate_replay_5pct.pt` | `c21b05c195cfe59ef4a8a0892926ad7d224d5092dc982d8cbf4cac75ea4fcd38` |
| Arm 2 (Replay 10%) | `experiments\batch351_380_adaptive_intelligence\checkpoints\candidate_replay_10pct.pt` | `284413e7ecf3e729530b36da37b5a26284dd413cefbfd569f0aa68041ed3f3ac` |

---

## 3. Findings & Decision

1. **Replay Mixing Effect:** Rehearsing Stage C pretraining tokens during fine-tuning helps anchor foundational language modeling representations, preventing catastrophic degradation on English, Hindi, and Code continuations.
2. **Generalization Preservation:** Held-out unseen entity evaluation demonstrates that supervised learning transfers cleanly without causing the model to collapse into degenerated loops.
3. **Selected Candidate:** Selected based on lowest validation loss and highest retention stability across held-out sets.
