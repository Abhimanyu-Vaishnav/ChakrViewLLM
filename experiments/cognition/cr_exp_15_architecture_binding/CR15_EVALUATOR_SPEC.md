# CR-EXP-15 Phase 3: Evaluator Specification

## 1. Metric Definitions & Scored Phenomena

The CR-EXP-15 evaluator enforces strict mathematical separation between answerable capability metrics and epistemic abstention.

### 1. Task A: Entity Copying Accuracy ($N=150$)
- A hit requires: `subject.lower() in generated.lower()`.
- Denominator: Exactly 150 probes.

### 2. Task B: Entity Selection Accuracy ($N=150$)
- A hit requires: `subject.lower() in generated.lower()` AND `all(d.lower() not in generated.lower() for d in distractors)`.
- Denominator: Exactly 150 probes.

### 3. Task C: Static Subject-Value Binding ($N=150$)
- A hit requires: `subject.lower() in generated.lower()` AND `expected_target.lower() in generated.lower()` AND `all(d.lower() not in generated.lower() for d in distractors)`.
- Denominator: Exactly 150 probes.

### 4. Tasks D, E, F, G: Dynamic State Tracking ($N=600$ total, 150 each)
- A hit requires:
  1. `subject.lower() in generated.lower()` (Subject Identification)
  2. `expected_target.lower() in generated.lower()` (Target Value Match)
  3. `earlier_value.lower() not in generated.lower()` (Stale Value Exclusion)
  4. `all(d.lower() not in generated.lower() for d in distractors)` (Distractor Exclusion)

### 5. Task H: Epistemic Abstention Calibration ($N=150$)
- Scored strictly against calibrated abstention cues (`"does not"`, `"not specify"`, `"unknown"`, `"unspecified"`, `"cannot"`, `"no information"`, `"not state"`).
- **Abstention Recall:** $\frac{\text{Correct Abstentions on Task H}}{150}$.
- **Abstention Precision:** $\frac{\text{Correct Abstentions on Task H}}{\text{Total Abstentions across All 1,200 Probes}}$.
- **Abstention F1-Score:** Harmonic mean of Precision and Recall.

---

## 2. Prohibition of Cross-Denominator Mixing
Under no circumstances may Task H (Abstention) be added to the numerators of Tasks A–G to inflate working memory scores.
All report tables must display independent percentages for each atomic task.

