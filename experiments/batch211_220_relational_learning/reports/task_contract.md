# Relational Learning & Exact-Value Binding Task Contract
**Milestone:** Batches 211–220: Batch 212 Task Specification  
**Status:** Frozen & Machine-Readable  

---

## 1. Relational Example Schema

Each relational instance is serialized as a JSON record with the following contract:

```json
{
  "example_id": "math_rel_0001",
  "domain": "mathematics",
  "language": "en",
  "entity": "Circle",
  "relation": "area formula",
  "target_value": "pi * r^2",
  "query": "What is the area formula of Circle?",
  "context": "For a standard geometric Circle, the area formula is pi * r^2 and circumference is 2 * pi * r. For a Sphere, volume is 4/3 * pi * r^3.",
  "answerability": "answerable",
  "distractors": [
    {"type": "wrong_relation_same_entity", "value": "2 * pi * r"},
    {"type": "wrong_entity_same_relation", "value": "4/3 * pi * r^3"}
  ],
  "evidence_span": "area formula is pi * r^2",
  "evaluation_slice": "held_out_entity"
}
```

---

## 2. Discrimination Categories

The evaluation task must evaluate models across 6 orthogonal discrimination categories:

1. **Category 1: Correct Entity & Correct Relation (Target Match):**
   - The model must identify the requested entity (`Circle`) and requested relation (`area formula`) and extract precisely `pi * r^2`.
2. **Category 2: Correct Entity, Wrong Relation (Distractor Rejection):**
   - The context provides multiple attributes for the same entity (`circumference: 2 * pi * r`). The model must not copy the wrong attribute.
3. **Category 3: Wrong Entity, Same Relation (Subject Binding):**
   - The context mentions other entities with similar attributes (`Sphere: 4/3 * pi * r^3`). The model must not bind the attribute of a distractor entity.
4. **Category 4: Numerically Plausible, Unsupported Value:**
   - A nearby numerical quantity exists (e.g. `1084 C` vs `1064 C`). The model must not confuse adjacent numeric tokens.
5. **Category 5: Missing Fact (Abstention):**
   - The entity is mentioned, but the specific requested relation is absent from the text. The model must abstain (*"Not mentioned in context"*).
6. **Category 6: Contradictory Evidence (Conflict Rejection):**
   - The context states multiple conflicting values for the same attribute. The system must report conflicting facts.

---

## 3. Evaluation Slices

The frozen dataset must partition records into 6 distinct generalization slices:
1. `seen_entity_seen_relation`: Interpolation baseline.
2. `seen_entity_unseen_relation`: Recombination of known entity with new attribute.
3. `held_out_entity`: Transfer to completely unseen entities.
4. `numeric_distractors`: Multiple competing decimal/integer values in the same context.
5. `multilingual_devanagari`: Hindi queries and contexts with Devanagari script.
6. `unanswerable_or_contradictory`: Missing facts or conflicting evidence requiring abstention.
