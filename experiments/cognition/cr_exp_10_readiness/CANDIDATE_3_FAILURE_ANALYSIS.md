# CANDIDATE 3 FAILURE ANALYSIS & BOTTLENECK TAXONOMY
**Subject:** Candidate 3 Champion Checkpoint (`cr_exp_09_candidate_3_best_val.pt`)  
**SHA-256:** `de871613aa28264014f22e227beea8976c359dca6f7026ed4221703ac7a3e592`  
**Dataset Analyzed:** 295 Missed Probes out of 600 in `cr_exp_09_reproduced_metrics.json`  

---

## 1. PRIMARY FAILURE DOMAINS

### A. Strict Negative Constraint Filtering (16.0% accuracy, 63/75 failures)
- **Observed Behavior:** When given `Constraint: Select the fruit and omit colors. Options: apple, red. Answer:`, the model frequently outputs `"apple, red"` or repeats the non-target token.
- **Root Cause:** In Candidate 3's 3,000-sample training dataset, constraint satisfaction was only 150 samples ($5\%$). The model treats options as simple sequence associations rather than inhibitory filters.
- **Intervention:** Dedicated negative constraint curriculum where the loss penalizes forbidden options with negative sampling ($3.0\times$ token weighting on correct target).

### B. Conditional Logical Reasoning (37.33% accuracy, 47/75 failures)
- **Observed Behavior:** On Modus Ponens questions (`Rule: If X then Y. Context: Not X. Question: Does Y happen?`), the model defaults to `"Yes"` (biased towards affirmation) instead of deducing `"No"`.
- **Root Cause:** Affirmative bias in conversational pretraining.
- **Intervention:** Explicit symmetric negative-context balancing ($50\%$ `"Yes"`, $50\%$ `"No"` counter-examples).

### C. Multi-Entity Distractor Disambiguation (42.67% accuracy, 43/75 failures)
- **Observed Behavior:** In contexts with 3 entities (`"Aarav lives in Delhi. Priya lives in Mumbai. Rohan lives in Jaipur."`), when asked about Rohan, the model occasionally outputs `"Delhi"` (first entity recency/primacy bias).
- **Root Cause:** Attention head saturation on the first sentence position.
- **Intervention:** Positional shuffling where target query entities appear uniformly at the beginning, middle, and end of the context.

### D. Missing-Information Epistemic Abstention (46.67% accuracy, 40/75 failures)
- **Observed Behavior:** Model abstains correctly on absent people, but when asked about unfamiliar attributes for a known person (e.g., `"What is Aarav's job?"`), it attempts to fabricate an answer rather than abstaining.
- **Root Cause:** Training only covered unmentioned subjects; unmentioned predicates were unrepresented.
- **Intervention:** Expand abstention curriculum to attribute/predicate abstention (`"The context does not state Aarav's occupation"`).

### E. Conversational Persona Preservation (77.33% accuracy, 17/75 failures)
- **Observed Behavior:** On open-ended greeting probes, it occasionally outputs a truncated fragment or reverts to generic completion without declaring its identity.
- **Intervention:** Re-anchor core conversational greetings and persona declarations as 20% of the training stream.

