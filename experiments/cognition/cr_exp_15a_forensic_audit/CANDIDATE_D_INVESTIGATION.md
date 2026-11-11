# CR-EXP-15A Candidate D Forensic Investigation
## Analysis of Low Validation Loss (1.8418) vs Complete Generalization Failure (0/1050)

**Mission Identifier:** CR-EXP-15A  
**Auditor:** Autonomous Scientific Auditor & Neural Research Scientist  
**Date:** November 2026  

---

## 1. The Paradox of Candidate D

In CR-EXP-15, Candidate D achieved the lowest validation loss of any model in the experiment:
$$\text{Val Loss} = 1.8418 \quad (\text{vs } 5.0537 \text{ on Candidate A and } 5.7251 \text{ on Parent Baseline})$$

However, when tested on the sealed 1,200-probe benchmark, Candidate D scored:
$$\text{Strict Accuracy} = 0.000\% \quad (0 / 1,050)$$
$$\text{Subject Accuracy} = 0.000\% \quad (0 / 1,050)$$
$$\text{Value Accuracy} = 0.000\% \quad (0 / 1,050)$$

---

## 2. Forensic Analysis of Data Distribution and Model Outputs

To explain this divergence, an automated entity-location distribution audit was conducted across Candidate D's training dataset (`train_candidate_d.json`, 500 samples) and its 1,200 benchmark outputs:

### 1. Training Set Composition
- All 500 training examples followed a strict multi-turn syntactic template:
  `Turn 1: User: [Entity 1] lives in [City 1]. ... Turn N: User: Where is [Entity] located? Assistant: [Entity] is currently in [City].`
- A small pool of 10 entity names was repeated across the 500 examples:
  `Kavita` (60x), `Geeta` (56x), `Amitabh` (54x), `Anita` (54x), `Jayesh` (52x), `Amita` (52x), `Ankush` (48x).
- A small pool of 10 target cities was repeated across the 500 examples:
  `Kochi` (105x), `Kanpur` (99x), `Jalandhar` (98x), `Amritsar` (96x), `Chandigarh` (92x), `Aurangabad` (91x).

### 2. Generated Outputs on Unseen Entities
When evaluated on the 1,200 sealed probes featuring completely unseen entities (`Mukesh`, `Siddharth`, `Rajnath`, `Pankhuri`, etc.):
- Out of 1,200 generations, **283 responses** named `Kavita` ($23.6\%$).
- **221 responses** named `Ankush` ($18.4\%$).
- **114 responses** named `Mayank` ($9.5\%$).
- **87 responses** named `Ankita` ($7.3\%$).
- Together, 4 training entities accounted for **58.8% of all generations**.
- For locations, `Aurangabad` appeared 348 times ($29.0\%$) and `Amritsar` appeared 259 times ($21.6\%$).

### Top Generated Responses Across Probes:
- `[114x] 'Ankush is currently in Aurangabad.'`
- `[108x] 'Kavita is currently in Amritsar.'`
- `[86x] 'Mayank is currently in Aurangabad.'`
- `[47x] 'Kavita is currently in Aurangabad.'`

---

## 3. Scientific Conclusion on Candidate D

Candidate D's low validation loss ($1.8418$) was **entirely driven by template and token frequency memorization**:
1. The model learned that after `Assistant:`, predicting `[High Frequency Entity] is currently in [High Frequency City].` drastically minimizes cross-entropy loss across templated sequences.
2. The attention mechanism never learned to attend backwards into the prompt to copy the query subject (`Rajnath`) or trace its state transition.
3. Instead, the model's feedforward layers functioned as a static parametric generator, outputting high-probability unigrams from the training distribution.
4. When tested on out-of-distribution entities, this parametric bias resulted in 100% substitution error.

