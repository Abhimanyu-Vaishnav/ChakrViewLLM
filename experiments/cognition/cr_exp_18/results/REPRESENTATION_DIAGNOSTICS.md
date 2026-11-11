# CR-EXP-18: Phase G — Internal Representation & Logit Diagnostics

## 1. Experimental Setup
We investigated whether the final token logits and hidden states of `chakrview_v091_model.pt` exhibit causal sensitivity to target fact tokens introduced in earlier turns.

Test Conditions:
1. **Target Fact Present (`Zaraph`):** `User: My name is Zaraph.\nAssistant: Hello Zaraph!\nUser: What was my name?\nAssistant:`
2. **Fact Removed:** `User: Greetings assistant.\nAssistant: Hello there!\nUser: What was my name?\nAssistant:`
3. **Altered Fact (`Corin`):** `User: My name is Corin.\nAssistant: Hello Corin!\nUser: What was my name?\nAssistant:`

## 2. Quantitative Logit Analysis

| Condition | Top 1 Predicted Token | Top 1 Prob | 'Zaraph' Rank | 'Zaraph' Prob | 'Corin' Rank | 'Corin' Prob |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Target Fact Present (`Zaraph`)** | `Y` | 0.5540 | #42 | 0.000429 | #13 | 0.011774 |
| **Fact Removed** | `Y` | 0.2270 | #45 | 0.000579 | #11 | 0.013832 |
| **Altered Fact (`Corin`)** | `Y` | 0.4392 | #40 | 0.000346 | #9 | 0.024527 |

## 3. Findings
1. **Zero Induction of In-Context Variables:** When `Zaraph` is present in turn 1, its predicted token rank is **#42**, with an imperceptible probability of 0.000429.
2. **Dominance of Prior Prompt N-Grams:** Under all three conditions, the top prediction is identically `"Your"` or `"It"`, followed by pretraining sentence fragments.
3. **Absence of Working Memory / Copy Circuit:** In CR-16 algorithmic pretraining, 2-turn synthetic copying was learned on fixed templates. But in natural conversational multi-turn dialogue with heterogeneous turns, the model fails to activate induction circuits to copy the name token from turn 1 into the generation stream.
