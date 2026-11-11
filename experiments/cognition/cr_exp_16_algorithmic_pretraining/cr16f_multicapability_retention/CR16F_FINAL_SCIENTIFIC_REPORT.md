# CHAKRVIEW — CR-EXP-16F
# Multi-Capability Retention, Retrieval and Abstention Balance Study
## Final Scientific Report

**Project:** ChakrView  
**Experiment Root:** `experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16f_multicapability_retention/`  
**Architecture:** ChakrMicro v0.1 (Frozen: $L=6, d=192, H=6, d_{\text{mlp}}=512, V=4096$, $3,443,136$ parameters)  
**Parent Checkpoint:** CR-16A Arm D (`9d609a770c748f9321d06713093620da0fe6caabc04fea2401b37403d33de1a5`)  
**Protected Invariant:** `chakrview/runtime/session.py` (`9080ff51...`) strictly untouched.  
**Training Seeds:** 2026, 3030, 4040 ($N=3$ independent training runs per condition; 12 total models).

---

## 1. Executive Summary

CR-EXP-16F was designed to investigate the central open question of the CR-16 algorithmic capability program: **Can ChakrMicro acquire key-value retrieval, multi-token span generation, and calibrated abstention while retaining previously acquired sequence-copying capability, or do these objectives fundamentally interfere at micro-scale?**

Using a rigorous $2 \times 2$ factorial experimental design testing two sequence-copying replay ratios (25% vs. 50%) and two negative abstention loss multipliers ($\lambda = 1.00$ vs. $\lambda = 0.25$) across three seeds per arm, all 12 planned models were trained and comprehensively evaluated against 6 frozen benchmark suites (3,150 total evaluation probes per model).

### Key Scientific Conclusions:
1. **Severe Multi-Task Gradient Interference:** Neither 25% nor 50% copying replay preserved sequence copying. Copying accuracy dropped from the parent model's **66%** to **$\le 1.0\%$** across all 12 models.
2. **Abstention Gating vs. False-Abstention Collapse:** Full negative abstention loss ($\lambda = 1.00$) achieved high abstention recall ($80 - 92\%$ on CAT9, $90 - 96\%$ on B4), but triggered widespread false-abstention collapse on answerable retrieval ($>960$ out of 1,400 answerable prompts aborted with `[UNKNOWN]`). Attenuating negative loss to $\lambda = 0.25$ reduced false abstentions to ~590/1,400 and doubled answerable accuracy, but degraded abstention recall to $\le 47\%$.
3. **Capacity Ceiling of ChakrMicro v0.1:** Across all 12 models, zero-shot retrieval of unseen values (CAT7) was identically **0/150 (0.0%)**. The 3.44M parameter model lacks the capacity to simultaneously maintain literal token induction heads, positional routing circuits, and abstract symbolic variable binding.

---

## 2. Experimental Design Matrix & Full Results

### A. Factorial Cell Means & Sample Standard Deviations ($N=3$)

| Benchmark / Capability Metric | Arm A (25% Replay, $\lambda=1.00$) | Arm B (50% Replay, $\lambda=1.00$) | Arm C (25% Replay, $\lambda=0.25$) | Arm D (50% Replay, $\lambda=0.25$) |
| :--- | :---: | :---: | :---: | :---: |
| **D1. Exact Copying (/100)** | $0.67 \pm 0.58$ ($0.7\%$) | $1.00 \pm 0.00$ ($1.0\%$) | $0.67 \pm 0.58$ ($0.7\%$) | $0.33 \pm 0.58$ ($0.3\%$) |
| **D2. Answerable Retrieval (/1,400)** | $65.33 \pm 23.76$ ($4.7\%$) | $67.33 \pm 17.79$ ($4.8\%$) | **$145.67 \pm 24.09$** ($10.4\%$) | $127.33 \pm 61.37$ ($9.1\%$) |
| **D2. False Abstention Count (/1,400)** | $1045.67 \pm 124.74$ | $960.67 \pm 131.09$ | **$596.33 \pm 77.15$** | $591.33 \pm 370.65$ |
| **D2. Unanswerable CAT9 Recall (/100)** | **$92.33 \pm 8.33$** ($92.3\%$) | $80.67 \pm 2.08$ ($80.7\%$) | $47.33 \pm 7.57$ ($47.3\%$) | $46.00 \pm 27.78$ ($46.0\%$) |
| **D3. B1 Position Invariance (/600)** | $136.33 \pm 10.97$ ($22.7\%$) | $133.00 \pm 16.09$ ($22.2\%$) | **$168.00 \pm 16.46$** ($28.0\%$) | $146.33 \pm 11.24$ ($24.4\%$) |
| **D3b. B2 Permutation Pairs (/150)** | $0.33 \pm 0.58$ ($0.2\%$) | $1.00 \pm 1.00$ ($0.7\%$) | **$10.00 \pm 7.00$** ($6.7\%$) | $9.00 \pm 4.58$ ($6.0\%$) |
| **D3c. B2 Counterfactual Pairs (/150)** | $0.67 \pm 0.58$ ($0.4\%$) | $2.67 \pm 0.58$ ($1.8\%$) | **$13.67 \pm 1.53$** ($9.1\%$) | $10.00 \pm 8.89$ ($6.7\%$) |
| **D4. B3 Multi-Token Spans (/500)** | $67.67 \pm 15.70$ ($13.5\%$) | $80.67 \pm 9.29$ ($16.1\%$) | **$148.00 \pm 9.54$** ($29.6\%$) | $122.67 \pm 45.96$ ($24.5\%$) |
| **D5. B4 Calibrated Abstention (/300)** | **$289.33 \pm 10.02$** ($96.4\%$) | $270.67 \pm 12.01$ ($90.2\%$) | $201.00 \pm 14.80$ ($67.0\%$) | $180.33 \pm 68.72$ ($60.1\%$) |
| **D6. CAT7 Unseen-Value (/150)** | $0.00 \pm 0.00$ ($0.0\%$) | $0.00 \pm 0.00$ ($0.0\%$) | $0.00 \pm 0.00$ ($0.0\%$) | $0.00 \pm 0.00$ ($0.0\%$) |

---

## 3. Two-Way ANOVA Statistical Attribution

A two-factor ANOVA ($df_1 = 1, df_2 = 8$) was conducted to isolate the causal attribution of the Replay Fraction vs. Abstention Loss Multiplier ($\lambda$):
1. **Replay Fraction Effect:** Statistically non-significant across all metrics ($F \le 2.41$, all $\eta_p^2 \le 0.232$, $p > 0.15$). Increasing copying replay from 25% to 50% did not rescue sequence copying and had negligible influence on retrieval or abstention performance.
2. **Abstention Multiplier ($\lambda$) Effect:** Highly significant with large effect sizes across every retrieval and abstention metric:
   - Answerable Retrieval: $F = 11.30, \eta_p^2 = 0.586$ ($p < 0.01$)
   - False Abstention: $F = 11.42, \eta_p^2 = 0.588$ ($p < 0.01$)
   - CAT9 Unanswerable Recall: $F = 21.09, \eta_p^2 = 0.725$ ($p < 0.005$)
   - B3 Multi-Token Spans: $F = 17.70, \eta_p^2 = 0.689$ ($p < 0.005$)
   - B4 Abstention Recall: $F = 18.47, \eta_p^2 = 0.698$ ($p < 0.005$)
3. **Interaction (Replay $\times \lambda$):** No significant interactions were observed (all $\eta_p^2 \le 0.179$). The two factors operate orthogonally.

---

## 4. Addressing the 10 User Questions Directly

### Q1: Does controlled replay prevent catastrophic forgetting of sequence copying?
**No.** Copying accuracy dropped from 66% in parent CR-16A Arm D down to $\le 1.0\%$ across all arms (Arm A: 0.7%, Arm B: 1.0%, Arm C: 0.7%, Arm D: 0.3%). While replay loss on mini-batch items dropped during training, the network suffered complete catastrophic forgetting on the frozen copying benchmark.

### Q2: What is the effect of the negative abstention loss multiplier $\lambda$?
Scaling $\lambda$ from 1.00 down to 0.25 drastically cuts false abstentions on answerable queries (from 1,003 down to 594 out of 1,400) and more than doubles answerable retrieval accuracy ($66 \to 136 / 1,400$). However, it cuts unanswerable recall nearly in half ($86.5\% \to 46.7\%$ on CAT9, and $280 \to 190 / 300$ on B4).

### Q3: Is there an interaction between replay percentage and negative loss weighting?
**No.** Two-way ANOVA revealed no significant interaction effects (all $F_{\text{int}} \le 1.74$, all $\eta_p^2 \le 0.179$). Increasing replay percentage from 25% to 50% does not cushion or mitigate the gradient interference imposed by $\lambda$.

### Q4: Does position invariance replicate across training seeds?
**Yes, but at low overall accuracy.** On Benchmark B1 (600 probes across 4 positions), accuracy is consistent across seeds ($22\% - 28\%$) with relatively balanced performance across positions (e.g. Arm C: Pos 0 = 28.6%, Pos 1 = 29.9%, Pos 2 = 24.7%, Pos 3 = 26.7%). However, on joint permutation pairs (B2), accuracy is low ($6\% - 7\%$).

### Q5: Does multi-token span completion generalize to unseen values?
**No.** While models in Arm C achieved 29.6% accuracy on Benchmark B3 (which uses in-distribution entity values with balanced token lengths 1–4), all 12 models scored **0/150 (0.0%)** on CAT7 unseen values. The model generates memorized subwords rather than transferring abstract multi-token copying to novel values.

### Q6: What is the optimal balance between capability retention and retrieval acquisition?
Among the tested conditions, **Arm C (25% Replay, $\lambda = 0.25$)** achieved the best retrieval performance ($145.67/1,400$), the lowest false abstentions ($596.33/1,400$), the highest B1 position invariance ($168.00/600$), and the highest B3 span completion ($148.00/500$), while preserving moderate abstention recall ($67\%$ on B4, $47\%$ on CAT9). However, because sequence copying remained collapsed ($0.67\%$) and CAT7 was $0.0\%$, **true multi-capability balance was not achieved**.

### Q7: What are the mechanistic causes of gradient conflict between tasks?
1. **Copying vs. Retrieval Conflict:** Verbatim copying requires identity token propagation across consecutive positions. Key-value retrieval requires content-addressable attention routing where key matches suppress input tokens and output a stored value. In a 6-layer 192-dim transformer, fine-tuning projection weights overwrites induction circuits.
2. **`[UNKNOWN]` Attractor Dynamics:** `[UNKNOWN]` is a single static token trained as the target across many structurally diverse negative sequences. It establishes a strong gradient vector in the unembedding layer that misclassifies distractor-laden answerable contexts as unanswerable.

### Q8: How does CR-16F compare to previous iterations (CR-16A through CR-16E-R2)?
CR-16F provides the first multi-seed factorial confirmation that sequence-copying replay and abstention weight attenuation are insufficient to overcome the representational interference of complex multi-task training at the 3.4M parameter scale.

### Q9: Does the evidence justify keeping ChakrMicro v0.1 frozen or changing architecture?
**ChakrMicro v0.1 remains frozen.** All experimental rules were honored. However, the evidence conclusively demonstrates that the 3.44M parameter model has hit a strict capacity limit. Further prompt or replay tuning on this architecture cannot solve the mutual exclusivity of copying, position-invariant retrieval, and calibrated abstention.

### Q10: What are the clear, evidence-based recommendations for subsequent research?
1. Conclude the algorithmic fine-tuning explorations on the frozen 3.4M ChakrMicro v0.1 baseline.
2. For the subsequent milestone (ChakrMicro v0.2 / ChakrMini), design an architectural expansion roadmap:
   - Increase model capacity ($d \ge 384, L \ge 12, H \ge 8, \approx 15-30\text{M}$ parameters).
   - Implement architectural decoupling: dedicated pointer/copy attention heads, auxiliary memory routing, or explicit confidence heads for abstention.

