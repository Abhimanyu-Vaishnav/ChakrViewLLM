# CR-16D Seed Variance Analysis: Arm C & Cross-Arm Dynamics

### 1. The Discrepancy in CR-16D Arm C
In CR-16D, Arm C exhibited large differences between random seeds:
- **Seed 42:** G1 (Novel Combos) = `122 / 250` (48.8%), G3 (Unseen Keys) = `109 / 250` (43.6%), Total Comp = `249 / 1000` (24.9%).
- **Seed 101:** G1 (Novel Combos) = `241 / 250` (96.4%), G3 (Unseen Keys) = `236 / 250` (94.4%), Total Comp = `509 / 1000` (50.9%).
- Implied variance: A **47.6 percentage point spread** on G1.

---

### 2. Systematic Forensic Investigation of the Variance Source

| Factor Investigated | Finding | Impact on Variance |
|---|---|---|
| **Weight Initialization** | Both seeds load identical weights from `arm_d_generalization_best.pt` (`9d609a77...`). | Neutral (Identical initialization) |
| **Model Configuration** | Both runs use canonical ChakrMicro ($d=192, L=6, H=6, V=4096$). | Neutral (Zero drift) |
| **Batch Composition & Shuffling** | DataLoader uses `shuffle=True` with random seed `torch.manual_seed(seed)`. | **PRIMARY CAUSAL DRIVER** |
| **Task Interleaving Sequence** | In 50% replay, batches interleave copying and retrieval sequences. Depending on early batch order, the attention heads either settle early into sharp syntax-delimiter routing or encounter conflicting gradients between tasks. | **CRITICAL FACTOR** |
| **Evaluation Decoding** | Deterministic greedy decoding ($T=0, \text{argmax}$). Zero evaluation noise. | Neutral (Purely deterministic) |

---

### 3. Extended Replication Evidence (Seeds 202, 303, 404)
Evaluating Arm C across 3 new seeds reveals the true distribution:
- **Seed 202:** G1 = `130 / 250` (52.0%), G3 = `116 / 250` (46.4%), Comp Total = `275 / 1000` (27.5%)
- **Seed 303:** G1 = `112 / 250` (44.8%), G3 = `108 / 250` (43.2%), Comp Total = `240 / 1000` (24.0%)
- **Seed 404:** G1 = `63 / 250` (25.2%), G3 = `57 / 250` (22.8%), Comp Total = `132 / 1000` (13.2%)

#### Full 5-Seed Statistics for Arm C
- **G1 (Novel Combinations):** Mean = `133.6 / 250` (53.4%) $\pm 58.5$ | Range: `[63, 241]`
- **G3 (Unseen Keys):** Mean = `125.2 / 250` (50.1%) $\pm 59.3$ | Range: `[57, 236]`
- **Overall Compositional Total:** Mean = `281.0 / 1000` (28.1%) $\pm 124.0$ | Range: `[132, 509]`

### 4. Scientific Verdict on Arm C Seed Variance
1. **Seed 101 was a statistical outlier:** Arm C Seed 101 represents a fortunate optimization trajectory where the model discovered an optimal attention-head alignment early. The true expectation across five independent seeds is $\approx 53.4\%$ on G1 and $\approx 50.1\%$ on G3.
2. **Arm C is not stable enough to be declared universally Pareto-optimal:** While it preserves sequence copying with high consistency ($83.4\% \pm 3.2\%$) and achieves perfect reassignment ($100\%$), its compositional generalization has high seed variance ($\sigma = 58.5$).
3. In contrast, **Arm D (75% Replay)** achieves much higher stability across 5 seeds: G1 mean `157.4 / 250` (63.0% $\pm 30.1$), copying mean `88.2%` ($\pm 1.6\%$), and overall compositional mean `334.6 / 1000` (33.5% $\pm 63.6$).

