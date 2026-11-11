# CR-EXP-16B Objective Factorial Design

## 1. Experimental Motivation & Scientific Question
CR-EXP-16A demonstrated significant sequence copying improvements, but combined target-only loss masking with template expansion and dataset growth, leaving the causal attribution open to question.

CR-EXP-16B isolated these variables via a strict $2\times 2$ Factorial Design:
- **Factor 1: Loss Objective**
  - Level 1: Full-sequence causal LM loss ($\mathcal{L}_{\text{full}}$)
  - Level 2: Target-only masked loss ($\mathcal{L}_{\text{target}}$, with prompt tokens masked to `-100`)
- **Factor 2: Template Diversity**
  - Level 1: Single-template (`Echo sequence: {seq}\nOutput: {seq}\n`)
  - Level 2: Multi-template (6 diverse templates)

## 2. The Four Primary Experimental Arms
1. **Arm A (Full Loss + Single Template):** Direct replication of the Stage A pilot mechanism under controlled data.
2. **Arm B (Target-Only Loss + Single Template):** Isolates the exact contribution of loss masking while holding templates constant.
3. **Arm C (Full Loss + Multi-Template):** Isolates the exact contribution of template diversity under full-sequence loss.
4. **Arm D (Target-Only Loss + Multi-Template):** Measures the synergistic interaction of loss masking and template diversity.

## 3. Strict Confounder Controls
- **Parent Weights:** Canonical Stage B step 10,000 (`96357705a...`) for all arms.
- **Model Architecture:** Canonical ChakrMicro ($d=192, 6\text{ layers}, 6\text{ heads}, 3.44\text{M params}$).
- **Sequence Substrate:** Exactly identical 3,000 core sequences formatted into both single-template and multi-template datasets.
- **Optimizer & Schedule:** AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay 0.01), $lr=3\times 10^{-4}$, 300 steps, batch size 16.
- **Independent Seeds:** Executed across Seed 42 and Seed 101.

