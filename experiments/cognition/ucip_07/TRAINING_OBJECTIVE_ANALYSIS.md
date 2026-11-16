# UCIP-07: Training Objective & Loss Dynamics Analysis

## 1. Overview of the Objective Formulation

ChakrView employs an autoregressive next-token prediction objective with sequence loss masking:

$$\mathcal{L}_{AR} = -\frac{1}{\sum_{t=1}^{T} m_t} \sum_{t=1}^{T} m_t \log P(x_t \mid x_{<t}; \theta)$$

where $m_t \in \{0, 1\}$ is a binary token mask.

---

## 2. Comparative Analysis of Training Regimes

### 2.1 Full-Sequence Masking vs Prompt-Masked Autoregressive Loss

In UCIP-06 and prior stages, loss was frequently evaluated over the prompt tokens as well as target completions. During UCIP-07 diagnostics, we isolated the effects:
- **Prompt-Included Loss**: Models learn to predict standard user greetings (`"User: Hello"`) with extremely low entropy ($loss < 0.3$), artificially deflating overall cross-entropy while leaving response generation undertrained.
- **Prompt-Masked Loss ($m_t = 0$ for prompt, $m_t = 1$ for completion)**: Directs 100% of gradient updates to token transitions following `\nAssistant:`. This eliminated prompt-token memorization and forced the model to prioritize response semantics.

### 2.2 Exposure Bias: Teacher-Forcing vs Free Rollout

Forensic tracing revealed that exposure bias is the primary cause of conversational attractor loops.
- **Under Teacher-Forcing**: The model conditions on ground-truth prior tokens at step $t-1$. Cross-entropy loss is low ($1.049$).
- **Under Free Rollout**: At step $t=1$, if the top-1 prediction deviates by even a fraction (e.g. predicting `' A '` followed by `'comp'` instead of `'CPU'`), the model encounters a sequence it never observed during teacher-forced training.
- Because the network lacks exposure to off-manifold states, it immediately snaps to its highest-density training attractor:
  `"A compiler transforms source code into machine language..."` or `"Your project is a computer..."`.

---

## 3. Label Smoothing & Entropy Regularization Experiments

We analyzed whether adding label smoothing $\epsilon = 0.05$ or entropy regularization prevents attractor collapse:
- **Without Label Smoothing ($\epsilon = 0.0$)**: Softmax probabilities peak sharply ($p > 0.92$ on common connectives). Once a single drift occurs, subsequent tokens follow deterministic unigram priors.
- **With Label Smoothing ($\epsilon = 0.05$)**: Prevents overconfidence in head weights and dampens the logit spikes that cause repetitive loops. Repetitive probe outputs dropped from $24$ down to $6$ across the benchmark.

---

## 4. Empirical Loss Trajectories

### UCIP-07 ChakrMedium Fine-Tuning
- **Initial Loss**: $1.3712$
- **Step 50**: $1.2184$
- **Step 100**: $1.1092$
- **Step 150 (Final Best)**: **$1.0492$**
- **Validation Drift**: None observed; validation loss tracks training loss steadily within $0.05$ margin.

### UCIP-07 ChakrCore Scratch Training
- **Initial Loss**: $5.7601$
- **Step 50**: $4.6210$
- **Step 100**: $3.7844$
- **Step 150**: $3.2981$
- **Step 200**: $3.0115$
- **Step 250 (Final Best)**: **$2.8339$**

The trajectory shows smooth, monotonic descent, confirming healthy gradient flow across all 8 layers and 27.8M parameters. However, achieving fluency requires continuing this trajectory below $1.50$, which requires $\sim 50,000+$ optimization steps on a multi-million token pretraining corpus.
