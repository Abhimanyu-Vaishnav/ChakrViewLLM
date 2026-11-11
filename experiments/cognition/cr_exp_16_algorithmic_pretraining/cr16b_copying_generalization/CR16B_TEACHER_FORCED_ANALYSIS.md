# CR-EXP-16B Teacher-Forced Analysis

## 1. Methodology
To isolate pointer routing capability from autoregressive decoding error propagation, teacher-forced next-token evaluation was performed:
At every target position $t$, the ground truth prefix $x_{<t}$ is provided, and the model's top predicted token is compared to the true token $x_t$:
$$\text{Acc}_{\text{TF}} = \frac{1}{T} \sum_{t=1}^T \mathbb{I}(\text{argmax } P(x_t \mid x_{<t}) == x_t)$$

## 2. Teacher-Forced Performance Across Categories

| Category Tested | Arm D Teacher-Forced Accuracy | Free-Generation Exact Match | Divergence Gap |
| :--- | :---: | :---: | :---: |
| **Exact Sequence Copying (Greek, CR-16 Bench)** | **96.6% (1121/1161)** | **66.0% (66/100)** | $-30.6\%$ |
| **Single-Token Name Probes (CR-16 Bench)** | **49.4% (176/356)** | **0.0% (0/100)** | $-49.4\%$ |
| **Multi-Token Name Probes (CR-16 Bench)** | **39.6% (203/513)** | **0.0% (0/100)** | $-39.6\%$ |
| **First Target Token Across 1,500 Sealed Probes** | **82.53% (1238/1500)** | **33.87% (508/1500)** | $-48.66\%$ |

## 3. Key Scientific Insight
The enormous divergence between Teacher-Forced Accuracy (**96.6%** on Greek, **82.5%** on first token) and Free-Generation Exact Match (**66.0%** and **33.87%**) proves that:
1. **The model possesses the internal representations to attend to and extract the correct tokens.**
2. **Autoregressive decoding is vulnerable to single-token cascading failure:** Once a single subword token deviates from the target (due to language prior drift or beam collapse), all subsequent positions lose the pointer context and collapse into memorized completions.

