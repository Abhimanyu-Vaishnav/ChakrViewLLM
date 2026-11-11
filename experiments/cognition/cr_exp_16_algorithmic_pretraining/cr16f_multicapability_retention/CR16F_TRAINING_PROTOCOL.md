# CR-16F Phase B: Frozen Training Protocol

### 1. Factorial Matrix Design ($2 \times 2$ with $N=3$ Seeds = 12 Models)

| Experimental Arm | Copying Replay Fraction | Target Negative Loss Multiplier ($\lambda$) | Batch Composition (Size 16) | Seeds Trained |
|---|:---:|:---:|:---:|:---:|
| **Arm A** | **25%** | **1.00** | 4 Copying + 12 Retrieval/Abstention | 2026, 3030, 4040 |
| **Arm B** | **50%** | **1.00** | 8 Copying + 8 Retrieval/Abstention | 2026, 3030, 4040 |
| **Arm C** | **25%** | **0.25** | 4 Copying + 12 Retrieval/Abstention | 2026, 3030, 4040 |
| **Arm D** | **50%** | **0.25** | 8 Copying + 8 Retrieval/Abstention | 2026, 3030, 4040 |

---

### 2. Precise Mathematical Loss Formulation

Let a batch consist of examples indexed by $i \in \{1, \dots, B\}$ with task $T_i \in \{\text{copying}, \text{retrieval}, \text{abstention}\}$.
For token position $t$ in example $i$, cross-entropy loss is computed on unmasked tokens ($y_{i,t} \neq -100$):
$$\ell_{i,t} = - \log P(y_{i,t} \mid x_{i,<t})$$

The task-weighted per-token loss is defined as:
$$w_i = \begin{cases} 
1.0 & \text{if } T_i \in \{\text{copying}, \text{retrieval}\} \\ 
\lambda & \text{if } T_i = \text{abstention} \quad (\lambda \in \{1.00, 0.25\}) 
\end{cases}$$

The total scalar loss optimized by backpropagation is:
$$\mathcal{L} = \frac{\sum_{i=1}^B \sum_{t} w_i \cdot \ell_{i,t} \cdot \mathbb{I}(y_{i,t} \neq -100)}{\sum_{i=1}^B \sum_{t} \mathbb{I}(y_{i,t} \neq -100)}$$

- In Arms A & B ($\lambda = 1.00$), $w_i = 1.0$ for all tasks, producing standard cross-entropy loss.
- In Arms C & D ($\lambda = 0.25$), negative abstention tokens contribute exactly $25\%$ of their gradient magnitude relative to standard cross-entropy, while copying and positive retrieval tokens remain at $100\%$ weight.
- The denominator normalizes by total active target tokens, preventing artificial gradient scaling.

---

### 3. Frozen Training Invariants

| Hyperparameter / Control Variable | Frozen Value across all 12 Runs |
|---|---|
| **Parent Checkpoint** | CR-16A Arm D (`9d609a77...`) |
| **Model Architecture** | Canonical ChakrMicro v0.1 ($3,443,136$ parameters) |
| **Vocabulary & Tokenizer** | Canonical `vocab_4096` |
| **Optimizer** | AdamW ($\beta_1 = 0.9, \beta_2 = 0.95$, weight decay $0.01$) |
| **Learning Rate** | $2 \times 10^{-4}$ (constant with linear warmup) |
| **Batch Size** | Exactly 16 examples |
| **Training Steps** | Exactly 400 optimizer steps |
| **Gradient Clipping** | Maximum norm $1.0$ |
| **Sequence Length** | 64 tokens |
| **Checkpoint Selection** | Best validation loss on `cr16e_multitask_val.json` |

