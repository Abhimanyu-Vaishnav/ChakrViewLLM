# CR-16E-R2 Phase A: Reconciliation of Previous Report Inconsistencies

### 1. Parameter Count Reconciliation (Canonical vs. Reported)
- **Canonical Model Documentation:** $3,443,136$ parameters.
- **CR-16E-R Reported Count:** $3,441,408$ parameters.
- **Discrepancy:** Exactly $1,728$ parameters ($3,443,136 - 3,441,408 = 1,728 = 9 \times 192$).

#### Layer-by-Layer Architectural Dissection of Instantiated `ChakrMicro`:
```
ChakrMicro (v0.1, d_model=192, n_layers=6, n_heads=6, d_mlp=512, vocab_size=4096)
├── Embedding: embedding.weight (4096, 192)                  = 786,432
├── 6 Transformer Layers (each identical):
│   ├── LayerNorm 1: norm_1.weight (192,)                   = 192
│   ├── Self-Attention:
│   │   ├── q_proj.weight (192, 192)                        = 36,864
│   │   ├── k_proj.weight (192, 192)                        = 36,864
│   │   ├── v_proj.weight (192, 192)                        = 36,864
│   │   └── out_proj.weight (192, 192)                      = 36,864
│   ├── LayerNorm 2: norm_2.weight (192,)                   = 192
│   └── SwiGLU FFN:
│       ├── gate_proj.weight (512, 192)                     = 98,304
│       ├── up_proj.weight (512, 192)                       = 98,304
│       └── down_proj.weight (192, 512)                     = 98,304
│   └── Subtotal per layer: 442,752 params x 6 layers       = 2,656,512
├── Final Norm: final_norm.weight (192,)                    = 192
└── Output Projection (lm_head): Tied to embedding.weight    = 0 (Weight-tied)
---------------------------------------------------------------------------------
TOTAL UNIQUE TRAINABLE PARAMETERS:                             3,443,136
```
- **Root Cause of Discrepancy:** The exact parameter count is strictly **3,443,136**. The figure $3,441,408$ was an analytical undercount in earlier reports that omitted $9$ of the $13$ RMSNorm scale parameters ($9 \times 192 = 1,728$ parameters, counting only $4$ norm layers instead of $13$).
- **Status:** **Reconciled.** The model weights and canonical architecture were never changed. The true parameter count is verified as $3,443,136$.

---

### 2. Training Span Curriculum & Dataset Distribution Audit

We audited all 4 training datasets using the frozen canonical tokenizer (`vocab_4096`):

#### Exact Training Target Distributions (4,000 examples per dataset):
1. **Task Breakdown Across All Four Arms:**
   - Key-Value Retrieval: $2,000$ examples ($50.0\%$).
   - Sequence Copying Replay: $1,500$ examples ($37.5\%$).
   - Negative Abstention: $500$ examples ($12.5\%$, all target string `[UNKNOWN]`).
2. **Standard Span Training Datasets (Arms A & B):**
   - **Retrieval Targets:** Strictly 1-word entities ($2,000 / 2,000$).
   - **Token Lengths in Retrieval:**
     - 2 tokens: $404$ ($20.2\%$ in Arm A), $431$ ($21.6\%$ in Arm B).
     - 3 tokens: $1,125$ ($56.3\%$ in Arm A), $1,112$ ($55.6\%$ in Arm B).
     - 4 tokens: $457$ ($22.9\%$ in Arm A), $434$ ($21.7\%$ in Arm B).
     - 5 tokens: $14$ ($0.7\%$ in Arm A), $23$ ($1.2\%$ in Arm B).
     - Range: 2 to 5 subwords (mean: 3.04 tokens).
   - **Copying Replay Targets:** 3 to 8 words ($230$ to $280$ examples each), corresponding to 5 to 28 tokens (peak at 9 tokens: $578$ examples).
3. **Balanced Span Training Datasets (Arms C & D):**
   - **Retrieval Targets:** Multi-token spans up to 8 subwords.
   - **Token Lengths in Retrieval:**
     - 2 tokens: $147$ ($7.4\%$) in C, $154$ ($7.7\%$) in D.
     - 3 tokens: $641$ ($32.1\%$) in C, $645$ ($32.3\%$) in D.
     - 4 tokens: $491$ ($24.6\%$) in C, $502$ ($25.1\%$) in D.
     - 5 tokens: $330$ ($16.5\%$) in C, $287$ ($14.4\%$) in D.
     - 6 tokens: $297$ ($14.9\%$) in C, $325$ ($16.3\%$) in D.
     - 7 tokens: $42$ ($2.1\%$) in C, $39$ ($2.0\%$) in D.
     - 8 tokens: $52$ ($2.6\%$) in C, $48$ ($2.4\%$) in D.
     - Range: 2 to 8 subwords (mean: 4.28 tokens).
4. **Resolution of Terminology Inconsistency:**
   - "Lengths 1–4" in CR-16E documentation referred to the *Benchmark B3 target span test groups* ($L \in \{1, 2, 3, 4\}$ subwords).
   - "Lengths 2–8" in dataset audit referred to the *actual tokenized length of the multi-word / long entity names* injected into the balanced training set.
   - Both descriptions are accurate within their respective contexts, but their conflation created ambiguity.

---

### 3. Metric and Classification Consistency (Paired Probes)
In Benchmark B2 Counterfactual Probes ($150$ pairs = $300$ probes), every pair is now categorized into an exhaustive, mutually exclusive partition:
- **Both Correct:** $m_A = \text{True} \land m_B = \text{True}$.
- **One Correct:** $(m_A = \text{True} \land m_B = \text{False}) \lor (m_A = \text{False} \land m_B = \text{True})$.
- **Both Wrong Invariant Failure:** $m_A = \text{False} \land m_B = \text{False} \land \hat{y}_A = \hat{y}_B$.
- **Both Wrong Variant Error:** $m_A = \text{False} \land m_B = \text{False} \land \hat{y}_A \neq \hat{y}_B$.

#### Audit of Partition Sums Across All 12 Checkpoints:
$$\text{Both Correct} + \text{One Correct} + \text{Both Wrong Invariant} + \text{Both Wrong Variant} \equiv 150 \text{ pairs (100\%)}$$
- In CR-16E-R Table D1, the column labeled `CF: Invariant Failure` reported all instances where $\hat{y}_A = \hat{y}_B$. In 10 of 12 models, this was identical to the failure cases ($m_A=F \land m_B=F$). In 2 models (Arm B s4040 and Arm D s4040), exactly 1 pair had $m_A = \text{True}$ while predicting $\hat{y}_B = \hat{y}_A$ (e.g. `expA = Chef, predA = Chef; expB = Tulip, predB = Chef`).
- The partition is now strictly reconciled and verified.

---

### 4. Copying Regression Clarification
We distinguish between individual model scores, arm means, and parent baseline:
- **Parent Baseline (CR-16A Arm D):** **66 / 100 (66.0%)** exact match ($74/100$ token presence).
- **CR-16E Individual Checkpoint Results ($N=100$ probes):**
  - Arm A (Fixed, Std): Seed 2026 = $20\%$, Seed 3030 = $3\%$, Seed 4040 = $2\%$. (Range: $2\% - 20\%$).
  - Arm B (Shuff, Std): Seed 2026 = $1\%$, Seed 3030 = $6\%$, Seed 4040 = $0\%$. (Range: $0\% - 6\%$).
  - Arm C (Fixed, Bal): Seed 2026 = $0\%$, Seed 3030 = $1\%$, Seed 4040 = $1\%$. (Range: $0\% - 1\%$).
  - Arm D (Shuff, Bal): Seed 2026 = $0\%$, Seed 3030 = $1\%$, Seed 4040 = $0\%$. (Range: $0\% - 1\%$).
- **Arm Means ($\pm s$):**
  - Arm A Mean: $8.33\% \pm 10.12\%$
  - Arm B Mean: $2.33\% \pm 3.21\%$
  - Arm C Mean: $0.67\% \pm 0.58\%$
  - Arm D Mean: $0.33\% \pm 0.58\%$
- **Cross-Arm Range:** Individual scores span $0\%$ to $20\%$, while arm means span $0.33\%$ to $8.33\%$.
- Every individual model in CR-16E suffered a severe regression compared to the $66.0\%$ parent baseline.

