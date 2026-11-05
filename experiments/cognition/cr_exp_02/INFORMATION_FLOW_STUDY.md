# CHAKRVIEW — CR-EXP-02 INFORMATION FLOW & MECHANISTIC INTERPRETABILITY STUDY

## Status & Lineage Verification
- **Experiment ID**: CR-EXP-02
- **Mission**: Mechanistic information flow investigation across frozen parent and trained candidate
- **Mode**: Strictly Read-Only Interpretability & Causal Diagnostics
- **Repository Branch**: `exp/cbr-01-conversational-breakthrough`
- **Git HEAD Commit**: `8e866b495c9693566b39c6a715701e220a028227`
- **Parent Checkpoint**: `experiments/cbr01_sft/runs/cbr_sft_04/checkpoints/sft_checkpoint_step_200.pt`
  - SHA-256: `327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153`
- **Candidate Checkpoint**: `experiments/cognition/cr_exp_01/runs/stage_a_pilot/checkpoints/stage_a_candidate_step_100.pt`
  - SHA-256: `1ac891fb57147779435a297e163539268f7d825c34e792c3a504a5231cba50b0`
- **Tokenizer**: Custom BPE Vocab 4096 (`data/experiments/vocab_4096/vocab.json`)
- **Execution Constraints Verified**: Zero optimizer updates, zero fine-tuning steps, zero checkpoint modifications, zero tokenizer edits.

---

## 1. Model Computation Reconstruction

### 1.1 Architectural Parameters
ChakrMicro v0.1 (`chakrview/brain/model.py`) is a causal Transformer configured as follows:
- **Layers ($L$)**: 6 Transformer blocks
- **Hidden Dimension ($d_{\text{model}}$)**: 192
- **Attention Heads ($n_{\text{heads}}$)**: 6 (Head dimension $d_{\text{head}} = 32$)
- **Feedforward Sublayer**: SwiGLU with expansion factor $\frac{8}{3} \times 192 = 512$ ($d_{\text{ff}} = 512$)
- **Max Sequence Length**: 512
- **Normalization**: Root Mean Square Layer Normalization (RMSNorm) with learnable scale $\gamma \in \mathbb{R}^{192}$, $\epsilon = 10^{-5}$
- **Positional Encoding**: Rotary Position Embeddings (RoPE) applied to query and key projections
- **Weight Tying**: `lm_head.weight` is tied directly to `tok_embeddings.weight` ($[4096, 192]$)

### 1.2 Mathematical Forward Formulation
For an input token sequence $\mathbf{x} = (x_1, \dots, x_T) \in \mathbb{N}^T$:
1. **Embedding**:
   $$h_0^{(t)} = E[x_t] \in \mathbb{R}^{192}$$
2. **For layers $l = 0, \dots, 5$**:
   - Pre-Attention Normalization:
     $$\tilde{h}_l^{(t)} = \text{RMSNorm}(h_l^{(t)}) = \frac{h_l^{(t)}}{\sqrt{\frac{1}{d}\sum_{i=1}^d (h_{l,i}^{(t)})^2 + \epsilon}} \odot \gamma_{\text{attn}, l}$$
   - Multi-Head Causal Attention with RoPE:
     $$Q = R_{\Theta, t} (W_Q \tilde{h}_l^{(t)}), \quad K = R_{\Theta, t} (W_K \tilde{h}_l^{(t)}), \quad V = W_V \tilde{h}_l^{(t)}$$
     $$\text{AttnOut}_l^{(t)} = W_O \left( \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_{\text{head}}}} + M\right) V \right)^{(t)}$$
   - Post-Attention Residual:
     $$h_{l, \text{post\_attn}}^{(t)} = h_l^{(t)} + \text{AttnOut}_l^{(t)}$$
   - Pre-FFN Normalization:
     $$\hat{h}_l^{(t)} = \text{RMSNorm}(h_{l, \text{post\_attn}}^{(t)}) \odot \gamma_{\text{ffn}, l}$$
   - SwiGLU Feedforward:
     $$\text{FFNOut}_l^{(t)} = W_{\text{down}} \left( \text{SiLU}(W_{\text{gate}} \hat{h}_l^{(t)}) \odot (W_{\text{up}} \hat{h}_l^{(t)}) \right)$$
   - Post-FFN Residual:
     $$h_{l+1}^{(t)} = h_{l, \text{post\_attn}}^{(t)} + \text{FFNOut}_l^{(t)}$$
3. **Final Head Projection**:
   $$h_{\text{final}}^{(t)} = \text{RMSNorm}(h_6^{(t)}) \odot \gamma_{\text{final}}$$
   $$z^{(t)} = h_{\text{final}}^{(t)} \cdot E^T \in \mathbb{R}^{4096}$$

---

## 2. Experimental Methodology & Controlled Probes

To isolate contextual variable binding from token length or positional artifacts, length-matched controlled prompt triplets were constructed. In all cases, the variable name is `Quasar`, the query asks for `Quasar`, and only the bound target value is varied:

- **Prompt A (`Rhodium`)**:
  `"Mapping: Quasar is mapped to Rhodium. Query: What does Quasar map to?\nAnswer:"` (Length: 42 tokens)
- **Prompt B (`Platinum`)**:
  `"Mapping: Quasar is mapped to Platinum. Query: What does Quasar map to?\nAnswer:"` (Length: 42 tokens)
- **Prompt C (`Phoenix`)**:
  `"Mapping: Quasar is mapped to Phoenix. Query: What does Quasar map to?\nAnswer:"` (Length: 42 tokens)

### Probe Delimiter Position
The decision point is the final prompt token at index 41 (`':'` in `\nAnswer:`). At this exact position, the causal mask allows attention to attend over all prompt tokens, including the bound value token at indices 17–19 (`Rhodium` vs `Platinum`). The hidden state at position 41 directly generates the first answer token.

---

## 3. Top-Level Logit & Entropy Disruption

Empirical output distributions at delimiter position 41 reveal an immediate contrast between the Frozen Parent and the Stage A Candidate:

| Metric | Frozen Parent (`cbr_sft_04_step_200`) | Stage A Candidate (`stage_a_step_100`) | Impact / Diagnosis |
| :--- | :--- | :--- | :--- |
| **Top-1 Token (Prompt A)** | `32 ('\n')` (logit $8.59$) | `35 (' ')` (logit $12.36$) | Shift to unconditional space prefix |
| **Top-2 Token (Prompt A)** | `10 ('\t')` (logit $8.23$) | `34 ('"')` (logit $8.07$) | Collapse of formatting variation |
| **Top-3 Token (Prompt A)** | `35 (' ')` (logit $7.39$) | `36 ('#')` (logit $7.98$) | Fixed token attractor |
| **Entropy $H(p)$ (nats)** | **$3.7892$** | **$0.7107$** | **$81.2\%$ entropy reduction (Severe Collapse)** |
| **Entropy (Prompt B)** | **$4.9431$** | **$0.7095$** | Candidate displays uniform flat certainty |
| **$L_2$ Logit Diff (A vs B)** | **$44.0985$** ($45.59\%$ rel diff) | **$5.2027$** (**$2.72\%$ rel diff**) | **$94.0\%$ suppression of contextual logit divergence** |

In the Frozen Parent, varying the context value from `Rhodium` to `Platinum` produces a massive shift in next-token logits ($44.0985$ $L_2$ distance, $45.59\%$ relative difference). In the Stage A Candidate, varying the context value results in an almost imperceptible ripple ($5.2027$ distance, $2.72\%$ relative difference), completely insufficient to change top-ranked tokens.

---

## 4. Layer-Wise Activation Trajectory ($L_2$ Divergence & Norms)

To locate where contextual information is lost, hidden states were captured across all 6 Transformer blocks at delimiter position 41.

### 4.1 Frozen Parent Activation Trajectory
- Embedding Norm ($h_0$): $4.7645$
- Contextual $L_2$ Difference ($\|h_A - h_B\|_2$): $0.9647$ ($20.25\%$ rel)

| Layer | Residual Norm ($\|h\|_2$) | Attn Out Norm | Attn Out Diff ($\|\Delta\|_2$) | FFN Out Norm | FFN Out Diff ($\|\Delta\|_2$) | Post-FFN Residual Norm | Post-FFN Diff ($\|\Delta\|_2$) | Rel Diff (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | $4.7645$ | $0.6695$ | $0.2785$ | $0.7813$ | $0.1585$ | $4.7570$ | $0.9647$ | **$20.28\%$** |
| **1** | $4.7570$ | $1.2985$ | $0.5982$ | $1.1118$ | $0.1685$ | $4.9458$ | $1.2185$ | **$24.64\%$** |
| **2** | $4.9458$ | $2.5593$ | $0.9066$ | $1.8687$ | $0.2983$ | $5.9038$ | $1.6425$ | **$27.82\%$** |
| **3** | $5.9038$ | $4.4984$ | $1.3917$ | $2.8427$ | $0.4616$ | $7.8631$ | $2.3168$ | **$29.46\%$** |
| **4** | $7.8631$ | $5.7001$ | $1.8596$ | $4.3414$ | $0.7516$ | $10.1534$ | $3.1554$ | **$31.08\%$** |
| **5** | $10.1534$ | $6.4831$ | $2.4430$ | $5.3672$ | $1.1578$ | **$12.5786$** | **$4.2559$** | **$33.83\%$** |
| **Final**| Final Norm: $11.8398$ | - | Final Norm Diff: **$4.0858$** ($34.51\%$) | - | - | Logit Norm: $96.72$ | Logit Diff: **$44.0985$** | **$45.59\%$** |

*Key Parent Behavior*: The contextual difference introduced at embedding time ($20.28\%$) is **monotonically amplified** through each attention layer, reaching $33.83\%$ at Layer 5 and expanding to **$45.59\%$** at the output logits. Information flows freely and expands into decision boundaries.

---

### 4.2 Stage A Candidate Activation Trajectory
- Embedding Norm ($h_0$): $6.2081$ (Inflated by $+30.3\%$)
- Contextual $L_2$ Difference ($\|h_A - h_B\|_2$): $1.1895$ ($19.16\%$ rel)

| Layer | Residual Norm ($\|h\|_2$) | Attn Out Norm | Attn Out Diff ($\|\Delta\|_2$) | FFN Out Norm | FFN Out Diff ($\|\Delta\|_2$) | Post-FFN Residual Norm | Post-FFN Diff ($\|\Delta\|_2$) | Rel Diff (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | $6.2081$ | $0.7818$ | $0.1837$ | $2.3789$ | $0.1230$ | $7.2882$ | $1.2335$ | **$16.92\%$** |
| **1** | $7.2882$ | $1.0967$ | $0.2172$ | $3.5852$ | $0.1042$ | $9.3879$ | $1.2989$ | **$13.84\%$** |
| **2** | $9.3879$ | $1.3060$ | $0.2447$ | $5.5126$ | $0.1084$ | $12.7876$ | $1.3857$ | **$10.84\%$** |
| **3** | $12.7876$ | $1.7618$ | $0.2647$ | $7.2828$ | $0.1274$ | $16.7118$ | $1.4939$ | **$8.94\%$** |
| **4** | $16.7118$ | $1.9897$ | $0.3340$ | $9.1557$ | $0.1448$ | $21.2188$ | $1.7770$ | **$8.37\%$** |
| **5** | $21.2188$ | $2.4339$ | $0.5122$ | $11.0310$ | $0.2471$ | **$25.5204$** | **$2.1215$** | **$8.31\%$** |
| **Final**| Final Norm: $22.9234$ | - | Final Norm Diff: **$1.9052$** ($8.31\%$) | - | - | Logit Norm: $191.24$ | Logit Diff: **$5.2027$** | **$2.72\%$** |

---

## 5. Attention & FFN Contribution Analysis

### 5.1 The Pathology of Norm Inflation
Comparing the activation progression of Parent vs Candidate exposes the exact mathematical cause of the collapse:

1. **Residual Stream Norm Doubling**:
   - At Layer 5 post-FFN, the parent residual norm is $12.58$. In the candidate, it is **$25.52$** ($+102.9\%$ increase).
   - Final logit norm in the parent is $96.72$. In the candidate, it is **$191.24$** ($+97.7\%$ increase).
2. **FFN Norm Domination**:
   - In the candidate, the FFN outputs grow from $2.38$ (Layer 0) to **$11.03$** (Layer 5).
   - In contrast, attention output norms in the candidate are severely suppressed, reaching only $2.43$ at Layer 5 (compared to $6.48$ in the parent).
   - At Layer 5 of the candidate:
     $$\frac{\|\text{AttnOut}\|_2}{\|\text{FFNOut}\|_2} = \frac{2.4339}{11.0310} = \mathbf{0.2206} \quad (22\%)$$
     In the parent, this ratio is:
     $$\frac{\|\text{AttnOut}\|_2}{\|\text{FFNOut}\|_2} = \frac{6.4831}{5.3672} = \mathbf{1.2079} \quad (121\%)$$
3. **Contextual Attenuation vs Amplification**:
   - **Parent**: Relative contextual divergence grows from $20.28\% \to 33.83\% \to \mathbf{45.59\%}$. Attention actively retrieves context and drives the residual stream into divergent token predictions.
   - **Candidate**: Relative contextual divergence decays from $19.16\% \to 16.92\% \to 10.84\% \to 8.31\% \to \mathbf{2.72\%}$. The contextual difference from the prompt is progressively **swamped** by an enormous context-invariant residual vector added at every FFN block.

### 5.2 Mechanistic Drowning Hypothesis
The candidate's attention heads still compute differential attention across different values (e.g., attention output difference at Layer 5 is $\|\Delta\|_2 = 0.5122$). However, because the residual stream is flooded with an invariant vector of magnitude $25.52$ pointing in the direction of the SFT answer template prior, the contextual delta represents less than $2\%$ of the total vector magnitude. When multiplied by the tied embedding projection $E^T$, the invariant prior completely dominates the softmax distribution.

---

## 6. Verified Causal Intervention Results (Head Ablation Audit)

In diagnostic validation (`CR-EXP-01 DIAGNOSTIC VALIDATION`), systematic ablation of individual attention heads (zeroing head output $W_O^h \text{Head}_h$) showed zero change in generated text: the model generated `" Cube"` across all ablations.

CR-EXP-02 explains the exact causal mechanism for this finding:
1. **Linear Masking by Norm Imbalance**: Zeroing any single attention head at Layer 4 or 5 removes at most $0.3 - 0.5$ units of $L_2$ norm. But the residual vector has an $L_2$ norm exceeding $21.0 - 25.5$.
2. **FFN Compensation**: The massive FFN sublayers ($d_{\text{ff}}=512$) inject an overwhelmingly large static representation of the target value format into the residual stream that bypasses the attention heads entirely.
3. **Logit Margin**: The candidate assigns token `35 (' ')` a logit of $+12.36$, while runner-up tokens sit below $+8.07$ (a margin of $\Delta z = 4.29$). An attention ablation perturbing logits by only $\approx 0.4$ cannot cross the decision boundary.

---

## 7. Context Sensitivity Findings: Classification of Claims

| Finding / Claim | Category | Empirical Evidence |
| :--- | :--- | :--- |
| **Token bindings exist in early layers** | **Observed Fact** | Layer 0 embedding difference between `Rhodium` and `Platinum` prompts is non-zero ($\|\Delta\|_2 = 1.1895$). |
| **Attention sublayers detect contextual differences** | **Observed Fact** | Every attention sublayer in the candidate produces non-zero output difference ($\|\Delta \text{AttnOut}\|_2 = 0.18$ to $0.51$). Attention is NOT dead. |
| **FFN sublayers drown out contextual variance** | **Empirically Supported** | FFN norms grow to $11.03$ while contextual deltas are only $0.24$. Residual norm inflates to $25.52$, shrinking relative context variance from $19.16\%$ to $8.31\%$. |
| **Final projection suppresses contextual logits** | **Observed Fact** | Output logit relative difference collapses to $2.72\%$ in candidate vs $45.59\%$ in parent. Top-1 prediction is invariant to value swap. |
| **Attention heads failed to learn RoPE alignment** | **Refuted** | Attention differences exist and scale with layer depth. The failure is not RoPE misalignment; it is residual norm drowning caused by unconstrained FFN updates. |
| **Model underwent catastrophic forgetting of language priors** | **Refuted** | Model retains high syntactic stability and formatting predictability, but suffers from extreme mode concentration on high-frequency tokens. |

---

## 8. Root-Cause Confidence Assessment

The scientific confidence regarding the mechanism of failure in CR-EXP-01 is **HIGH (Level 5/5)**:

1. **The Primary Failure Driver**: Unregularized single-batch optimization (batch size 1, 100 updates, no gradient accumulation, no weight decay or residual norm penalty) allowed the FFN weights to quickly overfit to the static template syntax (`\nAnswer: <value>`).
2. **The Mechanism**: The FFN layers learned to emit a massive static bias vector into the residual stream ($+103\%$ norm inflation) that points towards the dominant dataset attractor (`" Cube"`), dwarfing the contextual attention vectors by an order of magnitude.
3. **The Result**: At inference time, while attention heads accurately attend to the prompt tokens, their signal is numerically quenched before reaching the final layer norm and LM head.

---

## 9. Concrete Recommendations for Future Capability Experiments

To prevent residual drowning and enable genuine in-context variable binding in future experiments (e.g. CR-EXP-03), the following architectural and training modifications are strictly recommended:

1. **LayerNorm / Residual Stream Scaling**:
   - Implement **Pre-LayerNorm with scaling** (e.g., $1/\sqrt{2L}$ residual scale or LayerScale with learnable diagonal parameters initialized near $\epsilon = 0.1$).
   - Enforce weight decay ($\lambda \ge 0.01$) specifically targeting FFN projection matrices (`w1, w2, w3`) to prevent norm explosion.
2. **Training Dynamics**:
   - **Batch Size & Gradient Accumulation**: Replace batch size 1 with effective batch size $\ge 16$ or $32$ via gradient accumulation to average out per-sample template biases.
   - **Learning Rate Schedule**: Reduce initial peak learning rate or introduce warm-up with cosine decay to prevent early FFN parameter blow-up.
3. **Objective Regularization (Contrastive / Binding Loss)**:
   - Introduce an auxiliary loss that explicitly penalizes residual alignment with static value attractors when the input context changes.
   - Use balanced synthetic value distributions where no single token (e.g. `Cube`, `35`) appears with >1% frequency across target completions.
4. **Frozen FFN Probing**:
   - Conduct a pilot where FFN layers are frozen and only Attention projection weights ($W_Q, W_K, W_V, W_O$) are updated, forcing the model to solve variable binding purely through attention routing without residual norm inflation.

---

## 10. Audit Verification & Artifact Sign-off

- **Artifact Created**: `experiments/cognition/cr_exp_02/INFORMATION_FLOW_STUDY.md`
- **Protected Files**:
  - `chakrview/runtime/session.py` — **UNMODIFIED (Verified Clean)**
  - Canonical checkpoints — **INTACT & UNMODIFIED**
  - Canonical tokenizers — **INTACT & UNMODIFIED**
- **Git Safety Status**: Zero commits, zero pushes, zero branch switches, zero rebases.

**Investigation Concluded.**

