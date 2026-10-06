# Step 227: Association Objective Study

## 1. Scientific Objective
Evaluate and compare multi-task loss formulations to determine which objective function successfully drives Stage-A association formation without degrading general language capability:
- **Candidate A**: $\mathcal{L} = \mathcal{L}_{\text{language}}$
- **Candidate B**: $\mathcal{L} = \mathcal{L}_{\text{language}} + \lambda_{\text{assoc}} \mathcal{L}_{\text{association}}$
- **Candidate C**: $\mathcal{L} = \mathcal{L}_{\text{language}} + \lambda_{\text{val}} \mathcal{L}_{\text{value\_retrieval}}$
- **Candidate D**: $\mathcal{L} = \mathcal{L}_{\text{language}} + \lambda_1 \mathcal{L}_{\text{association}} + \lambda_2 \mathcal{L}_{\text{value\_retrieval}}$

Sweep parameters: $\lambda \in \{0.25, 0.50\}$.

## 2. Objective Loss Definitions
- $\mathcal{L}_{\text{association}}$: Contrastive binding loss penalizing mismatched key-value cross products:
  $$\mathcal{L}_{\text{assoc}} = \max\left(0,\, 0.50 - \left[\cos(k_q \odot v_{\text{target}}, k_q \odot v_{\text{target}}) - \cos(k_q \odot v_{\text{target}}, k_q \odot v_{\text{neg}})\right]\right)$$
- $\mathcal{L}_{\text{value\_retrieval}}$: Direct contextual alignment loss:
  $$\mathcal{L}_{\text{val}} = 1.0 - \cos(h_{\text{terminal}}, h_{v_{\text{target}}})$$

## 3. Comparative Evaluation Results

| Candidate | $\lambda_{\text{assoc}}$ | $\lambda_{\text{val}}$ | Train Loss | Lang Val Loss | In-Dist Acc | Held-Out Acc | Disjoint Rep Score | Stage-A Created? |
|---|---|---|---|---|---|---|---|---|
| A (Lang Only) | 0.00 | 0.00 | 1.1520 | 7.9210 | 0.4000 | 0.0000 | 0.0000 | False |
| B (Assoc 0.25) | 0.25 | 0.00 | 1.0850 | 7.8450 | 0.6000 | 0.0000 | 0.0450 | False |
| B (Assoc 0.50) | 0.50 | 0.00 | 1.1210 | 8.0120 | 0.5000 | 0.0000 | 0.0410 | False |
| C (ValRet 0.25) | 0.00 | 0.25 | 1.1400 | 7.8900 | 0.5000 | 0.0000 | 0.0380 | False |
| C (ValRet 0.50) | 0.00 | 0.50 | 1.1850 | 8.1500 | 0.4000 | 0.0000 | 0.0320 | False |
| D (Joint 0.25/0.25) | 0.25 | 0.25 | 1.0920 | 8.0250 | 0.6000 | 0.0000 | 0.0480 | False |

## 4. Scientific Verdict
- **Best Candidate**: `Candidate_B_Assoc_0.25` / `Candidate_D_Joint_0.25_0.25`.
- Both auxiliary objectives produce slight improvements in training stability and in-distribution accuracy ($0.60$).
- However, none of the tested objectives achieved the required Stage-A separation threshold ($\ge 0.50$) on held-out entities; Stage-A formation on disjoint identities remains a structural challenge.
