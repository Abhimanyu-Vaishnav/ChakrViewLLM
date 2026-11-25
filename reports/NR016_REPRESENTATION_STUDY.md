# NR-016 Phase 5 — Representation and Generalization Diagnostics
**Mission ID:** CV-GENESIS-NR-016  
**Date:** 2026-11-25  
**Architectures Evaluated:** ChakrGenesis-17M (10 layers) vs ChakrGenesis-38M (12 layers)  
**Artifacts:** `data/scientific_eval_016/representation_diagnostics.json`

---

## 1. Executive Summary
A comparative representation study was conducted to analyze hidden state trajectories, inter-layer cosine similarities, activation norms, and counterfactual sensitivity across depth for both 17M and 38M architectures.

---

## 2. Layer-Wise Representation Trajectory Comparison

| Layer Depth | 17M Activation Norm | 17M Cosine Sim | 38M Activation Norm | 38M Cosine Sim | Structural Behavior |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **Layer 1** | $0.3923$ | $0.2439$ | $0.4535$ | $0.1821$ | Initial token embedding projection |
| **Layer 2** | $1.4903$ | $0.5275$ | $2.5860$ | $0.5200$ | Early phrase syntax formation |
| **Layer 3** | $2.9172$ | $0.7499$ | $5.0951$ | $0.7194$ | Intermediate clustering |
| **Layer 4** | $3.9713$ | $0.8021$ | $6.6172$ | $0.8213$ | Representation stabilization |
| **Layer 5** | $4.8719$ | $0.8440$ | $8.0129$ | $0.8741$ | Onset of plateau |
| **Layer 6** | $5.9038$ | $0.8773$ | $9.1102$ | $0.9028$ | High similarity plateau |
| **Layer 7** | $6.8192$ | $0.9110$ | $10.1867$ | $0.9092$ | Inter-layer plateau |
| **Layer 8** | $7.2904$ | $0.9091$ | $10.9777$ | $0.9049$ | Inter-layer plateau |
| **Layer 9** | $7.9433$ | $0.9334$ | $11.9757$ | $0.9198$ | Output projection preparation |
| **Layer 10** | $8.4095$ | **$0.9379$** | $12.7714$ | $0.9384$ | Output plateau |
| **Layer 11** | — | — | $13.4680$ | **$0.9406$** | Scaled depth plateau |
| **Layer 12** | — | — | $14.5887$ | **$0.9401$** | Scaled depth plateau |

### Key Discovery: Depth Scaling Extends the Plateau
In both models, the network settles into a high-similarity regime ($>0.90 - 0.94$) across its upper half. Adding 2 transformer layers (10 $\rightarrow$ 12) simply adds two additional steps within the $>0.94$ cosine similarity plateau, accompanied by a linear scaling of activation norms ($8.4 \rightarrow 14.6$). Neither architecture exhibits dynamic iterative re-routing.

---

## 3. Counterfactual Sensitivity & Renaming Probes
Using canonical `ChakrGenesis-17M Stage-D Best`:
- **Base `(x=12, y=15)`:** Output: `'### 21'` (Incorrect arithmetic: $12 + 15 = 27$)
- **Renamed `(a=12, b=15)`:** Output: `'#### 3'` (Premature truncation)
- **OOD `(zeta=12, theta=15)`:** Output: `'The total number of is 12'` (Template collapse)
- **Commutative Reversal `(y=15, x=12)`:** Output: `'### 24'` (Non-commutative drift)

**Deduction:** The network lacks true symbolic variable binding or commutative operator invariance; its outputs are driven by superficial token associations.
