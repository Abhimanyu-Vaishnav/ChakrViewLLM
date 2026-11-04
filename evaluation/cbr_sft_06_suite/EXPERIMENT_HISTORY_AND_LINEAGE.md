# CBR Experiment Lineage and History: CBR-01 Through CBR-05

**Date:** 2026-11-04  
**Mission:** CBR-SFT-06 Architecture & Training Diagnostics  
**Repository:** `D:\Project\ChakrView`  

---

## 1. Lineage Tree & Evolution

```
Canonical Pretrained Step 10,000 (96357705...)
  │
  └── CBR-SFT-01 (Chat scaffolding baseline)
        │
        └── CBR-SFT-02 (Step 50: 592553c3...)
              │
              └── CBR-SFT-03B (Step 200: 297dcc60...)
                    │
                    └── CBR-SFT-04 (Conversational Grounding, Step 200: 327118ce...)
                          │  [Primary Production Ancestor: 25.0% CBR-04, 25.7% CBR-03B, 6.7% M3]
                          │
                          └── CBR-SFT-05 (Context Binding & KV Induction)
                                ├── Step 50:  a619297f...
                                ├── Step 100: 8653d0bb... [Best CBR-05 candidate: 4.2% CBR-05, 50.0% Syntax]
                                ├── Step 150: fc5f33e2...
                                ├── Step 200: 34bc6a12... [Collapsed into abstention basin]
                                ├── Step 250: 46d820df...
                                └── Step 300: 7dcb81dc...
```

---

## 2. Quantitative Comparison Across Iterations

| Iteration | Dataset Size | LR Schedule | Steps | Primary Strengths | Primary Weaknesses | Lineage Status |
| :--- | :---: | :---: | :---: | :--- | :--- | :--- |
| **CBR-SFT-02** | 200 | $1 \times 10^{-4}$ | 200 | Basic role delimitation, chat formatting | Rigid greetings, no time conditioning | Ancestor (Step 50) |
| **CBR-SFT-03B**| 120 | $1.5 \times 10^{-4}$| 300 | Natural conversational greetings, identity | No context retention, context-blind | Ancestor (Step 200) |
| **CBR-SFT-04** | 160 | $2 \times 10^{-4}$ | 300 | Time-conditioned greetings (100% morning, 60% evening), identity boundaries | Exact copy 0.0%, multi-turn retention 0.0% | **Current Lineage Anchor** (Step 200) |
| **CBR-SFT-05** | 400 | $1.2 \times 10^{-4}$| 300 | Prefix copying (leading digits/octets), 100% appropriate abstention | Catastrophic forgetting of greetings (1.2%), overfit abstention basin | Experimental candidate (Step 100) |
