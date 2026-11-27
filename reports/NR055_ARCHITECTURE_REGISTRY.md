# NR-055: AUTHORITATIVE ARCHITECTURE REGISTRY & TENSOR FORENSICS

**Mission Identifier:** NR-055  
**Date:** November 2026  
**Auditor:** ChakrView Independent Forensic Engineering Team  
**Scope:** Exhaustive tensor-level state dictionary inspection, architecture configuration verification, parameter accounting, and resolution of documented 38M architectural discrepancies.

---

## 1. Executive Summary

A comprehensive, byte-level and tensor-level audit was conducted across every model architecture definition and checkpoint in the ChakrView repository. Every checkpoint state dictionary was inspected to determine:
- Tensor shapes for embedding, query projections, key/value projections, output projections, MLP gates, and normalization layers.
- Number of unique trainable parameters versus total state dictionary elements (accounting for tied weights).
- Actual loaded architecture configurations versus documented specifications.

### Key Audit Resolution:
**The 38M Architectural Discrepancy is Resolved:**
- In NR-054 summary tables, the 38M model was cited as `d_model 768, 6 layers, 12 heads`.
- In contrast, the code in `chakrview/brain/genesis/chakr_genesis_38m.py` and the training configurations in NR-050, NR-051, and NR-052 specify `d_model 512, 12 layers, 8 Q heads / 4 KV heads, hidden_dim 1408`.
- **Finding:** Tensor inspection of all four 38M checkpoints (`chakr_38m_pretrained_base.pt`, `chakr_38m_sft_candidate.pt`, `chakr_38m_v4_base.pt`, and `chakr_38m_nr052_sft.pt`) confirms **100% agreement with `d_model 512, 12 layers, 8 Q heads, 4 KV heads, hidden_dim 1408`**.
- The `d=768, L=6` citation was a documentation transcription error carried over into the NR-054 table from an exploratory scratch config. There is **only one genuine 38M architecture family** in the repository: `ChakrGenesis38M`.

---

## 2. Reconciled Architectural Specifications by Family

The repository contains exactly two active foundational model architecture families:

### Family A: ChakrGenesis-17M (`ChakrGenesis17M`)
- **Model Dimension ($d_{\text{model}}$):** 384
- **Transformer Layers ($L$):** 10
- **Attention Heads ($n_{\text{q}}$):** 12 (Head dimension = 32)
- **Key/Value Heads ($n_{\text{kv}}$):** 3 (Grouped Query Attention 4:1 ratio)
- **Feed-Forward Dimension ($\text{hidden\_dim}$):** 1024 (SwiGLU: gate `[1024, 384]`, up `[1024, 384]`, down `[384, 1024]`)
- **Positional Encoding:** Rotary Position Embedding (RoPE, $\theta = 10000.0$)
- **Normalization:** RMSNorm ($\epsilon = 10^{-5}$, learned weights, no bias)
- **Embedding Tying:** Token embeddings tied to LM head (`W_out = E^T`)
- **Vocabulary Size:** 4,096 canonical BPE tokens
- **Max Context Length:** 256–512 tokens (dynamically supported up to 1024 via RoPE)
- **Unique Trainable Parameters:** **17,063,808** (~17.06M)
- **Total State Dictionary Elements:** **18,636,672** (~18.64M when tied LM head is exported as separate key)

### Family B: ChakrGenesis-38M (`ChakrGenesis38M`)
- **Model Dimension ($d_{\text{model}}$):** 512
- **Transformer Layers ($L$):** 12
- **Attention Heads ($n_{\text{q}}$):** 8 (Head dimension = 64)
- **Key/Value Heads ($n_{\text{kv}}$):** 4 (Grouped Query Attention 2:1 ratio)
- **Feed-Forward Dimension ($\text{hidden\_dim}$):** 1408 (SwiGLU: gate `[1408, 512]`, up `[1408, 512]`, down `[512, 1408]`)
- **Positional Encoding:** Rotary Position Embedding (RoPE, $\theta = 10000.0$)
- **Normalization:** RMSNorm ($\epsilon = 10^{-5}$, learned weights, no bias)
- **Embedding Tying:** Token embeddings tied to LM head (`W_out = E^T`)
- **Vocabulary Size:** 4,096 canonical BPE tokens
- **Max Context Length:** 256–512 tokens
- **Unique Trainable Parameters:** **37,499,392** (~37.50M)
- **Total State Dictionary Elements:** **39,596,544** (~39.60M when tied LM head is exported as separate key)

---

## 3. Authoritative Checkpoint State Dictionary Inspection Registry

| Checkpoint Name | File Path | Layers | $d_{\text{model}}$ | Attention Projections ($W_q, W_k, W_v, W_o$) | MLP Projections ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$) | Unique Parameters | Total Elements | Tied Weights? |
| :--- | :--- | :---: | :---: | :--- | :--- | :---: | :---: | :---: |
| **Stage-D Canonical** | `experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt` | 10 | 384 | Q:[384,384], K:[96,384], V:[96,384], O:[384,384] | G:[1024,384], U:[1024,384], D:[384,1024] | 17,063,808 | 18,636,672 | YES (Tied) |
| **RC1 Release** | `release/v1.0.0-rc1/chakr_1_0_rc1_best.pt` | 10 | 384 | Q:[384,384], K:[96,384], V:[96,384], O:[384,384] | G:[1024,384], U:[1024,384], D:[384,1024] | 17,063,808 | 18,636,672 | YES (Tied) |
| **RC2 Experimental** | `experiments/genesis_stage_rc2_046/chakr_1_0_rc2_best.pt` | 10 | 384 | Q:[384,384], K:[96,384], V:[96,384], O:[384,384] | G:[1024,384], U:[1024,384], D:[384,1024] | 17,063,808 | 18,636,672 | YES (Tied) |
| **RC3 Baseline** | `experiments/fast_track_047_049/chakr_1_0_rc3_best.pt` | 10 | 384 | Q:[384,384], K:[96,384], V:[96,384], O:[384,384] | G:[1024,384], U:[1024,384], D:[384,1024] | 17,063,808 | 18,636,672 | YES (Tied) |
| **NR051 17M Base** | `experiments/pretraining_nr051/chakr_17m_pretrained_base.pt` | 10 | 384 | Q:[384,384], K:[96,384], V:[96,384], O:[384,384] | G:[1024,384], U:[1024,384], D:[384,1024] | 17,063,808 | 18,636,672 | YES (Tied) |
| **NR051 38M Base** | `experiments/pretraining_nr051/chakr_38m_pretrained_base.pt` | 12 | 512 | Q:[512,512], K:[256,512], V:[256,512], O:[512,512] | G:[1408,512], U:[1408,512], D:[512,1408] | 37,499,392 | 39,596,544 | YES (Tied) |
| **NR051 17M SFT** | `experiments/pretraining_nr051/chakr_17m_sft_candidate.pt` | 10 | 384 | Q:[384,384], K:[96,384], V:[96,384], O:[384,384] | G:[1024,384], U:[1024,384], D:[384,1024] | 17,063,808 | 18,636,672 | YES (Tied) |
| **NR051 38M SFT** | `experiments/pretraining_nr051/chakr_38m_sft_candidate.pt` | 12 | 512 | Q:[512,512], K:[256,512], V:[256,512], O:[512,512] | G:[1408,512], U:[1408,512], D:[512,1408] | 37,499,392 | 39,596,544 | YES (Tied) |
| **NR052 17M Base** | `experiments/pretraining_nr052/chakr_17m_v4_base.pt` | 10 | 384 | Q:[384,384], K:[96,384], V:[96,384], O:[384,384] | G:[1024,384], U:[1024,384], D:[384,1024] | 17,063,808 | 18,636,672 | YES (Tied) |
| **NR052 38M Base** | `experiments/pretraining_nr052/chakr_38m_v4_base.pt` | 12 | 512 | Q:[512,512], K:[256,512], V:[256,512], O:[512,512] | G:[1408,512], U:[1408,512], D:[512,1408] | 37,499,392 | 39,596,544 | YES (Tied) |
| **NR052 17M SFT** | `experiments/pretraining_nr052/chakr_17m_nr052_sft.pt` | 10 | 384 | Q:[384,384], K:[96,384], V:[96,384], O:[384,384] | G:[1024,384], U:[1024,384], D:[384,1024] | 17,063,808 | 18,636,672 | YES (Tied) |
| **NR052 38M SFT** | `experiments/pretraining_nr052/chakr_38m_nr052_sft.pt` | 12 | 512 | Q:[512,512], K:[256,512], V:[256,512], O:[512,512] | G:[1408,512], U:[1408,512], D:[512,1408] | 37,499,392 | 39,596,544 | YES (Tied) |

---

## 4. Parameter Accounting Forensic Breakdown

### 17M Architecture Breakdown:
- Embedding: $4,096 \times 384 = 1,572,864$
- Transformer Block (per layer):
  - Attention:
    - $W_q: 384 \times 384 = 147,456$
    - $W_k: 96 \times 384 = 36,864$
    - $W_v: 96 \times 384 = 36,864$
    - $W_o: 384 \times 384 = 147,456$
    - Subtotal Attention: $368,640$
  - Feed-Forward (SwiGLU):
    - $W_{\text{gate}}: 1024 \times 384 = 393,216$
    - $W_{\text{up}}: 1024 \times 384 = 393,216$
    - $W_{\text{down}}: 384 \times 1024 = 393,216$
    - Subtotal MLP: $1,179,648$
  - Norms: 2 RMSNorms $\times 384 = 768$
  - Layer Total: $1,549,056$
- 10 Layers Total: $15,490,560$
- Final RMSNorm: $384$
- **Total Unique Trainable Parameters:** $1,572,864 + 15,490,560 + 384 = \mathbf{17,063,808}$

### 38M Architecture Breakdown:
- Embedding: $4,096 \times 512 = 2,097,152$
- Transformer Block (per layer):
  - Attention:
    - $W_q: 512 \times 512 = 262,144$
    - $W_k: 256 \times 512 = 131,072$
    - $W_v: 256 \times 512 = 131,072$
    - $W_o: 512 \times 512 = 262,144$
    - Subtotal Attention: $786,432$
  - Feed-Forward (SwiGLU):
    - $W_{\text{gate}}: 1408 \times 512 = 720,896$
    - $W_{\text{up}}: 1408 \times 512 = 720,896$
    - $W_{\text{down}}: 512 \times 1408 = 720,896$
    - Subtotal MLP: $2,162,688$
  - Norms: 2 RMSNorms $\times 512 = 1,024$
  - Layer Total: $2,950,144$
- 12 Layers Total: $35,401,728$
- Final RMSNorm: $512$
- **Total Unique Trainable Parameters:** $2,097,152 + 35,401,728 + 512 = \mathbf{37,499,392}$

---

## 5. Audit Conclusion

The parameter counts and architecture parameters across all checkpoints are verified with 100% precision. There is zero architectural ambiguity remaining in the repository.
