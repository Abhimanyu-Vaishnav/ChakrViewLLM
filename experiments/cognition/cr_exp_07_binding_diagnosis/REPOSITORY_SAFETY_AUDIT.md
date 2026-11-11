# Repository Safety & Preflight Audit — CR-EXP-07

**Audit Date**: 2026-11-06  
**Auditor**: Antigravity Autonomous Scientific Diagnostic Agent  
**Target Repository**: `D:\Project\ChakrView`  
**Git Branch**: `exp/cbr-01-conversational-breakthrough`  
**Git HEAD**: `564472a42245a47a7bf6f78ca6410e523383d2ce`  
**Host Environment**: Windows 11 AMD64 (20 logical cores, 31.64 GB total RAM, 15.65 GB available), Python 3.14.7, PyTorch 2.14.0+cpu  

---

## 1. Verified Cryptographic Lineage

All checkpoint files, tokenizer artifacts, canonical baseline states, and runtime files were verified bit-exact against repository ground truth:

| Checkpoint / Artifact | Relative Path | Expected SHA-256 Digest | Status |
|---|---|---|---|
| **Frozen Conversational Parent** | `experiments/cbr01_sft/runs/cbr_sft_04/checkpoints/sft_checkpoint_step_200.pt` | `327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153` | **PASS (Match)** |
| **Conversational Pilot Candidate** | `experiments/cognition/autonomous_12h/conversational_pilot/checkpoints/conversational_pilot_best_val.pt` | `ea6ee570854f5df71cec09af20ff66ff55b46d9584382cba4d77c3dbae33b6fd` | **PASS (Match)** |
| **CR-EXP-06 Control Arm Best Val** | `experiments/cognition/cr_exp_06_variable_binding/control_arm/checkpoints/control_arm_best_val.pt` | `729d5812c1ec02d217ef8cc342453a008ab3046a550e627b16b341c4a458c21f` | **PASS (Match)** |
| **CR-EXP-06 Treatment Arm Best Val**| `experiments/cognition/cr_exp_06_variable_binding/treatment_arm/checkpoints/treatment_arm_best_val.pt` | `2d8cea93ba8dbe06e1c714d5563ad89b18600f7a4f766fbb7cfc563c947a9d90` | **PASS (Match)** |
| **Canonical Baseline Model Init** | `chakrview/brain/model.py` (seed=42) | `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` | **PASS ($\Delta W = 0$)** |
| **Tokenizer Vocab** | `data/experiments/vocab_4096/vocab.json` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | **PASS (Match)** |
| **Tokenizer Merges** | `data/experiments/vocab_4096/merges.json` | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | **PASS (Match)** |
| **Tokenizer Config** | `data/experiments/vocab_4096/config.json` | `0fca72b6a70e9ae812589199c661ef6be20cb9973f06353db125a1ec7264e589` | **PASS (Match)** |
| **Protected Runtime** | `chakrview/runtime/session.py` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **PASS (Untouched)** |

---

## 2. In-Memory Activation Hook Locations

Inspection of `chakrview/brain/model.py` confirms the tensor flow through `ChakrMicro` (3,443,136 parameters):
1. **Input Tokens $\to$ Embedding**: `x = model.embedding(input_ids)` (Shape: `[B, T, 192]`).
2. **Transformer Blocks (6 Layers)**:
   - `norm1_out = layer.norm_1(x)` (Pre-attention RMSNorm)
   - `attn_out = layer.attn(norm1_out)` (Multi-Head Self-Attention with RoPE, causal mask)
   - `res_post_attn = x + attn_out` (Post-attention residual stream)
   - `norm2_out = layer.norm_2(res_post_attn)` (Pre-FFN RMSNorm)
   - `ffn_out = layer.ffn(norm2_out)` (SwiGLU FFN: $192 \to 512 \to 192$)
   - `x = res_post_attn + ffn_out` (Post-FFN residual stream)
3. **Final Normalization $\to$ Logits**:
   - `final_norm = model.final_norm(x)` (RMSNorm)
   - `logits = model.lm_head(final_norm)` (Tied to embedding weights, Shape: `[B, T, 4096]`).

**Gate 0 Verdict**: **PASSED**. Identity verified; hook insertion points established.

