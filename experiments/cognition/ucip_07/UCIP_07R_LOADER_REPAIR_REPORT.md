# UCIP-07R: Architecture-Aware Checkpoint Loading & Runtime Compatibility Repair Report

**Project:** ChakrView  
**Mission:** UCIP-07R — Architecture-Aware Checkpoint Loading & Runtime Compatibility Repair  
**Branch:** `exp/cbr-01-conversational-breakthrough`  
**Date:** November 17, 2026  
**Status:** Successfully Diagnosed, Repaired, Verified, and Released  

---

## 1. Problem Statement & Root Cause Analysis

### 1.1 The Owner-Reported Error
When the owner executed:
```powershell
.venv\Scripts\python.exe -m chakrview chat --checkpoint releases/ucip_07_candidates/chakrview_ucip07_medium_model.pt --memory off
```
The checkpoint SHA-256 verification succeeded:
`554c503e058311d8732216a3ad200cf1c7d60a98baa39f78c26328c66516d5b2`

However, loading failed catastrophically with:
```text
RuntimeError: Error(s) in loading state_dict for ChakrMicro:
Missing key(s) in state_dict: "lm_head.weight".
```

### 1.2 Forensic Findings & Root Cause
1. **Architecture Detection**:
   In `scripts/run_m3_evaluations.py` (`DirectNeuralRunner`), the configuration parser filtered keys from `ckpt["config"]` against `ModelConfig.__dataclass_fields__`. Because the checkpoint contained:
   ```python
   {'vocab_size': 4096, 'd_model': 384, 'n_layers': 7, 'n_heads': 6, 'hidden_dim': 896, 'max_seq_len': 512, 'rope_base': 50000.0, 'tie_embeddings': True}
   ```
   The loader correctly extracted `d_model=384`, `n_layers=7`, and identified the architecture as **`ChakrMedium`**. The runtime model class in ChakrView is `ChakrMicro` (which serves as the general parameterized transformer implementation for both Micro and Medium, defined in `chakrview/brain/model.py`). The error message `for ChakrMicro` reflected PyTorch's class name, not that it had selected the 3.44M configuration.
2. **Missing `lm_head.weight` Key**:
   ChakrView architectures utilize **strict weight tying** (`tie_embeddings=True`), where the vocabulary projection head shares the underlying parameter tensor with the input embedding:
   $$W_{\text{out}} = W_{\text{embed}}^T$$
   In `chakrview/brain/output.py`:
   ```python
   class LMHead(nn.Module):
       def __init__(self, embedding_weight: nn.Parameter) -> None:
           super().__init__()
           self.weight = embedding_weight
   ```
   In canonical checkpoints (such as UCIP-04, 05, and 06), PyTorch's `state_dict()` saved `lm_head.weight` as an alias pointing to the exact same tensor data pointer as `embedding.weight`.
   In `experiments/cognition/ucip_07/evolved_model.py`, the evolved model declared:
   ```python
   if config.tie_embeddings:
       self.lm_head = None
   ```
   and computed logits directly via `F.linear(x, self.embedding.weight)`. As a result, its serialized `state_dict` contained 65 tensors (including `embedding.weight`), omitting the duplicate `lm_head.weight` key.
3. **Strict Loading Collision**:
   When `DirectNeuralRunner` instantiated `ChakrMicro(self.cfg)` and invoked `model.load_state_dict(state_dict, strict=True)`, PyTorch raised a `Missing key: 'lm_head.weight'` error because `ChakrMicro`'s module structure includes `self.lm_head`.

---

## 2. Checkpoint Structure & Key Forensics

Programmatic inspection of `releases/ucip_07_candidates/chakrview_ucip07_medium_model.pt`:
- **Top-level Keys**: `['step', 'model_state_dict', 'val_loss', 'config']`
- **State Dict Keys Count**: 65 tensors
- **Embedding Key**: `'embedding.weight'` with shape `torch.Size([4096, 384])`
- **Output Key**: Intentionally omitted due to tied embeddings (`tie_embeddings: True`).
- **Layers**: 7 Transformer blocks (`layers.0` through `layers.6`), each containing:
  - `norm_1.weight` `[384]`
  - `attn.q_proj.weight` `[384, 384]`
  - `attn.k_proj.weight` `[384, 384]`
  - `attn.v_proj.weight` `[384, 384]`
  - `attn.out_proj.weight` `[384, 384]`
  - `norm_2.weight` `[384]`
  - `ffn.gate_proj.weight` `[896, 384]`
  - `ffn.up_proj.weight` `[896, 384]`
  - `ffn.down_proj.weight` `[384, 896]`
- **Final Norm**: `final_norm.weight` `[384]`
- **Total Unique Parameters**: **12,932,736**
- **Logit Equivalence**: The maximum absolute logit difference between `ChakrEvolvedLM` and `ChakrMicro` (with tied head) on identical inputs is **`0.0`** (mathematically bit-exact).

---

## 3. The Runtime Compatibility Repair

### 3.1 Repair Strategy
1. **Explicit Architecture Resolution**:
   - Extended configuration mapping in `scripts/run_m3_evaluations.py` to recognize `rope_base` $\to$ `rope_theta` and `norm_eps` $\to$ `rms_norm_eps`.
   - Explicitly map $d_{model}=192 \to \text{"ChakrMicro"}$, $d_{model}=384 \to \text{"ChakrMedium"}$, and $d_{model}=512 \to \text{"ChakrCore"}$.
2. **Strict Tied-Weight Reconstruction**:
   - When a checkpoint has `tie_embeddings=True` and contains `embedding.weight` of shape `(vocab_size, d_model)` while omitting `lm_head.weight`, `DirectNeuralRunner` explicitly aliases:
     ```python
     state_dict = state_dict.copy()
     state_dict["lm_head.weight"] = state_dict["embedding.weight"]
     ```
   - This satisfies PyTorch's `strict=True` loader without modifying any checkpoint weights on disk, without inventing parameters, and without using `strict=False`.
3. **Weight Pointer Identity Verification**:
   - Verified that `model.embedding.weight.data_ptr() == model.lm_head.weight.data_ptr()`.

### 3.2 Exact Code Diffs in `scripts/run_m3_evaluations.py`
```python
<<<<
        # Determine architecture configuration
        if "config" in ckpt and isinstance(ckpt["config"], dict):
            cfg_dict = ckpt["config"]
            valid_keys = set(ModelConfig.__dataclass_fields__.keys()) if hasattr(ModelConfig, "__dataclass_fields__") else set()
            filtered_cfg = {k: v for k, v in cfg_dict.items() if k in valid_keys} if valid_keys else cfg_dict
            self.cfg = ModelConfig(**filtered_cfg)
====
        # Determine architecture configuration
        if "config" in ckpt and isinstance(ckpt["config"], dict):
            cfg_dict = ckpt["config"].copy()
            # Map alternative naming conventions
            if "rope_base" in cfg_dict and "rope_theta" not in cfg_dict:
                cfg_dict["rope_theta"] = cfg_dict["rope_base"]
            if "norm_eps" in cfg_dict and "rms_norm_eps" not in cfg_dict:
                cfg_dict["rms_norm_eps"] = cfg_dict["norm_eps"]
            valid_keys = set(ModelConfig.__dataclass_fields__.keys()) if hasattr(ModelConfig, "__dataclass_fields__") else set()
            filtered_cfg = {k: v for k, v in cfg_dict.items() if k in valid_keys} if valid_keys else cfg_dict
            self.cfg = ModelConfig(**filtered_cfg)
>>>>
...
<<<<
        # Handle tied embedding / output head state dictionary key
        if isinstance(state_dict, dict):
            if self.cfg.tie_embeddings and "lm_head.weight" not in state_dict and "embedding.weight" in state_dict:
                emb_shape = state_dict["embedding.weight"].shape
                if emb_shape == (self.cfg.vocab_size, self.cfg.d_model):
                    state_dict = state_dict.copy()
                    state_dict["lm_head.weight"] = state_dict["embedding.weight"]
                else:
                    raise ValueError(
                        f"Tied embedding shape mismatch: expected ({self.cfg.vocab_size}, {self.cfg.d_model}), "
                        f"got {emb_shape}"
                    )
>>>>
```

---

## 4. Verification Evidence & Test Execution

### 4.1 Checkpoints Loaded with `strict=True`
- **UCIP-04 Micro** (3,443,136 params) $\implies$ PASS
- **UCIP-04 Medium** (12,932,736 params) $\implies$ PASS
- **UCIP-05 Medium** (12,932,736 params) $\implies$ PASS
- **UCIP-06 Medium** (12,932,736 params) $\implies$ PASS
- **UCIP-07 Medium** (12,932,736 params, $\Theta=50,000.0$) $\implies$ PASS
- **UCIP-07 Core** (27,795,968 params, $\Theta=50,000.0$) $\implies$ PASS

### 4.2 Automated Regression Suite
Executed:
```powershell
.\.venv\Scripts\pytest.exe tests/test_architecture_loader.py tests/test_inference_session.py tests/test_ucip05_regression.py tests/test_ucip07r_loader.py
```
**Results:** **25 passed in 23.82s (100% SUCCESS)**.

### 4.3 Interactive CLI Banner & `/info` Verification
Ran:
```powershell
.\.venv\Scripts\python.exe -m chakrview chat --checkpoint releases/ucip_07_candidates/chakrview_ucip07_medium_model.pt --memory off
```
Observed exact output:
```text
Loading ChakrView model weights from: chakrview_ucip07_medium_model.pt
Verified SHA-256: 554c503e058311d8732216a3ad200cf1c7d60a98baa39f78c26328c66516d5b2

=================================================================
  CHAKRVIEW CONVERSATIONAL TERMINAL (HYBRID NEURAL BRAIN)
=================================================================
Model: ChakrMedium (12,932,736 params: 7L, 384d) [chakrview_ucip07_medium_model.pt]
SHA-256: 554c503e058311d8...
Memory Mode: DISABLED (Pure Neural)
Commands: /reset (clear context), /memory (view facts), /info (view stats), /exit (quit)
Type your message and press Enter.

User > /info
--- ChakrView Session Info ---
Release:         v1.0 (Unified Conversational Intelligence)
Checkpoint:      chakrview_ucip07_medium_model.pt
SHA-256:         554c503e058311d8732216a3ad200cf1c7d60a98baa39f78c26328c66516d5b2
Architecture:    ChakrMedium (12,932,736 params: 7 layers, 384 dim, 6 heads, 896 ffn)
Memory Mode:     DISABLED (Pure Neural)
History Turns:   0 turns recorded
Active Facts:    N/A (Memory disabled)
Context Ceiling: 512 tokens
Prompt Budget:   384 tokens (Dynamic turn-budgeted retention)
Decoding:        temperature=0.0, repetition_penalty=1.15, no_repeat_ngram=3, max_new_tokens=48
```

---

## 5. Safety & Lineage Verification

- **Protected Runtime**:
  `chakrview/runtime/session.py` SHA-256:
  `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` (**100% BIT-EXACT UNMODIFIED**).
- **Candidate Checkpoints**:
  Unmodified and bit-exact to UCIP-07 release digests.
- **Distinction Notice**:
  This repair resolves runtime loading and architecture detection. It does not alter underlying checkpoint weights or claim newly acquired reasoning capabilities beyond those measured in the UCIP-07 evaluation report (31.37% on the 800-probe benchmark).
