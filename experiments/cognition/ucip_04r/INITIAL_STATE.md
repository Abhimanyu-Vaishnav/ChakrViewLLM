# UCIP-04R: Initial State Baseline Record

**Date:** 2026-11-16  
**Branch:** `exp/cbr-01-conversational-breakthrough`  
**Base Commit:** `f533741`  
**Task:** Architecture-Aware Checkpoint Loading & Medium Runtime Repair

---

## 1. Verified Cryptographic Hashes

| File | SHA-256 Checksum | Verification Status |
|---|---|---|
| `chakrview/runtime/session.py` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | MATCH (Frozen) |
| `releases/ucip_04_candidate/chakrview_ucip04_model.pt` | `234d04d598b6ff2f6c4514d9b973c51233872c118ad97e8fb6e60dec79d68bd6` | MATCH (Micro Candidate) |
| `releases/ucip_04_candidate/chakrview_ucip04_medium_model.pt` | `694c5888479bbd0a90d1ca116787c14e2e5362a6da0dfe782ea7304a55bd0693` | MATCH (Medium Candidate) |

---

## 2. Checkpoint Inspection Details

### Micro Checkpoint (`chakrview_ucip04_model.pt`)
- **Metadata config dict:**
  - `vocab_size`: 4096
  - `d_model`: 192
  - `n_layers`: 6
  - `n_heads`: 6
  - `hidden_dim`: 512
  - `max_seq_len`: 512
- **Parameters:** 3,443,136
- **Architecture Name:** ChakrMicro

### Medium Checkpoint (`chakrview_ucip04_medium_model.pt`)
- **Metadata config dict:**
  - `vocab_size`: 4096
  - `d_model`: 384
  - `n_layers`: 7
  - `n_heads`: 6
  - `hidden_dim`: 896
  - `max_seq_len`: 512
- **Parameters:** 12,932,736
- **Architecture Name:** ChakrMedium

---

## 3. Failure Mode Diagnostic

When running:
```powershell
.\.venv\Scripts\python.exe -m chakrview chat --checkpoint releases/ucip_04_candidate/chakrview_ucip04_medium_model.pt --memory off
```

The runtime threw:
```
RuntimeError: Error(s) in loading state_dict for ChakrMicro:
    size mismatch for embedding.weight: copying a param with shape torch.Size([4096, 384]) from checkpoint, the shape in current model is torch.Size([4096, 192]).
    ...
```

**Root cause:**
`DirectNeuralRunner` in `scripts/run_m3_evaluations.py` hardcoded `self.model = ChakrMicro(ModelConfig())` without checking the checkpoint's internal `config` dictionary or inspecting tensor dimensions. In addition, `ConversationalChatCLI.print_info` in `chakrview/chat.py` had hardcoded text stating `ChakrMicro v0.1: 6 layers, 192 dim, 6 heads`.
