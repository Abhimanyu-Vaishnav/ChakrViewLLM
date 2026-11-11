import os
import sys
import hashlib
from pathlib import Path

REPO_ROOT = Path("D:/Project/ChakrView")
sys.path.insert(0, str(REPO_ROOT))

import torch
from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

CKPT_PATH = REPO_ROOT / "releases/v0.9.1_conversational_preview/chakrview_v091_model.pt"
TOKENIZER_DIR = REPO_ROOT / "data/experiments/vocab_4096"

def audit_checkpoint():
    assert CKPT_PATH.exists(), f"Release checkpoint missing: {CKPT_PATH}"
    
    # 1. SHA-256 verification
    h = hashlib.sha256(CKPT_PATH.read_bytes()).hexdigest()
    expected_hash = "f161ccdc501bca34f6ed2019252f2e51d18fa113ce0cc90187c4ff0cdd180d10"
    assert h == expected_hash, f"Hash mismatch! Actual: {h}"
    
    # 2. PyTorch Load with safe globals
    torch.serialization.add_safe_globals([ModelConfig])
    payload = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=True)
    print(f"Payload keys: {list(payload.keys())}")
    
    state_dict = payload.get("model_state_dict", payload)
    
    # 3. Model instantiation from config
    cfg = ModelConfig(
        vocab_size=4096,
        d_model=192,
        n_layers=6,
        n_heads=6,
        hidden_dim=512,
        max_seq_len=512
    )
    model = ChakrMicro(cfg)
    
    # 4. Load state dict
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    print(f"Missing keys: {missing}, Unexpected keys: {unexpected}")
    assert len(missing) == 0 and len(unexpected) == 0, "Strict state dict load failed!"
    
    # 5. Parameter count
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameter count: {n_params}")
    assert n_params == 3443136, f"Expected 3,443,136 params, got {n_params}"
    
    # 6. Tokenizer audit
    tok, _ = load_tokenizer_artifacts(str(TOKENIZER_DIR))
    assert tok.vocab_size == 4096, f"Expected vocab 4096, got {tok.vocab_size}"
    bos_tok = tok.encode("", add_bos=True, add_eos=False)[0]
    eos_tok = tok.encode("", add_bos=False, add_eos=True)[0]
    
    # 7. CPU Inference test
    model.eval()
    test_prompt = "User: Hello!\nAssistant:"
    toks = tok.encode(test_prompt, add_bos=True, add_eos=False)
    inp = torch.tensor([toks], dtype=torch.long)
    
    with torch.no_grad():
        out1 = model(inp)
        out2 = model(inp)
        logits_diff = torch.max(torch.abs(out1 - out2)).item()
        assert logits_diff == 0.0, "Model forward pass is non-deterministic!"
        
    print("CPU forward pass successful and bit-exact deterministic.")
    
    report_md = f"""# CR-EXP-17R: Phase C — Independent Checkpoint Loading Report

## 1. Checkpoint Identification
- **Path:** `{CKPT_PATH}`
- **File Size:** {os.path.getsize(CKPT_PATH):,} bytes
- **SHA-256:** `{h}`
- **Verification Status:** **PASS (Exact Bit Match)**

## 2. Checkpoint Metadata in Payload
- **Step:** {payload.get('step')}
- **Best Step:** {payload.get('best_step')}
- **Validation Loss:** {payload.get('val_loss')}
- **Validation Token Accuracy:** {payload.get('val_acc') * 100:.2f}%
- **Parent Checkpoint:** `{payload.get('parent_checkpoint')}`
- **Parent SHA-256:** `{payload.get('parent_sha256')}`

## 3. Structural & Architectural Verification
- **Architecture:** `ChakrMicro v0.1`
- **Total Parameters:** {n_params:,} (100% match with canonical budget: 3,443,136)
- **Layer Count ($L$):** 6
- **Hidden Dimension ($d$):** 192
- **Attention Heads ($H$):** 6
- **MLP Dimension ($d_{{\\text{{mlp}}}}$):** 512
- **Context Ceiling ($T$):** 512
- **State Dict Load Mode:** `strict=True`
- **Missing Keys:** 0
- **Unexpected Keys:** 0

## 4. Tokenizer Integration
- **Tokenizer Vocab Size:** {tok.vocab_size}
- **Special Token Mapping:**
  - `BOS Token ID`: {bos_tok}
  - `EOS Token ID`: {eos_tok}

## 5. Execution Platform & Determinism
- **Device Tested:** Native Host CPU
- **Forward Determinism:** Verified (Max absolute logits discrepancy across consecutive forward passes = 0.000000).
- **External Dependencies:** Zero external API or pretrained backends. Self-contained indigenous architecture.
"""
    with open(REPO_ROOT / "experiments/cognition/cr_exp_17r/audit/CHECKPOINT_LOADING_REPORT.md", "w", encoding="utf-8") as f:
        f.write(report_md)
    print("CHECKPOINT_LOADING_REPORT.md created successfully.")

if __name__ == "__main__":
    audit_checkpoint()
