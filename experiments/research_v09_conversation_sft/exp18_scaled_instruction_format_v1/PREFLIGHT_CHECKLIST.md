# ChakrView Exp18 — Preflight Verification Checklist

**Experiment**: `exp18_scaled_instruction_format_v1`  
**Location**: `experiments/research_v09_conversation_sft/exp18_scaled_instruction_format_v1/PREFLIGHT_CHECKLIST.md`  
**Phase**: Preflight Verification & Governance Gate  
**Status**: DRAFT SPECIFICATION (To be executed in a dedicated, authorized preflight run)  

---

## 1. Preflight Verification Steps & Exact Commands

Before initiating any optimizer steps for Exp18, every verification check listed below must be executed and logged with exit codes:

### Check 1: Repository Test Suite & Regressions
Verify that all unit and regression tests pass cleanly without errors:
```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
.venv\Scripts\python.exe -m unittest tests/test_task_evaluator.py -v
.venv\Scripts\python.exe -m unittest discover -s experiments/research_v09_conversation_sft/exp17_instruction_sft_v1 -p "test_*.py"
```
*Pass Criteria*: 147/147 tests pass; exit code 0.

### Check 2: Protected File Integrity Audit
Verify that all 8 protected assets match their established SHA-256 hashes:
```powershell
.venv\Scripts\python.exe -c "
import hashlib, json
files = {
    'canonical_checkpoint': 'experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt',
    'tokenizer_vocab': 'data/experiments/vocab_4096/vocab.json',
    'tokenizer_merges': 'data/experiments/vocab_4096/merges.json',
    'protected_runtime': 'chakrview/runtime/session.py',
    'protected_model': 'chakrview/brain/model.py',
    'exp17_train': 'experiments/research_v09_conversation_sft/exp17_instruction_sft_v1/data/train_exp17_v1.jsonl',
    'exp17_val': 'experiments/research_v09_conversation_sft/exp17_instruction_sft_v1/data/val_exp17_v1.jsonl',
    'exp17_test': 'experiments/research_v09_conversation_sft/exp17_instruction_sft_v1/data/test_exp17_v1.jsonl'
}
expected = {
    'canonical_checkpoint': '96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee',
    'tokenizer_vocab': 'db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a',
    'tokenizer_merges': 'ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7',
    'protected_runtime': '9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67',
    'protected_model': '1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361',
    'exp17_train': '074fafe5c1dbcbd7467430cb6f5eccd2c11f2f8679ff27e13c0d38cc90bac181',
    'exp17_val': '15d9c942c01fb6afaf6280b7515d9e621ff8df7740be8010daa2be98841f4564',
    'exp17_test': 'be10afaa3957713cead5399c2b01e5ad2f4fb61d6355835432c7d6e0fdcc6430'
}
for k, path in files.items():
    with open(path, 'rb') as f:
        h = hashlib.sha256(f.read()).hexdigest()
    assert h == expected[k], f'Hash mismatch on {k}'
print('All 8 hashes verified successfully.')
"
```
*Pass Criteria*: All 8 hashes match expected; assertion passes.

### Check 3: Checkpoint Loading & Architecture Invariant
Verify that the base checkpoint loads into `ChakrMicro` without missing or unexpected keys:
```powershell
.venv\Scripts\python.exe -c "
import torch
from chakrview.brain.model import ChakrMicro, ModelConfig
config = ModelConfig(d_model=192, n_layers=6, n_heads=6, hidden_dim=512, max_seq_len=512, vocab_size=4096)
model = ChakrMicro(config)
ckpt = torch.load('experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt', map_location='cpu')
state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
model.load_state_dict(state_dict, strict=True)
print(f'Successfully loaded checkpoint. Parameter count: {sum(p.numel() for p in model.parameters())}')
"
```
*Pass Criteria*: Exactly 3,443,136 parameters loaded with `strict=True`.

### Check 4: Zero-Update Forward / Loss Smoke Check
Run a no-update forward and backward pass on dummy batch tokens to verify loss masking and finite gradient norms:
```powershell
.venv\Scripts\python.exe -c "
import torch
import torch.nn.functional as F
from chakrview.brain.model import ChakrMicro, ModelConfig
model = ChakrMicro(ModelConfig(d_model=192, n_layers=6, n_heads=6, hidden_dim=512, max_seq_len=512, vocab_size=4096))
inputs = torch.randint(0, 4096, (2, 32))
targets = torch.full((2, 32), -100, dtype=torch.long)
targets[:, 16:] = torch.randint(0, 4096, (2, 16))
logits = model(inputs)
loss = F.cross_entropy(logits.view(-1, 4096), targets.view(-1), ignore_index=-100)
loss.backward()
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
assert torch.isfinite(loss), 'Loss is not finite'
assert torch.isfinite(grad_norm), 'Grad norm is not finite'
print(f'Smoke check pass: loss={loss.item():.4f}, grad_norm={grad_norm.item():.4f}')
"
```
*Pass Criteria*: Clean forward-backward pass, finite loss, finite gradient norm.

### Check 5: Hardware & Memory Headroom Check
Verify that the host CPU has sufficient memory and storage:
- Available RAM: $\ge 4.0 \text{ GB}$ free.
- Available Disk: $\ge 1.0 \text{ GB}$ free on drive `D:`.

---

## 2. Sign-off Criteria
All five preflight checks must pass synchronously with raw logs recorded in a timestamped run directory before any training execution script is invoked.
