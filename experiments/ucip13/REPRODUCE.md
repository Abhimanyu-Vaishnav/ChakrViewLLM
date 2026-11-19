# UCIP-13 Reproduction Guide
**Mission:** ChakrMicro Foundational Neural Pretraining  
**Repository:** `D:\Project\ChakrView`  
**Branch:** `exp/cbr-01-conversational-breakthrough`  
**Execution Environment:** Windows / Python 3.14.7 (`.venv`) / CPU-first

---

## 1. Quick Verification Commands

### Step 1: Verify Protected Hash Invariants
Verify that `chakrview/runtime/session.py` has not been modified ($\Delta W = 0$):
```powershell
Get-FileHash -Algorithm SHA256 chakrview/runtime/session.py
# Expected SHA256: 9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67
```

### Step 2: Verify Initial Seed 42 Model Architecture Hash
Verify that ChakrMicro deterministically initializes from random seed 42 to the exact canonical state:
```powershell
.venv\Scripts\python.exe -c "import torch; from chakrview.model.micro import ChakrMicro, ChakrMicroConfig; torch.manual_seed(42); m = ChakrMicro(ChakrMicroConfig()); import hashlib; print(hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in m.parameters())).hexdigest())"
# Expected SHA256: c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da
```

### Step 3: Run Full Automated Test Suite
Execute the 6 comprehensive regression and validation unit tests:
```powershell
.venv\Scripts\python.exe -m pytest experiments/ucip13/tests/test_ucip13_training.py -v
# Expected: 6 passed in ~2s
```

---

## 2. Checkpoint Lineage & State Verification

The pretraining run generated 7 checkpoints at steps 0, 50, 100, 200, 400, 600, and 800.
Their file hashes and validation metrics are recorded in `experiments/ucip13/checkpoints/checkpoint_manifest.json`.

To inspect the primary step 800 final checkpoint:
```powershell
.venv\Scripts\python.exe -c "import torch; ckpt = torch.load('experiments/ucip13/checkpoints/primary_seed42/checkpoint_step_800.pt', map_location='cpu', weights_only=False); print('Step:', ckpt['step'], 'Val Loss:', ckpt['val_loss'], 'Val PPL:', ckpt['val_ppl'])"
# Expected Output: Step: 800 Val Loss: 4.6859 Val PPL: 108.41
```

---

## 3. Dataset Preflight & Evaluation Verification

### Evaluate Held-Out Test Shards:
The held-out test partition (`data/tokenized/stage_c/test/`, 3 shards, 591,872 tokens) was evaluated once strictly after training completed:
```powershell
# Raw results are recorded in:
Get-Content experiments/ucip13/evaluations/held_out_test_results.json
# Test Loss: 4.6595 | Test Perplexity: 105.59
```

### Inspect Generation Samples:
Inspect qualitative generation progression across the checkpoints:
```powershell
Get-Content experiments/ucip13/samples/samples_step_800.json
```

---

## 4. Re-Running Pretraining (Optional Full Reproduction)

To execute the full Stage C pretraining run (800 steps, 6.55M tokens, ~22 minutes on CPU):
```powershell
.venv\Scripts\python.exe experiments/ucip13/scripts/train_stage_c.py
```
Outputs, logs, metric curves, and checkpoints will be generated in `experiments/ucip13/`.
