# ChakrView Alpha v0.1 Release Candidate

Welcome to **ChakrView Alpha v0.1**, an indigenous neural language model prototype developed from first principles.

---

## 1. Release Contents

The `releases/chakrview_alpha_v0.1` package provides the complete standalone runtime configuration:

| File | Purpose |
|---|---|
| `MODEL_CARD.md` | Authoritative architecture, pretraining lineage, evaluation, and limitation disclosure |
| `README.md` | Quickstart guide, installation, and usage instructions |
| `inference.py` | Local CPU autoregressive text generation script |
| `smoke_test.py` | Verification script ensuring bit-exact loading and model execution |
| `vocab.json` | Chakr-BPE 4,096 vocabulary definition |
| `merges.json` | Chakr-BPE subword merge rules (724 Devanagari merges) |
| `config.json` | Model configuration parameters |
| `RELEASE_MANIFEST.json` | Cryptographic SHA-256 manifest of all release files |

---

## 2. Requirements & Installation

ChakrView Alpha v0.1 is designed for **100% CPU execution** with minimal overhead (<500MB RAM).

### Dependencies
- Python 3.10+ (tested on Python 3.14)
- PyTorch >= 2.0.0 (CPU or CUDA)

```bash
pip install torch
```

---

## 3. Quickstart Inference

Run standalone inference using the pre-trained base model checkpoint:

```bash
python releases/chakrview_alpha_v0.1/inference.py \
  --checkpoint experiments/ucip13/checkpoints/primary_seed42/checkpoint_step_800.pt \
  --prompt "User: What is machine learning?\nAssistant:" \
  --max-new-tokens 48 \
  --temperature 0.0
```

### Interactive Smoke Test
Verify installation integrity and forward-pass execution:
```bash
python releases/chakrview_alpha_v0.1/smoke_test.py
```

---

## 4. Operational Boundaries & Status

> [!IMPORTANT]
> The current candidate checkpoint (`checkpoint_step_800.pt`) is the **UCIP-13 foundational pre-trained base model** (3.44M parameters, validation loss 4.6859). It has completed autoregressive pretraining on 6.55M token exposures, but has **NOT yet undergone instruction fine-tuning (SFT)**. 
> 
> As a pure pre-trained base model, greedy generation on conversational prompts will display pre-training language completions or repetitive suffix loops rather than aligned dialogue assistant behavior. **Instruction-aligned conversational capability will be activated in UCIP-15 SFT training.**
