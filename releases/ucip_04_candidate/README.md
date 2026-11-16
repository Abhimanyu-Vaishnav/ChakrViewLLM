# ChakrView UCIP-04 Autoregressive Language Candidate

## Overview
This directory contains the neural language acquisition candidate developed during mission **UCIP-04**.
It replaces broken proverb attractors (`"The planned of the bold"`) with genuine sentence representations.

## Checkpoint Files
- `chakrview_ucip04_model.pt`: Default ChakrMicro 3.44M candidate checkpoint (SHA-256: `234d04d598b6ff2f6c4514d9b973c51233872c118ad97e8fb6e60dec79d68bd6`)
- `chakrview_ucip04_medium_model.pt`: Expanded ChakrMedium 12.9M candidate checkpoint (SHA-256: `694c5888479bbd0a90d1ca116787c14e2e5362a6da0dfe782ea7304a55bd0693`)
- `config.json`: Model configuration parameters
- `vocab.json` & `merges.json`: Canonical tokenizer artifacts

## Quick Start CLI
Interactive chat with Hybrid Memory (Recommended):
```bash
python -m chakrview chat --checkpoint releases/ucip_04_candidate/chakrview_ucip04_model.pt --memory on
```

Testing direct neural parametric generation without memory:
```bash
python -m chakrview chat --checkpoint releases/ucip_04_candidate/chakrview_ucip04_model.pt --memory off --debug-prompt
```
