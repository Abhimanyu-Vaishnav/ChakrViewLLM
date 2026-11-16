# ChakrView UCIP-03 Conversation Candidate Release

## Overview
This directory contains the experimental conversation candidate produced during mission **UCIP-03**.

## Files
- `chakrview_ucip03_model.pt`: Candidate model weights (SHA-256: `b27313593e2b323781d65feeb8d4d7318fbfabe2eb53f56230bdf126c23f20c6`)
- `config.json`: ChakrMicro architectural configuration (3,443,136 parameters)
- `vocab.json`: 4,096-token vocabulary mapping
- `merges.json`: BPE merge table
- `MODEL_CARD.md`: Detailed performance metrics, lineage, and behavioral boundaries

## Quick Start CLI
To chat with this candidate using Hybrid Memory (Recommended):
```bash
python -m chakrview chat --checkpoint releases/ucip_03_conversation_candidate/chakrview_ucip03_model.pt --memory on
```

To test raw neural parametric generation without memory augmentation:
```bash
python -m chakrview chat --checkpoint releases/ucip_03_conversation_candidate/chakrview_ucip03_model.pt --memory off --debug-prompt
```
