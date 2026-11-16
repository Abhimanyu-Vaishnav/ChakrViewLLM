# ChakrView Conversational Model v0.11

Welcome to the ChakrView v0.11 Conversational Release.

This directory contains the standalone weights, tokenizer merges/vocabulary, and configuration files for ChakrView v0.11.

## Quickstart

Run interactive chat using Python:
```bash
python -m chakrview chat --checkpoint releases/v0.11_conversational_model/chakrview_v011_model.pt
```

Run automated runtime verification:
```bash
python -m chakrview chat --test-mode --checkpoint releases/v0.11_conversational_model/chakrview_v011_model.pt
```

## Package Manifest
- `chakrview_v011_model.pt`: Model weights (SHA-256: `5ae98b369b08373df1222364d1b6165493b34287baf297ba32320c16b76f2754`)
- `config.json`: Architecture configuration (ChakrMicro v0.1: 6 layers, 192 dim, 6 heads, 4096 vocab)
- `vocab.json` & `merges.json`: Chakr-BPE tokenizer vocabulary and merge tables
- `MODEL_CARD.md`: Comprehensive capability scorecard, benchmark results, and scientific audit details

