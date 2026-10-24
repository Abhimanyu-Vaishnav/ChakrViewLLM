# ChakrView Exp16 — Dataset Generation Report

**Date**: 2026-10-22
**Status**: `DATASETS COMPILED & INTEGRITY VERIFIED`

## 1. Summary of Generated Splits

- **Train Split**: 300 examples (Abstentions: 132, 44.0%), SHA-256: `55abc82154801da753f07bf1cf0ab1226ac5f07fc2506550d2c69b02d3372804`
- **Validation Split**: 60 examples (Abstentions: 24, 40.0%), SHA-256: `569a0091223aca34a84606a99ac60416b1efe3b0fdddf4d72b174b1b854c59a7`
- **Held-Out Test Split**: 120 examples (Abstentions: 50, 41.7%), SHA-256: `48de3477de8e50ba605165c093f2e329f16edf734ecdd4899b2c02e18e4885ad`
- Total Examples: **480**

## 2. Structural Characteristics

- All splits utilize the verified fail-closed character-to-token span aligner with 0 silent fallbacks.
- Variable preamble lengths (15 to 90 tokens) prevent fixed positional shortcuts.
- Multi-entity distractor candidate pools prevent naive category-to-type matching.
- Distinct entity namespaces (`_tr_`, `_va_`, `_te_`) strictly isolate train, val, and test.
