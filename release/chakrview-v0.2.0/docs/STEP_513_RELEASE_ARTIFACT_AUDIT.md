# Wave 513: Release Artifact Audit

## 1. Executive Summary
As mandated by the Wave 512–519 Master Execution directives, an exhaustive audit of all release artifacts intended for the ChakrView v0.2.0 release candidate was conducted.
A deterministic inventory has been established, cataloging the type, size, role, provenance, and SHA-256 digest of each constituent artifact.
Temporary training experiment outputs, checkpoints outside the release scope, and scratch logs are isolated and excluded from the public bundle.

## 2. Release Artifact Inventory

| Relative Path | Type | Size (Bytes) | SHA-256 Digest | Role / Purpose | Provenance | Required |
|---|---|---|---|---|---|---|
| `chakrview/__init__.py` | Python Package Root | 749 | `163de224cdcb5c7d0d0f2525164cf3ff2406dd9514e21a2c9bfd9bc574880c59` | Package export & metadata | v0.2.0 Release | Yes |
| `chakrview/brain/model.py` | Core Neural Baseline | 7,014 | `de0cc0b2f254f3b7c89f5bc3a6a9be5fc3e129185a676b7017646ba416c141d6` | Canonical Baseline Transformer | Wave 449 Baseline | Yes |
| `chakrview/brain/config.py` | Config Definition | 3,468 | `ba609d488a0b0ca653556eb1bc20a1ceec4ce19a7ee5f4fa6e0a811c79878dc9` | Model hyperparameters & specs | Wave 449 Baseline | Yes |
| `chakrview/cognition/unified_compositional_core.py` | Cognitive Architecture | 13,938 | `77432d522f13ff03d987daaa64a75878473a25ef140cfa9a28bf2aa886bfe230` | Compositional reasoning engine | Wave 466–472 | Yes |
| `chakrview/cognition/adaptive_recurrent_reasoning_core.py` | Cognitive Architecture | 17,990 | `b8ee77f54f0fc5f992a7fa7bf8379f8fa1761664d9526732959dd82470e8893d` | I5 multi-hop recursive reasoning | Wave 480–487 | Yes |
| `chakrview/cognition/neural_routing_gateway.py` | Neural Gateway | 2,506 | `fbe53c61d482d3f9b2f4f228cbdf27092c7304123dc10b1a03a743472d73fc06` | Learned Language↔Cognition routing | Wave 473–479 | Yes |
| `chakrview/cognition/metacognitive_evaluator.py` | Metacognitive Engine | 4,162 | `dabbb2d757abbc590ca157f44a86b194d2ba05f32fa60ea8dbdbfaae15b3c373` | Confidence evaluation & strategy | Wave 496–503 | Yes |
| `chakrview/cognition/governed_learning/reasoning_experience.py` | Governed RIL | 2,770 | `f7124fd2bb1a8f9026210f03848b525f38515c1e9e0486c8d7e63bda6c13ff09` | Structured episodic experience | Wave 488–495 | Yes |
| `chakrview/runtime/unified_cognitive_runtime.py` | Unified Runtime | 8,033 | `b4efdc48b27e9e7bdf1cf1a2d52ef13ea0c86ee7a2f58ff79c8828905391500f` | End-to-end cognitive execution | Wave 504–511 | Yes |
| `chakrview/tokenizer/tokenizer.py` | Tokenizer Code | 3,680 | `d7e6cff106f654b4260a9f5d3767c9c059cbe354be412bcf5aa4ca124e4d5673` | Byte-level BPE tokenizer | Wave 449 Baseline | Yes |
| `data/tokenizer_experiments/v4096/vocab.json` | Tokenizer Data | 104,007 | `4fd17f5f5e6bcaaf8ea82b7936fc3fae9ef1823eb55ae639a0937a0c10d32c5f` | 4096 vocabulary tokens | Wave 449 Baseline | Yes |
| `data/tokenizer_experiments/v4096/merges.json` | Tokenizer Data | 76,149 | `b24a4edfbeb82fe4379ba9d9241517cc074bc933ecfa0683050436a53fe1db5e` | BPE merges definition | Wave 449 Baseline | Yes |
| `scripts/verify_release.py` | Release Verifier | 5,851 | `180f3154d625539fa086f6d6f5194fc2a7585ab6222f7b8dfa91b2bf13b5bf0b` | Canonical release checker | Wave 449 Baseline | Yes |
| `LICENSE` | Legal / Distribution | 10,767 | `842afcc90a60423985d2040de21f471b0db75f1025a7a7dbe888e23f958f2766` | Apache License Version 2.0 | Wave 513 Hardening | Yes |
| `README.md` | Documentation | 9,608 | `5cb5cd629b5e52ea354f0a004eb7e31e5bb49bb1e457f005d5ea66e2c39db84a` | Primary project specification | Repository Root | Yes |

## 3. Isolation of Non-Release Artifacts
- Checkpoints from isolated candidate trials (`candidate_*.pt`, `ril_trial_*.pt`) remain in `artifacts/` and are explicitly excluded from production release packaging to avoid ambiguity with the canonical baseline.
- Ephemeral logs and debug traces are sequestered.
- Canonical baseline parameters: 3,443,136. Canonical baseline SHA-256: `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`.

**Wave 513 Status: PASS**
