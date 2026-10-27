# ChakrView Release Gates Specification

**Document Status**: Official Release Engineering Standard
**Authority**: Senior ML Infrastructure & Release Management

---

## 1. Overview of Release Gates

Before any version tag (`v0.X.0`) is published or code promoted to `main`, the release candidate must satisfy all 12 mandatory gates without exception.

```
       [Candidate Branch]
               |
  Gate 1: Canonical Hashes (Bit-exact match)
  Gate 2: Architecture & Parameter Invariance (3.44M fp32)
  Gate 3: Corpus Provenance & Shard Checksums (100% mapped)
  Gate 4: Deep Privacy & Secrets Scan (0 exposed)
  Gate 5: Epistemic Bounds & Negative Evidence (Exp18 rejections documented)
  Gate 6: Experimental Quarantine (Pointers/Previews isolated)
  Gate 7: GGUF Status Accuracy (Preview only)
  Gate 8: Smoke & Regression Test Suite (100% pass)
  Gate 9: Fresh-User & Installation Verification (CPU executable)
  Gate 10: Clean Uncommitted Work Triage (No stray commits)
  Gate 11: Release Notes & Manifest Alignment (Complete documentation)
  Gate 12: Anonymous Public Download Verification (Asset checksum match)
               |
               v
       [Release Published]
```

---

## 2. Gate Definitions & Pass Criteria

| Gate ID | Name | Pass Criteria | Verification Method |
| :--- | :--- | :--- | :--- |
| **GATE-01** | **Artifact Hashes** | SHA-256 for checkpoint, vocab, merges, model.py, session.py match expected reference values bit-exact. | Python hashlib preflight script |
| **GATE-02** | **Parameter Accounting** | Exactly 3,443,136 unique parameters with tied input/output embeddings accounted for. | PyTorch tensor inspection |
| **GATE-03** | **Corpus Provenance** | 100% of training documents attributed to legitimate open licenses (MIT, CC-BY-4.0, CC-BY-SA, PSF, GODL). Shards match manifest hashes. | Document manifest audit |
| **GATE-04** | **Privacy & Secrets** | Zero API keys, private tokens, passwords, personal data, or machine filepaths in metadata or history. | Deep regex scan across all strings and tensors |
| **GATE-05** | **Epistemic Honesty** | Bounded capabilities; negative experimental results (including 20/20 Exp18 rejections) fully reported without concealment. | Human review dossier inspection |
| **GATE-06** | **Architecture Quarantine** | Speculative or experimental extensions (e.g. pointer networks) quarantined from production inference. | Runtime import & dependency check |
| **GATE-07** | **GGUF Demarcation** | GGUF labeled as Preview/Prototype unless native C++ inference engine is fully integrated. | Code review |
| **GATE-08** | **Smoke Suite** | 100% pass on automated release smoke test (`tests/test_v060_release_smoke.py`). | `pytest tests/test_v060_release_smoke.py -v` |
| **GATE-09** | **Fresh-User Setup** | Standalone installation and greedy generation reproducible on commodity CPU without errors. | Clean environment execution |
| **GATE-10** | **Clean Work Triage** | All relevant source, tests, and documentation staged; local scratch excluded from Git history. | `git status --porcelain` audit |
| **GATE-11** | **Documentation** | Release notes, manifest, installation guide, known limitations, and provenance docs complete and linked. | Markdown link verification |
| **GATE-12** | **Public Download** | Release asset uploaded to GitHub Release and verified via anonymous unauthenticated HTTPS download. | Unauthenticated curl / urllib fetch |
