# ChakrView Conversation Alpha Release Candidate Report

**Project:** ChakrView  
**Mission ID:** UCIP-15  
**Release Target:** `releases/chakrview_conversation_alpha/`  
**Candidate Checkpoint:** `chakrview_conversation_alpha.pt`  
**Checkpoint SHA-256:** `004ca38f3e796c4590bcfd73a029a3493173637f22bcf0dcad41e98c218511e7`  
**Status:** RELEASE CANDIDATE PREPARED & CPU-VERIFIED  

---

## 1. Release Deliverables

The release package is packaged in `releases/chakrview_conversation_alpha/`:

| Artifact | Size | SHA-256 Checksum | Purpose |
|---|---:|---|---|
| `chakrview_conversation_alpha.pt` | 41,388,435 | `004ca38f3e796c4590bcfd73a029a3493173637f22bcf0dcad41e98c218511e7` | SFT-tuned conversational weights |
| `MODEL_CARD.md` | 1,847 | `e430da8a50993fe94f54e1dc3ea043fa6e1a49f572ae7e7ca46757f4955a12ef` | Authoritative model specification |
| `README.md` | 1,846 | `97d620570b556b26cf8d43dc84fe576082ec05fbcc71d6aa0d59247c4033fe92` | Quickstart & CPU running guide |
| `inference.py` | 4,204 | `91ea2829a8a72a9df273ca85a3a0e41712a3d0ae697d983ae4fef8e7ea418293` | Standalone dialogue CLI script |
| `smoke_test.py` | 2,125 | `d557ccae25381a1796d8e2ea5c517865ec0ea5d471df42f36f6aa979b0bfdc26` | Verification test script |
| `vocab.json` | 112,627 | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | Chakr-BPE 4,096 vocab |
| `merges.json` | 76,299 | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | Subword BPE merge rules |
| `config.json` | 1,130 | `0fca72b6a70e9ae812589199c661ef6be20cb9973f06353db125a1ec7264e589` | Model configuration |
| `RELEASE_MANIFEST.json` | 1,132 | `adabeb0fe86a11e83e954c2a5c53102287e07661005ca77f2409ae01828f7d98` | Release file manifest |

---

## 2. Verification & Smoke Test
Running `python releases/chakrview_conversation_alpha/smoke_test.py`:
- [PASS] Conversation checkpoint verified (`004ca38f...`).
- [PASS] Tokenizer round-trip & vocab verified (4,096).
- [PASS] Model instantiated with 3,443,136 parameters.
- [PASS] Conversational forward pass verified.
- **Result: 100% Operational on CPU.**

---

## 3. Remote Git Status
- Remote push: **NOT PERFORMED** (strictly local commits).
- Canonical baseline & runtime session: **UNMODIFIED** ($\Delta W = 0$).
