# UCIP-04: Git Release Record

**Mission ID:** UCIP-04  
**Project:** ChakrView  
**Repository:** `D:\Project\ChakrView`  
**Active Branch:** `exp/cbr-01-conversational-breakthrough`  
**Starting Commit:** `51714cd`  
**Delivered Candidate Checkpoints:**
- `releases/ucip_04_candidate/chakrview_ucip04_model.pt` (ChakrMicro 3.44M, SHA-256: `234d04d598b6ff2f6c4514d9b973c51233872c118ad97e8fb6e60dec79d68bd6`)
- `releases/ucip_04_candidate/chakrview_ucip04_medium_model.pt` (ChakrMedium 12.9M, SHA-256: `694c5888479bbd0a90d1ca116787c14e2e5362a6da0dfe782ea7304a55bd0693`)

---

## 1. Tracked Mission Artifacts
- `experiments/cognition/ucip_04/` (Audit, Data, Checkpoints, Evaluation, Results, Final Report)
- `releases/ucip_04_candidate/` (Candidate Checkpoints, Tokenizer, Config, Model Card, README, Owner Test Guide, Feedback Template)

---

## 2. Invariant & Safety Record
- `chakrview/runtime/session.py` verified unmodified (`9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67`).
- All regression and chat runtime test suites pass.
- No historical checkpoints or release tags were altered.
