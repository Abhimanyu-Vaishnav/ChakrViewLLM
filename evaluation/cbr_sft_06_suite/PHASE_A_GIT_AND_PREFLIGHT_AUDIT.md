# CBR-SFT-06 Phase A: Repository and Git Preflight Audit

**Date:** 2026-11-04  
**Auditor:** Antigravity Autonomous Research Agent  
**Repository:** `D:\Project\ChakrView`  
**Active Branch:** `exp/cbr-01-conversational-breakthrough`  

---

## 1. Git Preflight Status

- **Current HEAD Commit:** `f12c5549915109ac426bb0a348a28e4bf76a1148`
- **Preceding Phase 1 Commit:** `0f0c0551e59bb72b2165f696aa12bf03926a62ef`
- **Current Branch:** `exp/cbr-01-conversational-breakthrough`
- **Status of Working Tree:** Unmodified tracked code files; untracked diagnostic evaluation caches and evaluation suites properly isolated.

### Commit Lineage (Most Recent First):
1. `f12c554` - `research(cbr-sft-05): complete controlled context-binding experiment`
2. `0f0c055` - `research(cbr-sft-05): add audited context-binding dataset and evaluation suite`
3. `257801b` - `feat(experiment): record CBR-SFT-03B focused conversational steering pilot and evaluation`
4. `38dbe23` - `docs(forensics): record CBR-SFT-03A first-token and generation drift forensic audit`
5. `77f70e2` - `feat(experiment): implement, train and independently evaluate CBR-SFT-02 unified chat scaffolding`

---

## 2. Protected Artifact Integrity Verification

| Target Path | Expected SHA-256 | Actual SHA-256 | Status |
| :--- | :--- | :--- | :---: |
| `chakrview/brain/model.py` | `1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361` | `1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361` | **PASS** |
| `chakrview/runtime/session.py` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **PASS** |
| Canonical Step 10k Training Ckpt | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **PASS** |
| CBR-SFT-02 Step 50 Ckpt | `592553c3e27b654997e8f5df61e94fe1de25bbf577391df6c1a87f974730bf9d` | `592553c3e27b654997e8f5df61e94fe1de25bbf577391df6c1a87f974730bf9d` | **PASS** |
| CBR-SFT-03B Step 200 Ckpt | `297dcc60f39cadd6bae9a5bd969b1a7d831845d7a67823b30c194a67ef75b5ac` | `297dcc60f39cadd6bae9a5bd969b1a7d831845d7a67823b30c194a67ef75b5ac` | **PASS** |
| CBR-SFT-03B Step 300 Ckpt | `e540bd2364113d7eaf271c3495d882921f4421e59659f19f06cd7d590f74710a` | `e540bd2364113d7eaf271c3495d882921f4421e59659f19f06cd7d590f74710a` | **PASS** |
| Parent CBR-SFT-04 Step 200 Ckpt | `327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153` | `327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153` | **PASS** |
| CBR-SFT-05 Step 100 Ckpt | `8653d0bbe1b7ba3ea7eebeedf37a2b6b2e1d5d44d67743c4bcb7b7b065d1985f` | `8653d0bbe1b7ba3ea7eebeedf37a2b6b2e1d5d44d67743c4bcb7b7b065d1985f` | **PASS** |
| CBR-SFT-05 Step 200 Ckpt | `34bc6a1200d41d8e1b3c58a0b15ff8e5fbf41d451e208c7e1d953a4a0cbbe40f` | `34bc6a1200d41d8e1b3c58a0b15ff8e5fbf41d451e208c7e1d953a4a0cbbe40f` | **PASS** |
| CBR-SFT-05 Step 300 Ckpt | `7dcb81dccd009f1f50834bac968b68c34541116ad0b912f4e705b110285f06ab` | `7dcb81dccd009f1f50834bac968b68c34541116ad0b912f4e705b110285f06ab` | **PASS** |

### Safety Invariants Check:
- Protected paths unmodified: **YES**
- Restricted directories untouched (`experiments/overnight_runs/`, `D:\AI\SmartProjectMemory`): **YES**
- No external model dependencies: **YES**
- Ready to proceed to Phase B (Experiment History & Lineage Reconstruction).
