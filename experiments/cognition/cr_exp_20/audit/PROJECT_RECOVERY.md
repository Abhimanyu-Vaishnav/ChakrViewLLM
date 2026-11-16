# CR-EXP-20 — Technical Audit & Baseline Verification (Phase A)

**Mission ID:** CR-EXP-20  
**Phase:** Phase A — Project Recovery and Technical Audit  
**Date:** 2026-11-15  
**Investigator:** Antigravity Autonomous Agent  
**Repository:** `D:\Project\ChakrView`  
**Git Branch:** `exp/cbr-01-conversational-breakthrough`  

---

## 1. Hardware & Environment Audit

- **Operating System:** Windows 11 (AMD64)
- **CPU:** Intel Core i9-13900H (13th Gen, 14 cores / 20 threads)
- **RAM:** 32.0 GB Physical RAM
- **GPU:** NVIDIA RTX A1000 6GB Laptop GPU
- **PyTorch Installation:** `2.14.0+cpu`
- **Execution Mode:** CPU-first execution mode. Because PyTorch in `.venv` is compiled for CPU, all training and evaluation run directly on Intel Core i9 CPU with fast multi-threading (`torch.set_num_threads(14)`).

---

## 2. Canonical Artifact Cryptographic Audit

All parent artifacts required by Section 4 were independently audited via bit-exact SHA-256 verification:

| Subject | Path / Definition | Expected SHA-256 | Actual Measured SHA-256 | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Canonical Architecture Init** | `torch.manual_seed(42)` param buffer | `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` | `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` | **PASS (Bit-Exact)** |
| **Canonical Stage B Checkpoint** | `experiments/batch16_21_combined/...` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **PASS (Bit-Exact)** |
| **Conversational Parent** | `experiments/cbr01_sft/runs/cbr_sft_04/...` | `327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153` | `327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153` | **PASS (Bit-Exact)** |
| **v0.9.1 Release Checkpoint** | `releases/v0.9.1_conversational_preview/...`| `f161ccdc501bca34f6ed2019252f2e51d18fa113ce0cc90187c4ff0cdd180d10` | `f161ccdc501bca34f6ed2019252f2e51d18fa113ce0cc90187c4ff0cdd180d10` | **PASS (Bit-Exact)** |
| **Protected Runtime** | `chakrview/runtime/session.py` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **PASS (Bit-Exact)** |

Zero canonical artifacts have been modified or overwritten.

---

## 3. Audit of CR-19 Checkpoints and Findings

The 6 training runs from CR-EXP-19 were verified:
- `arm_a_seed101_best.pt`: `68ec27e1085b388616afaf6c2f05082cac5e9fac33e8278ecd6c05ff656984a8`
- `arm_a_seed202_best.pt`: `0347f40b85dc07e0ac8a397c77e6b7fda49ffac6a3ccc0a3da411666b71584b1`
- `arm_a_seed303_best.pt`: `e1d3ccaa05ee3d22d777a96d4679643a70c7eb223e64fa8e6fdc6eae745775d0`
- `arm_b_seed101_best.pt`: `9433ced65ce6af7cbf26b454ffcf956e2f3d1c5a6fd8ac3b582942a0c6e1afe1`
- `arm_b_seed202_best.pt`: `792604d8cd4d1fe4482f70cb65dea7fd8dd67407e0de3c54edc7377d7f9ccb30`
- `arm_b_seed303_best.pt`: `d6ed3342b7c22b4dbebfad34eb8575b5848c4b1b4aeae90024d33045f8dc0ed4`

### Verification of Runtime History Fix:
The dynamic token-budgeted history packing in `chakrview/chat.py` was executed through `test_runtime_history.py`:
- 8 out of 8 tests passed without failure.
- Context packing correctly accommodates up to 8 complete conversational turns within the 384-token prompt ceiling without turn truncation.

---

## 4. Root Diagnosis: Why CR-19 Achieved 0/200 on Retrieval

Inspection of CR-19 logs and generated outputs reveals the root cause:
1. **Casual Dialogue Supervision vs. Entity Attention:** CR-19 fine-tuned on casual chit-chat dialogues where the assistant responded with generic polite phrases. There was no explicit loss gradient penalizing the model for failing to attend to user-declared facts (names, locations, preferences).
2. **Tokenizer Fragmentation on Multilingual/Hindi:** The existing 4,096-token Chakr-BPE tokenizer splits Devanagari Hindi characters into raw UTF-8 byte tokens, severely bloating context and causing Hindi representations to fragment into 3–4 tokens per character.
3. **Absence of Targeted In-Context Query-Fact Grounding in Training:** A neural model does not magically develop associative memory from chat syntax alone; it requires explicit multi-turn queries that demand retrieval of previously stated state.

---

## 5. Phase A Sign-Off
Phase A is complete. All prerequisites are verified. Proceeding directly to Phase B (Tokenizer & Multilingual Representation).

