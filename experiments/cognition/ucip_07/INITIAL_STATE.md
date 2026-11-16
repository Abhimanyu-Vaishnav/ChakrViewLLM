# UCIP-07 Preflight & Baseline Invariant Record

**Mission:** UCIP-07 — Neural Architecture Evolution & Intelligence Foundation  
**Repository:** `D:\Project\ChakrView`  
**Branch:** `exp/cbr-01-conversational-breakthrough`  
**Base Commit (HEAD):** `4de4a2fe1efd30dc4169b0aca4bec7bdecfcb931`  
**Execution Timestamp:** 2026-11-16T16:52:00+05:30  

---

## 1. Environment & Hardware Profile
- **Operating System:** Windows-11-10.0.26300-SP0
- **Python Version:** 3.14.7 (.venv at `D:\Project\ChakrView\.venv\Scripts\python.exe`)
- **PyTorch Version:** 2.14.0+cpu
- **Execution Target:** CPU-First
- **Physical CPU Cores:** 14 (Logical Threads: 20, 13th Gen Intel Core i9)
- **Total Physical RAM:** 31.64 GB (Available: ~16.59 GB)
- **CUDA Acceleration:** Not active on current PyTorch CPU build (RTX A1000 6GB available on host, preserving CPU-first local invariant).

---

## 2. Invariant & Lineage Verification (All SHA-256 Passed)

| Target Artifact | Expected SHA-256 | Actual SHA-256 | Status | Notes |
|---|---|---|:---:|---|
| **UCIP-06 Medium Candidate** | `be843bc2cbc08f3f8459b85d6eec2cdce534441ac5ccc797f0bc0d2be8aa1b72` | `be843bc2cbc08f3f8459b85d6eec2cdce534441ac5ccc797f0bc0d2be8aa1b72` | **MATCH** | 12.93M parameters |
| **UCIP-06 Micro Candidate** | `7706e19ba568946c527cc15d90dc7b50180b7c754eb2b5b6dd7a5cdb3c55fd71` | `7706e19ba568946c527cc15d90dc7b50180b7c754eb2b5b6dd7a5cdb3c55fd71` | **MATCH** | 3.44M parameters |
| **UCIP-05 Medium Parent** | `089335f838c3e9999600144edccbb1caf1a4a741f43f17d493625b457e90e234` | `089335f838c3e9999600144edccbb1caf1a4a741f43f17d493625b457e90e234` | **MATCH** | Historical baseline |
| **Canonical Stage B Pretrained** | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **MATCH** | Step 10,000 checkpoint |
| **Protected Session Runtime** | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **MATCH** | `chakrview/runtime/session.py` |
| **Canonical Architecture Init** | `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` | `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` | **MATCH** | `ChakrMicro` (Seed 42, $\Delta W = 0$) |

---

## 3. Active Tokenizer Profile
- **Canonical Vocab Path:** `data/experiments/vocab_4096`
- **Vocab Size:** 4,096 tokens
- **Merges:** 3,840 BPE merges
- **Special Tokens:** `BOS_ID=0`, `EOS_ID=1`, `PAD_ID=2`
- **Encoding Mode:** Byte-level BPE with UTF-8 byte fallback
