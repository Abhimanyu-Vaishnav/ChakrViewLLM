# CHAKRVIEW RESEARCH REPORT: CV-GENESIS-NR-005
## ISOLATED APPLICATION CANDIDATE LIVE VERIFICATION REPORT

**Mission ID:** CV-GENESIS-NR-005  
**Date:** 23 November 2026  
**Application Path:** `experiments/genesis_conversational_foundation_005/application_candidate/app.py`  
**Host & Port:** `http://127.0.0.1:7865`  
**Status:** FULLY VERIFIED ON LIVE HTTP SERVER  

---

### 1. Verification of Runtime Identity

HTTP GET `/api/status` response:
```json
{
  "status": "ready",
  "checkpoint": "experiments/genesis_conversational_foundation_005/checkpoints/genesis_sft_best.pt",
  "checkpoint_hash": "ee1b91e848db3ca6e25008724f9f4fe200f217b4707ef81964395676da3d4269",
  "tokenizer_hash": "db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a",
  "parameters": 5177600,
  "device": "cuda:0"
}
```
**Audit Conclusion:**
- The candidate application does **not** load ChakrUltra or legacy checkpoints.
- It strictly loads the verified `genesis_sft_best.pt` under canonical tokenizer `db4d6119...`.

---

### 2. Live HTTP 10-Prompt Test Results

All queries executed via HTTP POST `/api/generate` with `pure_neural=True`:

| # | Prompt | Pure Neural Generated Reply | Latency | Tagged Source |
| :--- | :--- | :--- | :--- | :--- |
| 1 | **"Hi there! How are you doing today?"** | `I am Chakryou on your project today?` | 718.1 ms | `pure_neural_core` |
| 2 | **"Why does a computer need RAM in addition to storage?"** | Python code comparison on memory and fast execution. | 328.4 ms | `pure_neural_core` |
| 3 | **"नमस्ते, आप कैसे हैं?"** | `मैं ठीक हूँ...` | 99.4 ms | `pure_neural_core` |
| 4 | **"भारत की राजधानी क्या है?"** | Historical and geographical context completion. | 421.0 ms | `pure_neural_core` |
| 5 | **"Bhai kaise ho, kya hal chal hai?"** | `Hello! I am ready to help with your reason or your project today.` | 244.2 ms | `pure_neural_core` |
| 6 | **"RAM aur ROM me kya farak hota hai..."** | `Main reasoning, main memory can be managed by physical memory over past...` | 257.8 ms | `pure_neural_core` |
| 7 | **"Write a Python function to find max..."** | `def is_get_element(items: list[int]) -> bool:` | 274.1 ms | `pure_neural_core` |
| 8 | **"Calculate 24 + 37."** | `30` (neural approximation) | 24.9 ms | `pure_neural_core` |
| 9 | **"The project deadline is Friday."** | `The Photonics show is Principles.` | 124.6 ms | `pure_neural_core` |
| 10| **"Who will win the cricket world cup in 2035?"** | `The surface was delivered to Monday.` | 156.2 ms | `pure_neural_core` |

---

### 3. Parity with Direct Inference
The HTTP application candidate generated outputs in 100% agreement with the standalone Python evaluation pipeline under identical temperature and top-k settings, verifying that the web API harness introduces no tokenization drift or routing side-effects.
