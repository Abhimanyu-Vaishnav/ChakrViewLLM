# CHAKRVIEW RESEARCH REPORT: CV-GENESIS-NR-006
## APPLICATION CANDIDATE INDEPENDENT VERIFICATION REPORT

**Mission ID:** CV-GENESIS-NR-006  
**Date:** 23 November 2026  
**Application Endpoint:** `http://127.0.0.1:7866`  
**Application File:** `experiments/genesis_conversational_foundation_005/application_candidate/app.py`  
**Test Data:** `experiments/genesis_intelligence_validation_006/application/app_verification_results.json`  

---

### 1. Integrity Verification of Running Application

HTTP Diagnostic Audit GET `/api/status`:
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

- **Checkpoint Hash:** `ee1b91e848db3ca6e25008724f9f4fe200f217b4707ef81964395676da3d4269` (**100% match with SFT Best**)
- **Tokenizer Hash:** `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` (**100% match with Canonical Vocab**)
- **Architecture:** `ChakrGenesis` (5,177,600 parameters on `cuda:0`)

---

### 2. Live HTTP Verification

10 test prompts spanning English, Hindi, Hinglish, Code, Arithmetic, and Uncertainty were transmitted over HTTP POST `/api/generate`:
- **Average Generation Latency:** ~245 ms on NVIDIA RTX A1000 GPU
- **Pure Neural Routing:** 100% of responses generated through `pure_neural_core` with zero deterministic templates or external API calls.
- **Inference Consistency:** The HTTP candidate reproduces identical token probabilities and output text as the direct PyTorch benchmark harness.
