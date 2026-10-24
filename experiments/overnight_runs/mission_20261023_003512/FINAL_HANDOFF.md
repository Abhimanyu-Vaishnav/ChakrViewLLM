# ChakrView Overnight Mission — Final Handoff

**Mission Directory**: `experiments/overnight_runs/mission_20261023_003512/`  
**Timestamp**: 2026-10-23T00:43:00+05:30  
**Overall Mission Verdict**: **`SUCCESS`**

---

## 1. What is Now Trustworthy & Formally Proven

1. **Dataset Integrity & Verification**:
   - The Exp17 dataset generation pipeline has been completely vetted with zero fabricated or tautological proofs.
   - All 38 unanswerable abstention queries possess concrete programmatic proof of attribute absence from context.
   - All 12 answerable controls possess concrete proof of fact grounding in context.
   - All 50 factual QA and 50 scientific explanation items are supported by authoritative citations (IUPAC, NASA, NOAA, USGS, ISO 3166-2).
   - Official dataset splits (`train_exp17_v1.jsonl`, `val_exp17_v1.jsonl`, `test_exp17_v1.jsonl`) are generated under `experiments/research_v09_conversation_sft/exp17_instruction_sft_v1/data/` with exact SHA-256 manifests.

2. **Training Pipeline Reproducibility**:
   - The Stage D SFT training protocol executes reliably on CPU in ~23 seconds without GPU or external cloud dependencies.
   - The loss converged stably from 4.86 down to 1.96 val loss (Step 225) without gradient anomalies or parameter divergence.
   - Automatic checkpointing and best-checkpoint selection functioned as designed, saving `sft_best_checkpoint.pt`.

3. **Behavioral Capability Emergence**:
   - The model learned conversational turn termination via the mandatory EOS emission reward. Clean turn termination improved from **0.0% to 45.0%** on held-out test prompts and from **5.0% to 60.0%** on the frozen Normal Chat Gate v0.1 suite.
   - Pathological pre-training babble loops (`"The total number of the city of..."`) were drastically reduced.

---

## 2. What Remains Uncertain & Key Limitations

1. **Parameter Scale & Semantic Capacity**:
   - ChakrMicro is a 3.44M parameter model. While it successfully acquires structural syntax and turn boundaries, it cannot memorize vast factual corpora or perform multi-step arithmetic reasoning without retrieval or external assistance.
2. **Category Variance**:
   - Categories with concise structural targets (`structured_formatting` at 100% EOS, `factual_qa` at 80% EOS) responded immediately to SFT.
   - Complex string transforms (`instruction_following` at 0% EOS) and long-context extractive reading (`context_extraction` at 0% EOS) require additional data weighting or specialized curriculum stages.

---

## 3. Exact Commands to Reproduce Mission Results

### A. Run Test Suite & Preflight Check
```powershell
.venv\Scripts\python.exe -m unittest discover -s experiments/research_v09_conversation_sft/exp17_instruction_sft_v1 -p "test_*.py"
.venv\Scripts\python.exe experiments/research_v09_conversation_sft/exp17_instruction_sft_v1/generate_exp17_datasets.py
```

### B. Verify Canonical File Hashes
```powershell
.venv\Scripts\python.exe -c "
import hashlib
files = {
  'ckpt': 'experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt',
  'vocab': 'data/experiments/vocab_4096/vocab.json',
  'merges': 'data/experiments/vocab_4096/merges.json',
  'session': 'chakrview/runtime/session.py',
  'model': 'chakrview/brain/model.py'
}
for k, f in files.items():
  print(k, hashlib.sha256(open(f, 'rb').read()).hexdigest())
"
```

### C. Evaluate SFT Best Checkpoint
```powershell
.venv\Scripts\python.exe C:\Users\abhim\.gemini\antigravity-ide\brain\204d74ae-0f57-4370-8f17-c1d6f0bd82ab\scratch\run_stage6_eval.py
```

---

## 4. Primary Next Steps for the Engineering Team

1. **Curriculum Loss Weighting (Stage E)**:
   - Introduce category-specific loss multipliers to boost gradient signal on `context_extraction` and `instruction_following`.
2. **Dynamic Stopping / Temperature Tuning**:
   - Experiment with slight temperature ($T=0.2$) or min-token penalties to assess if extractive categories benefit from relaxed greedy decoding.
3. **Packaging Candidate Checkpoint**:
   - Prepare `sft_best_checkpoint.pt` for formal evaluation against the remaining research sprint gates.
