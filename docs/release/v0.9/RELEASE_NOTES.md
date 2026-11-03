# ChakrView v0.9.0 — Release Notes

**Version:** `v0.9.0`  
**Classification:** Indigenous Neural Language Model Research & Release Preview  
**Target Hardware:** CPU-first native execution (ARM / x86_64, zero external GPU requirement)  
**Baseline Model:** `ChakrMicro` (3,443,136 unique parameters, tied embeddings, 6 layers, 6 heads, width 192, MLP 512, RoPE)  
**Canonical Baseline Checkpoint:** `training_checkpoint_step_10000.pt` (SHA-256 `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`)  
**Release Date:** 2026-11-03  

---

## 1. Executive Summary & Project Identity

ChakrView is an indigenous, modular, CPU-first neural language model built from scratch. It is developed under strict scientific and engineering principles:
1. **Zero External Pretrained Weights**: No weights borrowed from LLaMA, Mistral, GPT, or Hugging Face hubs.
2. **Zero Hidden API Calls**: No hosted external LLM secretly powering model generation.
3. **100% CPU-First Native Execution**: Runs purely on consumer and server CPU cores without requiring CUDA or discrete GPUs.
4. **Epistemic Honesty**: Capabilities, failures, and mathematical limitations are verified and published transparently.

**ChakrView v0.9.0** consolidates months of empirical investigation across pretraining, vocabulary extraction, experimental fine-tuning, and conversational evaluation.

---

## 2. Key Milestones in v0.9.0

### A. Stage C Production Frequency Extraction Closure
- Successfully executed the authorized physical Stage C corpus extraction across all 27 training binary shards (`data/tokenized/stage_c/train/*.bin`), reconciling exactly **6,651,093 uint16 tokens** ($13,302,186$ bytes) in **0.217 seconds**.
- Formally resolved the 754 surviving candidate tokens and stratified the 450-token training pool ($E_{train}$) and 150-token held-out testing pool ($E_{test}$) with complete partition disjointness ($E_{train} \cap E_{test} = \emptyset$).
- Verified all 27 pre-read SHA-256 shard checksums with zero read errors and zero source mutations.

### B. Strict Canonical Authorization Gate
- Implemented and verified a tamper-evident authorization gate in `scripts/extract_stage_c_frequencies.py`.
- Eliminates path-spoofing by binding directly to the canonical repository path (`REPO_ROOT/docs/experiments/CAND_01_STAGE_C_EXTRACTION_AUTHORIZATION_DECISION.md`) and enforcing bit-exact SHA-256 verification (`e5b433adebb53b45e8250f07cae38282e1a06839bed15fa788d16d4db5192f00`).
- Rejects external copies, path traversal, symlink redirection, and environment variable bypasses.

### C. Conversational Bottleneck Diagnosis (M4 Findings)
- Completed forensic analysis explaining why microscopic models (<5M parameters) struggle with open-ended dialogue:
  1. **Dataset Semantic Collapse**: Early SFT mixtures repeated small numbers of fixed answers across varied prompts, causing the model to memorize static response strings rather than learning dynamic variable binding.
  2. **Evaluator False Positives**: Evaluators using loose substring matches masked true capability ceilings. Robust evaluations require word-boundary checks and isolated AST parsing.
  3. **Representation Capacity Ceiling**: A 3.44M parameter model trained on small synthetic corpora cannot hallucinate factual world knowledge absent from pretraining.

---

## 3. Verified Artifacts & Checksums

| Artifact | Canonical Path | SHA-256 Digest | Status |
| :--- | :--- | :--- | :---: |
| **Brain Model** | `chakrview/brain/model.py` | `1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361` | **IMMUTABLE** |
| **Runtime Session** | `chakrview/runtime/session.py` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **IMMUTABLE** |
| **Canonical Checkpoint**| `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **VERIFIED** |
| **Tokenizer Vocab** | `data/experiments/vocab_4096/vocab.json` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | **VERIFIED** |
| **Tokenizer Merges**| `data/experiments/vocab_4096/merges.json` | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | **VERIFIED** |
| **Decision Doc** | `docs/experiments/CAND_01_STAGE_C_EXTRACTION_AUTHORIZATION_DECISION.md` | `e5b433adebb53b45e8250f07cae38282e1a06839bed15fa788d16d4db5192f00` | **VERIFIED** |
