# CHAKRVIEW CURRENT STATE

**Current Phase:** Mission CV-24H Completed (Conversational Research Preview Alpha `v0.10.0-alpha.1`)  
**Status:** RELEASED — EXPERIMENTAL RESEARCH PREVIEW (OUTCOME B)  
**Active Release Candidate Checkpoint (Frozen):** `release/v1.0.0-rc1/chakr_1_0_rc1_best.pt` (SHA-256 `cf585ebf9aa70f630516eec9c45e85f127e29ed91d6b6ae6a26d44fd20fcce4e`)  
**RC2 Candidate Checkpoint (Frozen):** `release/v1.0.0-rc2/chakr_1_0_rc2_best.pt` (SHA-256 `33f93c75ba6c4ccd06a87ddb68ad300e11ce0f67aef45d6aa0a6ae7868c2be30`)  
**RC3 Milestone Candidate Checkpoint (Frozen):** `experiments/fast_track_047_049/chakr_1_0_rc3_best.pt` (SHA-256 `61b39ae2bc55c116ed3ec8164497ec798d1fb6994a449d852248b266cd9dc73c`)  
**Pretrained Foundation Checkpoint (38M, 1 Epoch):** `experiments/stage1_nr058b/chakr_chakrgenesis38m_stage1_final.pt` (SHA-256 `92f2aa7487361ec112f1e40a8760993f167733a1943a32e50078338e77ecfcea`)  
**Conversational Alpha Checkpoint (`v0.10.0-alpha.1`):** `experiments/cv24h_release/chakr_38m_conversational_sft.pt` (SHA-256 `e4beba3ae20f66eccc2cd06a45582087a059eb9117ed15b8ed9ab8e5500ff106`)  
**Active Tokenizer:** Canonical Byte-Level BPE 4096 (`data/experiments/vocab_4096`, Vocab SHA-256 `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a`)  
**Active Hardware:** Intel Core i9-13900H (14C/20T) | 34GB RAM | NVIDIA RTX A1000 6GB Laptop GPU | CUDA 12.6 | PyTorch 2.14.1+cu126  
**Latest Git State:** Working tree updated with Mission CV-24H release artifacts, model card, limitations, inference package, and report [CV24H_FINAL_REPORT.md](file:///d:/Project/ChakrView/reports/CV24H_FINAL_REPORT.md), tracking `origin/exp/cbr-01-conversational-breakthrough` on GitHub.  

### Core Scientific Conclusions from CV-24H:
1. **Uncontaminated Conversational SFT Mixture:** 1,612 dialogues (English 90%, Hindi 7%, Hinglish 3%) compiled with 75 contaminated items eliminated against sealed benchmark 8-grams.
2. **Response-Only Loss Masking:** Prompt tokens masked with ignore_index=-100; loss computed exclusively on assistant tokens. SFT validation loss improved from 4.1777 (27.04% accuracy) to 2.8924 (49.70% accuracy).
3. **Pure Neural Evaluation:** Evaluated across 12 distinct conversational dimensions. 100% UTF-8 valid, 0% repetition degeneration, greeting and Python syntax acquisition confirmed. Semantic drift and factual deficit honestly reported.
4. **Standalone Inference:** Delivered reproducible `ChakrViewGenerator` and interactive `chat_cli.py` with greedy, temperature, top-k, top-p, and repetition penalty controls.
5. **Release Classification:** Tagged `v0.10.0-alpha.1` as an Engineering Research Preview Alpha under Outcome B. Zero hidden rule fakes.
