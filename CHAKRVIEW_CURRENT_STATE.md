# CHAKRVIEW CURRENT STATE

**Current Phase:** Mission HF-05 Completed (Authorized Private Hugging Face Upload & Verification)  
**Status:** RELEASED PRIVATELY — EXPERIMENTAL RESEARCH PREVIEW (OUTCOME B)  
**Hugging Face Repository:** `AbhimanyuVaishnav/chakrview-conversational-alpha` (Commit `12d12f9cfad06373cc6b903156080367375826d3`, **PRIVATE**)  
**Active Release Candidate Checkpoint (Frozen):** `release/v1.0.0-rc1/chakr_1_0_rc1_best.pt` (SHA-256 `cf585ebf9aa70f630516eec9c45e85f127e29ed91d6b6ae6a26d44fd20fcce4e`)  
**RC2 Candidate Checkpoint (Frozen):** `release/v1.0.0-rc2/chakr_1_0_rc2_best.pt` (SHA-256 `33f93c75ba6c4ccd06a87ddb68ad300e11ce0f67aef45d6aa0a6ae7868c2be30`)  
**RC3 Milestone Candidate Checkpoint (Frozen):** `experiments/fast_track_047_049/chakr_1_0_rc3_best.pt` (SHA-256 `61b39ae2bc55c116ed3ec8164497ec798d1fb6994a449d852248b266cd9dc73c`)  
**Pretrained Foundation Checkpoint (38M, 1 Epoch):** `experiments/stage1_nr058b/chakr_chakrgenesis38m_stage1_final.pt` (SHA-256 `92f2aa7487361ec112f1e40a8760993f167733a1943a32e50078338e77ecfcea`)  
**Conversational Alpha Checkpoint (`v0.10.0-alpha.1`):** `experiments/cv24h_release/chakr_38m_conversational_sft.pt` (SHA-256 `e4beba3ae20f66eccc2cd06a45582087a059eb9117ed15b8ed9ab8e5500ff106`)  
**Staged Model Safetensors:** `artifacts/huggingface/chakrview-conversational-alpha/model.safetensors` (SHA-256 `88c7ecd817648356a3e399872682c20a59bebc02873f7af6ae7e795cf8f66724`)  
**Active Tokenizer:** Canonical Byte-Level BPE 4096 (`data/experiments/vocab_4096`, Vocab SHA-256 `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a`)  
**Active Hardware:** Intel Core i9-13900H (14C/20T) | 34GB RAM | NVIDIA RTX A1000 6GB Laptop GPU | CUDA 12.6 | PyTorch 2.14.1+cu126  
**Latest Git State:** Working tree updated with Mission HF-05 documentation report [HF05_PRIVATE_UPLOAD_FINAL_REPORT.md](file:///d:/Project/ChakrView/reports/HF05_PRIVATE_UPLOAD_FINAL_REPORT.md), tracking `origin/exp/cbr-01-conversational-breakthrough` on GitHub.  

### Core Scientific & Deployment Conclusions:
1. **Private Hugging Face Deployment:** Successfully uploaded all 17 standalone release artifacts (including `model.safetensors`, standalone architecture, tokenizer engine, and generator) to `AbhimanyuVaishnav/chakrview-conversational-alpha`.
2. **End-to-End Integrity Re-Verification:** Every file was freshly re-downloaded into an isolated directory and verified via SHA-256. 100% hash parity achieved against local staging manifests.
3. **Pure Neuro-Conversational Execution:** Verified that the model executes autoregressive generation without local repository dependencies across CPU and GPU hardware.
4. **Visibility Governance:** The repository remains strictly **PRIVATE** in accordance with project constraints.
