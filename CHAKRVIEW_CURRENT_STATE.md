# CHAKRVIEW CURRENT STATE

**Current Phase:** Mission NR-058A Completed (Controlled Training Pilot: Corpus V6 | 17M vs 38M)  
**Status:** 100-STEP PILOTS COMPLETED & VERIFIED (PASS / PASS)  
**Active Release Candidate Checkpoint (Frozen):** `release/v1.0.0-rc1/chakr_1_0_rc1_best.pt` (SHA-256 `cf585ebf9aa70f630516eec9c45e85f127e29ed91d6b6ae6a26d44fd20fcce4e`)  
**RC2 Candidate Checkpoint (Frozen):** `release/v1.0.0-rc2/chakr_1_0_rc2_best.pt` (SHA-256 `33f93c75ba6c4ccd06a87ddb68ad300e11ce0f67aef45d6aa0a6ae7868c2be30`)  
**RC3 Milestone Candidate Checkpoint (Frozen):** `experiments/fast_track_047_049/chakr_1_0_rc3_best.pt` (SHA-256 `61b39ae2bc55c116ed3ec8164497ec798d1fb6994a449d852248b266cd9dc73c`)  
**NR-058A Pilot Checkpoint 17M (100 steps):** `experiments/pilot_nr058a/chakr_chakrgenesis17m_pilot_100step.pt` (SHA-256 `81acd27c7992c15c15d29d17107db03ecdbaef334a61c0d3f1f0f59f6701c6f3`)  
**NR-058A Pilot Checkpoint 38M (100 steps):** `experiments/pilot_nr058a/chakr_chakrgenesis38m_pilot_100step.pt` (SHA-256 `bf37d46f926a8337c77c2308e4b0a6e761fe20f1f2c2fa61e2d6626db256a743`)  
**Active Tokenizer:** Canonical Byte-Level BPE 4096 (`data/experiments/vocab_4096`, Vocab SHA-256 `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a`)  
**Active Pretraining Dataset:** `Corpus V6` (4,573,231 clean tokens; Train: 4,163,555 tokens; Val: 191,686 tokens; Test: 217,990 tokens)  
**Active Hardware:** Intel Core i9-13900H (14C/20T) | 34GB RAM | NVIDIA RTX A1000 6GB Laptop GPU | CUDA 12.6 | PyTorch 2.14.1+cu126  
**Latest Git State:** Working tree updated with NR-058A controlled pilot deliverables, tracking `origin/exp/cbr-01-conversational-breakthrough` on GitHub.  

### Core Scientific Conclusions from NR-058A:
1. **Pilot Execution Flawless:** Both models executed their 100-step pilots on Corpus V6 Train under identical seeds, sequences, and AdamW optimizer settings, exhibiting zero NaNs, zero infs, and smooth gradient norms (<1.0).
2. **Loss Scaling Verified:** ChakrGenesis38M demonstrated faster validation loss decay ($\Delta = -2.3295$ vs $-2.2184$) and reached lower final validation loss ($6.0940$ vs $6.1671$) and lower perplexity ($443.19$ vs $476.80$) compared to ChakrGenesis17M after matched exposure (~318k tokens).
3. **Throughput Comparison:** 17M achieved higher throughput (**23,491.8 tokens/sec**, latency 126.5 ms) than 38M (**16,093.6 tokens/sec**, latency 176.6 ms) at sequence length 256.
4. **Memory Headroom Confirmed:** Peak VRAM was **1.491 GB** for 17M and **2.057 GB** for 38M, confirming substantial headroom on the 6.00 GB GPU for larger batch sizes or sequence lengths up to 512.
5. **No Full-Scale Training Executed:** The pilot was strictly bounded to 100 steps. No benchmark test data was accessed, and no claims of semantic reasoning are asserted.
