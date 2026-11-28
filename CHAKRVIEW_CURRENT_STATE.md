# CHAKRVIEW CURRENT STATE

**Current Phase:** Mission NR-058B Stage 1 Completed (1-Epoch Controlled Pretraining)  
**Status:** STAGE 1 VERIFIED — 1 FULL EPOCH EXECUTED ON CORPUS V6 FOR 17M AND 38M  
**Active Release Candidate Checkpoint (Frozen):** `release/v1.0.0-rc1/chakr_1_0_rc1_best.pt` (SHA-256 `cf585ebf9aa70f630516eec9c45e85f127e29ed91d6b6ae6a26d44fd20fcce4e`)  
**RC2 Candidate Checkpoint (Frozen):** `release/v1.0.0-rc2/chakr_1_0_rc2_best.pt` (SHA-256 `33f93c75ba6c4ccd06a87ddb68ad300e11ce0f67aef45d6aa0a6ae7868c2be30`)  
**RC3 Milestone Candidate Checkpoint (Frozen):** `experiments/fast_track_047_049/chakr_1_0_rc3_best.pt` (SHA-256 `61b39ae2bc55c116ed3ec8164497ec798d1fb6994a449d852248b266cd9dc73c`)  
**NR-058B Stage 1 Final Checkpoint 17M (1,305 steps):** `experiments/stage1_nr058b/chakr_chakrgenesis17m_stage1_final.pt` (SHA-256 `2584b9a486e8c1c0c9725e42cebda16d0240a25079967b179c1a592fb1e2f09b`)  
**NR-058B Stage 1 Final Checkpoint 38M (1,305 steps):** `experiments/stage1_nr058b/chakr_chakrgenesis38m_stage1_final.pt` (SHA-256 `92f2aa7487361ec112f1e40a8760993f167733a1943a32e50078338e77ecfcea`)  
**Active Tokenizer:** Canonical Byte-Level BPE 4096 (`data/experiments/vocab_4096`, Vocab SHA-256 `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a`)  
**Active Pretraining Dataset:** `Corpus V6` (4,573,231 clean tokens; Train: 4,163,555 manifest / 4,170,667 active tokens; Val: 191,132 active tokens; Test: 217,990 tokens)  
**Active Hardware:** Intel Core i9-13900H (14C/20T) | 34GB RAM | NVIDIA RTX A1000 6GB Laptop GPU | CUDA 12.6 | PyTorch 2.14.1+cu126  
**Latest Git State:** Working tree updated with Stage 1 pretraining checkpoints, domain validation analysis, and report [NR058B_STAGE1_FINAL_REPORT.md](file:///d:/Project/ChakrView/reports/NR058B_STAGE1_FINAL_REPORT.md), tracking `origin/exp/cbr-01-conversational-breakthrough` on GitHub.  

### Core Scientific Conclusions from NR-058B Stage 1:
1. **Convergence on Real Pretraining:** Exactly 1 complete epoch (1,305 steps) was executed for both models from scratch (`seed=42`). ChakrGenesis17M dropped validation loss from 8.3890 to **3.1004** (PPL: 22.21, Acc: 39.80%). ChakrGenesis38M dropped validation loss from 8.4240 to **2.9562** (PPL: 19.22, Acc: 41.84%).
2. **Definitive 38M Superiority:** ChakrGenesis38M outperformed ChakrGenesis17M on overall validation loss (-0.1442), perplexity (-2.99), and across every individual domain (Curated Foundation: 1.3026 vs 1.4870; Python: 2.8861 vs 3.0054; English STEM: 3.3348 vs 3.4396; Hindi Devanagari: 3.7700 vs 4.0257).
3. **Execution Efficiency:** Pretraining 1 epoch on RTX A1000 6GB took 3.25 minutes for 17M (21.3k tok/s) and 4.67 minutes for 38M (14.8k tok/s). Peak VRAM remained well below ceiling (1.491 GB for 17M, 2.057 GB for 38M).
4. **Zero Contamination & Isolation:** Canonical checkpoints (`RC1`, `RC2`, `RC3`), canonical tokenizer, Benchmark V3/V2, and test sets remained 100% untouched. All artifacts were isolated strictly to `experiments/stage1_nr058b/`.
