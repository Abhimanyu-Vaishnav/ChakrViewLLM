# CHAKRVIEW CURRENT STATE

**Current Phase:** Mission NR-050 Completed (Controlled Architecture Scaling Study & Git Synchronization)  
**Status:** COMPLETED & SYNCHRONIZED  
**Active Release Candidate Checkpoint (Frozen):** `release/v1.0.0-rc1/chakr_1_0_rc1_best.pt` (SHA-256 `CF585EBF9AA70F630516EEC9C45E85F127E29ED91D6B6AE6A26D44FD20FCCE4E`)  
**RC2 Candidate Checkpoint (Frozen):** `experiments/genesis_stage_rc2_046/chakr_1_0_rc2_best.pt` (SHA-256 `33F93C75BA6C4CCD06A87DDB68AD300E11CE0F67AEF45D6AA0A6AE7868C2BE30`)  
**RC3 Milestone Candidate Checkpoint (Frozen):** `experiments/fast_track_047_049/chakr_1_0_rc3_best.pt` (SHA-256 `61B39AE2BC55C116ED3EC8164497EC798D1FB6994A449D852248B266CD9DC73C`)  
**Latest Scaling Experiment Checkpoint:** `experiments/scaling_038m_050/chakr_genesis_38m_best.pt` (SHA-256 `76240745A58D420F78C693424BE313E0CC9893E718536FF442F3A79F56CF6BCD`)  
**Active Tokenizer:** Canonical Byte-Level BPE 4096 (`data/experiments/vocab_4096`, Vocab SHA-256 `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a`)  
**Active Hardware:** Intel Core i9-13900H (14C/20T) | 32GB RAM | NVIDIA RTX A1000 6GB Laptop GPU | CUDA 12.6 | PyTorch 2.14.1+cu126  
**Latest Git State:** Clean working tree, fully synchronized with `origin/exp/cbr-01-conversational-breakthrough` on GitHub (`https://github.com/Abhimanyu-Vaishnav/ChakrViewLLM.git`).  
**Core Scientific Finding from NR-050:** Parameter scaling from 17.06M to 37.50M without a commensurate pretraining base does not improve conversational semantic capability (RC3 achieves 5.3% vs 38M's 2.2%). Scaling must begin with an extensive pretraining campaign before conversational SFT.  
**Recommended Next Step:** Design and launch the **ChakrGenesis-38M Pretraining Campaign** (pretraining base on large-scale tokenized corpora) before conversational fine-tuning, while deploying the **ChakrGenesis-17M RC3 + Neuro-Symbolic Runtime** for immediate local usage.
