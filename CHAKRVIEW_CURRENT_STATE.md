# CHAKRVIEW CURRENT STATE

**Current Phase:** Mission NR-051 Completed (True Language Pretraining & Controlled Scaling Study)  
**Status:** COMPLETED & VALIDATED  
**Active Release Candidate Checkpoint (Frozen):** `release/v1.0.0-rc1/chakr_1_0_rc1_best.pt` (SHA-256 `CF585EBF9AA70F630516EEC9C45E85F127E29ED91D6B6AE6A26D44FD20FCCE4E`)  
**RC2 Candidate Checkpoint (Frozen):** `experiments/genesis_stage_rc2_046/chakr_1_0_rc2_best.pt` (SHA-256 `33F93C75BA6C4CCD06A87DDB68AD300E11CE0F67AEF45D6AA0A6AE7868C2BE30`)  
**RC3 Milestone Candidate Checkpoint (Frozen):** `experiments/fast_track_047_049/chakr_1_0_rc3_best.pt` (SHA-256 `61B39AE2BC55C116ED3EC8164497EC798D1FB6994A449D852248B266CD9DC73C`)  
**NR-051 Pretrained Base 17M:** `experiments/pretraining_nr051/chakr_17m_pretrained_base.pt` (SHA-256 `8d871ca0aff6e23398bdb17f86521eb26dfb4b778d227035d77e2fe44803c25f`)  
**NR-051 Pretrained Base 38M:** `experiments/pretraining_nr051/chakr_38m_pretrained_base.pt` (SHA-256 `4fd67a877e123646b51a1c70239e283763b816d6a4b6ab3391e0c8de6bb85f50`)  
**NR-051 SFT Candidate 17M:** `experiments/pretraining_nr051/chakr_17m_sft_candidate.pt` (SHA-256 `9f63e097108f4c3ef2fdc882826d570845a101d02f036bc1c055ad66183600b4`)  
**NR-051 SFT Candidate 38M:** `experiments/pretraining_nr051/chakr_38m_sft_candidate.pt` (SHA-256 `a0639a3fba90352c47b151fb70e60209a1889cd3265b80c3cadcb751e3c5b31f`)  
**Active Tokenizer:** Canonical Byte-Level BPE 4096 (`data/experiments/vocab_4096`, Vocab SHA-256 `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a`)  
**Active Hardware:** Intel Core i9-13900H (14C/20T) | 32GB RAM | NVIDIA RTX A1000 6GB Laptop GPU | CUDA 12.6 | PyTorch 2.14.1+cu126  
**Latest Git State:** Working tree updated with NR-051 artifacts, tracking `origin/exp/cbr-01-conversational-breakthrough` on GitHub.  
**Core Scientific Finding from NR-051:**
1. Causal pretraining from scratch on ~1.03M tokens restores standard scaling behavior: 38M achieves lower pretraining validation loss (2.8502 vs 3.0516) and lower subsequent SFT validation loss (2.2262 vs 2.4155) than 17M.
2. Under a constrained ~1M token budget, the 17M parameter model retains higher parameter saturation density, yielding better performance on narrow held-out semantic prompts (2.5% vs 1.9%).
3. Pretraining scaling laws dictate that models $\ge 38\text{M}$ parameters require $\ge 25\text{M}-50\text{M}$ tokens to unlock superior downstream factual generalization over the 17M architecture.
