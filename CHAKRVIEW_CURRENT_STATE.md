# CHAKRVIEW CURRENT STATE

**Current Phase:** Mission NR-053 Completed (Generalization, Data Leakage & Overfitting Forensic Validation)  
**Status:** COMPLETED & FORENSICALLY VALIDATED  
**Active Release Candidate Checkpoint (Frozen):** `release/v1.0.0-rc1/chakr_1_0_rc1_best.pt` (SHA-256 `CF585EBF9AA70F630516EEC9C45E85F127E29ED91D6B6AE6A26D44FD20FCCE4E`)  
**RC2 Candidate Checkpoint (Frozen):** `experiments/genesis_stage_rc2_046/chakr_1_0_rc2_best.pt` (SHA-256 `33F93C75BA6C4CCD06A87DDB68AD300E11CE0F67AEF45D6AA0A6AE7868C2BE30`)  
**RC3 Milestone Candidate Checkpoint (Frozen):** `experiments/fast_track_047_049/chakr_1_0_rc3_best.pt` (SHA-256 `61B39AE2BC55C116ED3EC8164497EC798D1FB6994A449D852248B266CD9DC73C`)  
**NR-052 Pretrained Base 17M:** `experiments/pretraining_nr052/chakr_17m_v4_base.pt` (SHA-256 `58781710bb4d85b40d77281eb6ccecb1754a3f3fadc468db8c1fa8b51a2dd062`)  
**NR-052 Pretrained Base 38M:** `experiments/pretraining_nr052/chakr_38m_v4_base.pt` (SHA-256 `7fb29b3519a957dbb4ffc12bdcc1cc50d6ea9bf49739458ebb9e3de031f504fb`)  
**NR-052 SFT Candidate 17M:** `experiments/pretraining_nr052/chakr_17m_nr052_sft.pt` (SHA-256 `c13211bea859ab6ff8cb89c749b29ae57df638a85f2bef60908e23827fa5a4bb`)  
**NR-052 SFT Candidate 38M:** `experiments/pretraining_nr052/chakr_38m_nr052_sft.pt` (SHA-256 `e08d363efdaac44cc62fa854ad1802de8ffb9de4060e090ad5373d66174c53ed`)  
**Active Tokenizer:** Canonical Byte-Level BPE 4096 (`data/experiments/vocab_4096`, Vocab SHA-256 `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a`)  
**Active Hardware:** Intel Core i9-13900H (14C/20T) | 32GB RAM | NVIDIA RTX A1000 6GB Laptop GPU | CUDA 12.6 | PyTorch 2.14.1+cu126  
**Latest Git State:** Working tree updated with NR-053 forensic artifacts, tracking `origin/exp/cbr-01-conversational-breakthrough` on GitHub.  

### Core Forensic Conclusions from NR-053:
1. **Mathematical Metric Validity:** Verified that validation loss weighting is strictly token-level and $\text{PPL} = \exp(\text{Validation Loss})$ holds with 100% precision ($\Delta = 0.0000$).
2. **Corpus V4 Split Integrity:** Confirmed 0 document overlaps and 0 normalized text overlaps between train and val splits.
3. **Paraphrase Leakage Audit:** Discovered that factual recall hits on Canberra, New Delhi, and Mercury in NR-052 represent successful memorization and syntactic transfer of training facts from Corpus V3, where minor paraphrases existed with `(Ref #...)` suffixes.
4. **Independent Generalization Confirmed:** Evaluated on a brand-new, hand-crafted 50-prompt held-out benchmark with zero corpus overlap; NR-052 38M achieved the highest capability score (0.240 / 8.0%) with 100.0% syntactic coherence and 0.0% prompt echo, verifying genuine architectural capacity scaling.
