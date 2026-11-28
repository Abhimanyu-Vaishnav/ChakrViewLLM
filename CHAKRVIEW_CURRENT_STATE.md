# CHAKRVIEW CURRENT STATE

**Current Phase:** Mission NR-057 Completed (Clean Corpus Expansion & Training Readiness Engineering)  
**Status:** DATA EXPANDED & TRAINING READINESS VERIFIED  
**Active Release Candidate Checkpoint (Frozen):** `release/v1.0.0-rc1/chakr_1_0_rc1_best.pt` (SHA-256 `cf585ebf9aa70f630516eec9c45e85f127e29ed91d6b6ae6a26d44fd20fcce4e`)  
**RC2 Candidate Checkpoint (Frozen):** `release/v1.0.0-rc2/chakr_1_0_rc2_best.pt` (SHA-256 `33f93c75ba6c4ccd06a87ddb68ad300e11ce0f67aef45d6aa0a6ae7868c2be30`)  
**RC3 Milestone Candidate Checkpoint (Frozen):** `experiments/fast_track_047_049/chakr_1_0_rc3_best.pt` (SHA-256 `61b39ae2bc55c116ed3ec8164497ec798d1fb6994a449d852248b266cd9dc73c`)  
**NR-052 Pretrained Base 17M:** `experiments/pretraining_nr052/chakr_17m_v4_base.pt` (SHA-256 `58781710bb4d85b40d77281eb6ccecb1754a3f3fadc468db8c1fa8b51a2dd062`)  
**NR-052 Pretrained Base 38M:** `experiments/pretraining_nr052/chakr_38m_v4_base.pt` (SHA-256 `7fb29b3519a957dbb4ffc12bdcc1cc50d6ea9bf49739458ebb9e3de031f504fb`)  
**NR-052 SFT Candidate 17M:** `experiments/pretraining_nr052/chakr_17m_nr052_sft.pt` (SHA-256 `c13211bea859ab6ff8cb89c749b29ae57df638a85f2bef60908e23827fa5a4bb`)  
**NR-052 SFT Candidate 38M:** `experiments/pretraining_nr052/chakr_38m_nr052_sft.pt` (SHA-256 `e08d363efdaac44cc62fa854ad1802de8ffb9de4060e090ad5373d66174c53ed`)  
**Active Tokenizer:** Canonical Byte-Level BPE 4096 (`data/experiments/vocab_4096`, Vocab SHA-256 `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a`)  
**Active Pretraining Dataset:** `Corpus V5` (3,565,296 clean tokens; Train: 3,187,865 tokens; Val: 221,282 tokens; Test: 156,149 tokens)  
**Active Hardware:** Intel Core i9-13900H (14C/20T) | 34GB RAM | NVIDIA RTX A1000 6GB Laptop GPU | CUDA 12.6 | PyTorch 2.14.1+cu126  
**Latest Git State:** Working tree updated with NR-057 clean corpus expansion and training readiness deliverables, tracking `origin/exp/cbr-01-conversational-breakthrough` on GitHub.  

### Core Scientific Conclusions from NR-057:
1. **Corpus Expansion Achieved:** Expanded the clean verified pretraining pool from 624,068 tokens to **3,565,296 tokens** (5.71x expansion) across Python algorithmic code (42.4%), STEM/humanities expository text (23.5%), foundational language records (15.4%), and Hindi Devanagari literature/science (7.4%).
2. **Benchmark Contamination Firewall:** Built an active 8-gram firewall protecting 445 sealed benchmark test items across Benchmark V3, Benchmark V2, and NR-053 Fresh 50 (6,116 protected 8-grams). Intercepted 43 candidate documents during ingestion.
3. **Tokenizer Compatibility Proven:** Evaluated 221,282 tokens with the canonical BPE 4096 tokenizer; achieved **0.00% UNK rate** and **100.0% round-trip fidelity** across English, Hindi, and Python code.
4. **Hardware Performance & VRAM Headroom:** Benchmarked both architectures on the RTX A1000 6GB GPU:
   - 17M ($d=384, L=10$): **26,166.8 tokens/s**, peak VRAM **3.20 GB** (1 full epoch in 2.03 minutes).
   - 38M ($d=512, L=12$): **23,192.6 tokens/s**, peak VRAM **3.56 GB** (1 full epoch in 2.29 minutes).
5. **Training Readiness Verified:** All 8 scientific and architectural gates passed. The project is ready for controlled multi-seed neural pretraining in **NR-058**.
