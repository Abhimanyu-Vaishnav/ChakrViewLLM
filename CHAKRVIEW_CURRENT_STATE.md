# CHAKRVIEW CURRENT STATE

**Current Phase:** Mission NR-057-R1 Completed (Corpus Expansion Completion & Training Gate Revalidation)  
**Status:** CORPUS V6 EXPANDED (4.57M TOKENS) & FULL TRAINING GATES REVALIDATED  
**Active Release Candidate Checkpoint (Frozen):** `release/v1.0.0-rc1/chakr_1_0_rc1_best.pt` (SHA-256 `cf585ebf9aa70f630516eec9c45e85f127e29ed91d6b6ae6a26d44fd20fcce4e`)  
**RC2 Candidate Checkpoint (Frozen):** `release/v1.0.0-rc2/chakr_1_0_rc2_best.pt` (SHA-256 `33f93c75ba6c4ccd06a87ddb68ad300e11ce0f67aef45d6aa0a6ae7868c2be30`)  
**RC3 Milestone Candidate Checkpoint (Frozen):** `experiments/fast_track_047_049/chakr_1_0_rc3_best.pt` (SHA-256 `61b39ae2bc55c116ed3ec8164497ec798d1fb6994a449d852248b266cd9dc73c`)  
**NR-052 Pretrained Base 17M:** `experiments/pretraining_nr052/chakr_17m_v4_base.pt` (SHA-256 `58781710bb4d85b40d77281eb6ccecb1754a3f3fadc468db8c1fa8b51a2dd062`)  
**NR-052 Pretrained Base 38M:** `experiments/pretraining_nr052/chakr_38m_v4_base.pt` (SHA-256 `7fb29b3519a957dbb4ffc12bdcc1cc50d6ea9bf49739458ebb9e3de031f504fb`)  
**NR-052 SFT Candidate 17M:** `experiments/pretraining_nr052/chakr_17m_nr052_sft.pt` (SHA-256 `c13211bea859ab6ff8cb89c749b29ae57df638a85f2bef60908e23827fa5a4bb`)  
**NR-052 SFT Candidate 38M:** `experiments/pretraining_nr052/chakr_38m_nr052_sft.pt` (SHA-256 `e08d363efdaac44cc62fa854ad1802de8ffb9de4060e090ad5373d66174c53ed`)  
**Active Tokenizer:** Canonical Byte-Level BPE 4096 (`data/experiments/vocab_4096`, Vocab SHA-256 `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a`)  
**Active Pretraining Dataset:** `Corpus V6` (4,573,231 clean tokens; Train: 4,163,555 tokens; Val: 191,686 tokens; Test: 217,990 tokens)  
**Active Hardware:** Intel Core i9-13900H (14C/20T) | 34GB RAM | NVIDIA RTX A1000 6GB Laptop GPU | CUDA 12.6 | PyTorch 2.14.1+cu126  
**Latest Git State:** Working tree updated with NR-057-R1 revalidated corpus and training profile deliverables, tracking `origin/exp/cbr-01-conversational-breakthrough` on GitHub.  

### Core Scientific Conclusions from NR-057-R1:
1. **Corpus V6 Expanded & Balanced:** Rebalanced domain skew from Corpus V5 to create `Corpus V6` (4,573,231 tokens across 10,935 unique documents). Increased Devanagari Hindi to 14.4% (660K tokens), expanded STEM expository text to 34.3% (1.57M tokens), and reduced Python code dominance from 45.1% to 35.1%.
2. **Benchmark Firewall V2 Enforced:** Intercepted and rejected 40 contaminated candidate documents matching Benchmark V3, V2, and Fresh 50 test phrases and 8-grams.
3. **Tokenizer Fidelity Confirmed:** 0.00% UNK across all 4.57M tokens, 100.0% round-trip fidelity, and 94.21% active vocabulary utilization.
4. **Decoupled Throughput Profiling:** Full optimizer step throughput measured at **26,039.6 tokens/sec** for 17M and **23,118.6 tokens/sec** for 38M. 1 full epoch over Corpus V6 Train takes 2.66 min (17M) and 3.00 min (38M). Peak VRAM for 38M is 3.565 GB, leaving 2.435 GB of headroom.
5. **Pretraining Authorization Granted:** All 10 gates passed. The project is fully prepared for controlled pretraining execution in **NR-058**.
