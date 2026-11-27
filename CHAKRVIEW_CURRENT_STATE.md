# CHAKRVIEW CURRENT STATE

**Current Phase:** Mission NR-055 Completed (Architecture Forensics & Genuine Capability Benchmark V2)  
**Status:** COMPLETED & SCIENTIFICALLY RECONCILED  
**Active Release Candidate Checkpoint (Frozen):** `release/v1.0.0-rc1/chakr_1_0_rc1_best.pt` (SHA-256 `cf585ebf9aa70f630516eec9c45e85f127e29ed91d6b6ae6a26d44fd20fcce4e`)  
**RC2 Candidate Checkpoint (Frozen):** `release/v1.0.0-rc2/chakr_1_0_rc2_best.pt` (SHA-256 `33f93c75ba6c4ccd06a87ddb68ad300e11ce0f67aef45d6aa0a6ae7868c2be30`)  
**RC3 Milestone Candidate Checkpoint (Frozen):** `experiments/fast_track_047_049/chakr_1_0_rc3_best.pt` (SHA-256 `61b39ae2bc55c116ed3ec8164497ec798d1fb6994a449d852248b266cd9dc73c`)  
**NR-052 Pretrained Base 17M:** `experiments/pretraining_nr052/chakr_17m_v4_base.pt` (SHA-256 `58781710bb4d85b40d77281eb6ccecb1754a3f3fadc468db8c1fa8b51a2dd062`)  
**NR-052 Pretrained Base 38M:** `experiments/pretraining_nr052/chakr_38m_v4_base.pt` (SHA-256 `7fb29b3519a957dbb4ffc12bdcc1cc50d6ea9bf49739458ebb9e3de031f504fb`)  
**NR-052 SFT Candidate 17M:** `experiments/pretraining_nr052/chakr_17m_nr052_sft.pt` (SHA-256 `c13211bea859ab6ff8cb89c749b29ae57df638a85f2bef60908e23827fa5a4bb`)  
**NR-052 SFT Candidate 38M:** `experiments/pretraining_nr052/chakr_38m_nr052_sft.pt` (SHA-256 `e08d363efdaac44cc62fa854ad1802de8ffb9de4060e090ad5373d66174c53ed`)  
**Active Tokenizer:** Canonical Byte-Level BPE 4096 (`data/experiments/vocab_4096`, Vocab SHA-256 `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a`)  
**Active Hardware:** Intel Core i9-13900H (14C/20T) | 32GB RAM | NVIDIA RTX A1000 6GB Laptop GPU | CUDA 12.6 | PyTorch 2.14.1+cu126  
**Latest Git State:** Working tree updated with NR-055 architecture forensics and Benchmark V2 deliverables, tracking `origin/exp/cbr-01-conversational-breakthrough` on GitHub.  

### Core Scientific Conclusions from NR-055:
1. **Architecture Identity Reconciled:** Tensor-level audit confirms exactly two model families: `ChakrGenesis17M` (17,063,808 unique params: d=384, L=10, Q=12, KV=3, hidden=1024) and `ChakrGenesis38M` (37,499,392 unique params: d=512, L=12, Q=8, KV=4, hidden=1408). The 38M discrepancy citing d=768 was resolved as a documentation error.
2. **Model Loaders 100% Strict:** All 12 checkpoints load under strict tensor mapping with zero missing or unexpected keys.
3. **Benchmark V2 Operational:** Quarantined, independently authored 95-prompt multi-category evaluation suite (`data/evaluation/benchmark_v2/benchmark_v2_100.json`) paired with an adversarial boundary-enforcing scoring engine.
4. **Candidate Capability Hierarchy:** Evaluated across 5 models; `NR052_38M_SFT` achieved the highest score (raw mean 0.179 / 3.0, 6.0%), with statistically significant superiority over NR-051 candidates ($p < 0.05$). RC3 Baseline scored second (0.137 / 3.0, 4.6%).
5. **Reasoning Frontier:** SLMs at 17M–38M remain incapable of multi-step arithmetic, logic, or reading comprehension extraction; pretraining token budget (10.5M tokens) is identified as the primary capability bottleneck.
