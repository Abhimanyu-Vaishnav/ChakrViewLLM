# CHAKRVIEW CURRENT STATE

**Current Phase:** Mission NR-056 Completed (Benchmark Integrity, Statistical Replication & Architecture Lineage Audit)  
**Status:** COMPLETED & FORENSICALLY RECONCILED  
**Active Release Candidate Checkpoint (Frozen):** `release/v1.0.0-rc1/chakr_1_0_rc1_best.pt` (SHA-256 `cf585ebf9aa70f630516eec9c45e85f127e29ed91d6b6ae6a26d44fd20fcce4e`)  
**RC2 Candidate Checkpoint (Frozen):** `release/v1.0.0-rc2/chakr_1_0_rc2_best.pt` (SHA-256 `33f93c75ba6c4ccd06a87ddb68ad300e11ce0f67aef45d6aa0a6ae7868c2be30`)  
**RC3 Milestone Candidate Checkpoint (Frozen):** `experiments/fast_track_047_049/chakr_1_0_rc3_best.pt` (SHA-256 `61b39ae2bc55c116ed3ec8164497ec798d1fb6994a449d852248b266cd9dc73c`)  
**NR-052 Pretrained Base 17M:** `experiments/pretraining_nr052/chakr_17m_v4_base.pt` (SHA-256 `58781710bb4d85b40d77281eb6ccecb1754a3f3fadc468db8c1fa8b51a2dd062`)  
**NR-052 Pretrained Base 38M:** `experiments/pretraining_nr052/chakr_38m_v4_base.pt` (SHA-256 `7fb29b3519a957dbb4ffc12bdcc1cc50d6ea9bf49739458ebb9e3de031f504fb`)  
**NR-052 SFT Candidate 17M:** `experiments/pretraining_nr052/chakr_17m_nr052_sft.pt` (SHA-256 `c13211bea859ab6ff8cb89c749b29ae57df638a85f2bef60908e23827fa5a4bb`)  
**NR-052 SFT Candidate 38M:** `experiments/pretraining_nr052/chakr_38m_nr052_sft.pt` (SHA-256 `e08d363efdaac44cc62fa854ad1802de8ffb9de4060e090ad5373d66174c53ed`)  
**Active Tokenizer:** Canonical Byte-Level BPE 4096 (`data/experiments/vocab_4096`, Vocab SHA-256 `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a`)  
**Active Hardware:** Intel Core i9-13900H (14C/20T) | 32GB RAM | NVIDIA RTX A1000 6GB Laptop GPU | CUDA 12.6 | PyTorch 2.14.1+cu126  
**Latest Git State:** Working tree updated with NR-056 audit deliverables and Benchmark V3 suite, tracking `origin/exp/cbr-01-conversational-breakthrough` on GitHub.  

### Core Scientific Conclusions from NR-056:
1. **Architecture Lineage Fully Clarified:** Exactly two physical model families exist: `ChakrGenesis17M` (17,063,808 unique params: d=384, L=10) and `ChakrGenesis38M` (37,499,392 unique params: d=512, L=12). Historical citations of `d=512, L=6` for 17M and `d=768, L=6` for 38M were paper design proposals and transcription errors, respectively.
2. **Token Budget Reconciliation:** Corrected the NR-055 typographical error citing 10.5M tokens. Actual unique pretraining volume was **624,068 tokens** (Corpus V4), yielding 1.18M tokens seen over ~1.9 epochs.
3. **Benchmark V2 Contamination Proved:** 40% of Benchmark V2 prompts had factual co-occurrences in Corpus V1/V3. All non-zero factual hits (Tokyo, Paris, Canberra, Pacific Ocean, British Pound) represented memorization of training facts.
4. **Holm Multiple-Comparison Correction:** Under family-wise error rate control, the statistical superiority of 38M over 17M on Benchmark V2 fails significance ($p_{\text{adj}} = 0.7464$), and 38M vs RC3 Baseline fails ($p_{\text{adj}} = 1.0000$). No candidate possesses a defensible capability advantage on Benchmark V2.
5. **Sealed Benchmark V3 (300 Prompts) Capability Frontier:** Evaluated across 300 clean, independently authored items; performance dropped to **1.22%** (NR052 38M) and **0.89%** (RC3 Baseline). 0/35 arithmetic problems, 0/30 word problems, and 0/35 reading comprehension extractions were solved. The primary bottleneck is extreme pretraining under-exposure ($\sim 624\text{k}$ tokens).
