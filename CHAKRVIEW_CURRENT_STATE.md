# CHAKRVIEW CURRENT STATE

**Current Phase:** Mission NR-054 Completed (Benchmark Reconciliation, Checkpoint Lineage & Scientific Integrity Audit)  
**Status:** COMPLETED & SCIENTIFICALLY RECONCILED  
**Active Release Candidate Checkpoint (Frozen):** `release/v1.0.0-rc1/chakr_1_0_rc1_best.pt` (SHA-256 `cf585ebf9aa70f630516eec9c45e85f127e29ed91d6b6ae6a26d44fd20fcce4e`)  
**RC2 Candidate Checkpoint (Frozen):** `release/v1.0.0-rc2/chakr_1_0_rc2_best.pt` (SHA-256 `33f93c75ba6c4ccd06a87ddb68ad300e11ce0f67aef45d6aa0a6ae7868c2be30`)  
**RC3 Milestone Candidate Checkpoint (Frozen):** `release/v1.0.0-rc3/chakr_1_0_rc3_best.pt` (SHA-256 `61b39ae2bc55c116ed3ec8164497ec798d1fb6994a449d852248b266cd9dc73c`)  
**NR-052 Pretrained Base 17M:** `checkpoints/nr052/chakr_17m_v4_base.pt` (SHA-256 `58781710bb4d85b40d77281eb6ccecb1754a3f3fadc468db8c1fa8b51a2dd062`)  
**NR-052 Pretrained Base 38M:** `checkpoints/nr052/chakr_38m_v4_base.pt` (SHA-256 `7fb29b3519a957dbb4ffc12bdcc1cc50d6ea9bf49739458ebb9e3de031f504fb`)  
**NR-052 SFT Candidate 17M:** `checkpoints/nr052/chakr_17m_nr052_sft.pt` (SHA-256 `c13211bea859ab6ff8cb89c749b29ae57df638a85f2bef60908e23827fa5a4bb`)  
**NR-052 SFT Candidate 38M:** `checkpoints/nr052/chakr_38m_nr052_sft.pt` (SHA-256 `e08d363efdaac44cc62fa854ad1802de8ffb9de4060e090ad5373d66174c53ed`)  
**Active Tokenizer:** Canonical Byte-Level BPE 4096 (`data/experiments/vocab_4096`, Vocab SHA-256 `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a`)  
**Active Hardware:** Intel Core i9-13900H (14C/20T) | 32GB RAM | NVIDIA RTX A1000 6GB Laptop GPU | CUDA 12.6 | PyTorch 2.14.1+cu126  
**Latest Git State:** Working tree updated with NR-054 forensic audit deliverables, tracking `origin/exp/cbr-01-conversational-breakthrough` on GitHub.  

### Core Scientific Conclusions from NR-054:
1. **Benchmark Means Discrepancy Resolved:** Prompt-level audit confirmed reported means (`0.220`, `0.180`, `0.180`, `0.180`, `0.240`) are mathematically exact. The NR-053 markdown distribution table was erroneously pasted from an uncommitted draft run with a loose regex rubric. The true distribution for NR052 38M is `[46, 0, 0, 4]` (sum 12, mean 0.240).
2. **Checkpoint Lineage & Hashing Standardized:** Resolved all SHA-256 prefixes across the 12 primary checkpoints. Confirmed `chakr_17m_nr052_sft.pt` (`c13211be...`) as the authentic file referred to as `chakr_17m_sft_v4.pt`.
3. **Independent Validation Reproduction (100% Precision):** GPU-based forward recomputation reproduced all validation figures with zero error: 17M base loss `1.4763`, 38M base loss `1.1684`, 17M SFT response loss `2.2376`, 38M SFT response loss `1.9964`.
4. **Evaluator False Positives Identified:** Automated rubric awarded score 3 to malformed outputs when short substring matches (`'2'`, `'ice'`, `'red'`, `'अ'`) matched. Only 2 prompts (`FQA-01` Tokyo and `FQA-03` Pacific) are verified factual hits across all models.
5. **Statistical Significance Reality:** Bootstrap resampling ($B=10,000$) and paired permutation testing establish that 38M (4 hits) vs 17M (3 hits) on 50 items has $p = 1.0000$ (not statistically significant). Validation loss scaling is mathematically proven, but small-benchmark claims are statistically unsupported.
