# CHAKRVIEW CURRENT STATE

**Current Phase:** Mission NR-052 Completed (Corpus V4, Data Quality & Controlled Language Acquisition)  
**Status:** COMPLETED & SCIENTIFICALLY VALIDATED  
**Active Release Candidate Checkpoint (Frozen):** `release/v1.0.0-rc1/chakr_1_0_rc1_best.pt` (SHA-256 `CF585EBF9AA70F630516EEC9C45E85F127E29ED91D6B6AE6A26D44FD20FCCE4E`)  
**RC2 Candidate Checkpoint (Frozen):** `experiments/genesis_stage_rc2_046/chakr_1_0_rc2_best.pt` (SHA-256 `33F93C75BA6C4CCD06A87DDB68AD300E11CE0F67AEF45D6AA0A6AE7868C2BE30`)  
**RC3 Milestone Candidate Checkpoint (Frozen):** `experiments/fast_track_047_049/chakr_1_0_rc3_best.pt` (SHA-256 `61B39AE2BC55C116ED3EC8164497EC798D1FB6994A449D852248B266CD9DC73C`)  
**NR-052 Pretrained Base 17M:** `experiments/pretraining_nr052/chakr_17m_v4_base.pt` (SHA-256 `e55b172a5a92a5d20739c9da9e623a96f13e77f0a149021e82ef45b7ee7ce2d9`)  
**NR-052 Pretrained Base 38M:** `experiments/pretraining_nr052/chakr_38m_v4_base.pt` (SHA-256 `bdf50337da5049369b7cb0dfa902dfebcfdd8f49fc35ecdf0c495cebb32ae7aa`)  
**NR-052 SFT Candidate 17M:** `experiments/pretraining_nr052/chakr_17m_nr052_sft.pt` (SHA-256 `18501ea917992ef15ae50db3e47bf9b13926868dc3bfa594a919010e9fcebce3`)  
**NR-052 SFT Candidate 38M:** `experiments/pretraining_nr052/chakr_38m_nr052_sft.pt` (SHA-256 `ca44b1c2ee5d9297e68ee25a4d95b5bafe3857db9ee647962c5a08272895f32b`)  
**Active Tokenizer:** Canonical Byte-Level BPE 4096 (`data/experiments/vocab_4096`, Vocab SHA-256 `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a`)  
**Active Hardware:** Intel Core i9-13900H (14C/20T) | 32GB RAM | NVIDIA RTX A1000 6GB Laptop GPU | CUDA 12.6 | PyTorch 2.14.1+cu126  
**Latest Git State:** Working tree updated with NR-052 artifacts, tracking `origin/exp/cbr-01-conversational-breakthrough` on GitHub.  

### Core Scientific Findings from NR-052:
1. **Uncovered Synthetic Template Multiplier:** Proved that legacy datasets claiming 15k+ items (`ucip14r`, `ucip17`) were heavily duplicated Cartesian products where 3 to 15 templates were multiplied across prompts. In conversational datasets, a single refusal phrase occurred 110 times identically.
2. **Corpus V4 Created:** Built an isolated, deduplicated pretraining corpus of 624,068 verified unique tokens across 6,134 documents with 0.0% benchmark contamination.
3. **Restored Semantic Scaling Advantage (38M > 17M):**
   - Clean pretraining on Corpus V4 reduced pretraining validation loss from 3.05 $\to$ 1.47 on 17M and from 2.85 $\to$ 1.16 on 38M.
   - 38M achieved sub-2.0 conversational SFT response loss (1.9964) for the first time in ChakrView history.
   - On the sealed 120-prompt benchmark, 38M cleanly outperformed 17M (0.117 vs 0.092, +27.2% relative) and generated exact zero-shot factual retrievals (Canberra, New Delhi).
4. **Primary Remaining Bottleneck:** Pretraining token volume. The data pipeline is now proven clean and effective; expanding from 624k tokens to $\ge 25\text{M}-50\text{M}$ tokens is the critical requirement for broad reasoning.
