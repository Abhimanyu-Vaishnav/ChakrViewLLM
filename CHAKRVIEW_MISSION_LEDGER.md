# CHAKRVIEW 1.0 MISSION LEDGER

## Autonomous End-to-End Neural Brain Development Program
**Repository:** `D:\Project\ChakrView`  
**Active Branch:** `exp/cbr-01-conversational-breakthrough`

---

| Mission ID | Phase Title | Input Checkpoint / Commit | Output Artifacts | Primary Metric / Score | Scientific Verdict |
| :---: | :--- | :--- | :--- | :---: | :---: |
| **NR-020** | Benchmark Integrity & Neural Contribution Audit | Canonical Stage-D (`2c21632e`) | `data/stage_j_020/matched_benchmark_1000_results.json` | Sys D: 37.7% (1,000 tasks) vs Sys A: 1.4% | Matched benchmark eliminates artificial OOD advantage; confirms source grounding superiority. |
| **NR-046** | Controlled Remediation & RC2 Development | Stage-D + Corpus V2 | `chakr_1_0_rc2_best.pt` (`33f93c75`) | Response Val Loss: 1.5382, Coherence: 100%, Echo: 0% | Fixed prompt echo and unmasked loss defect. |
| **NR-047-049**| Fast Track Milestone: Corpus V3 & RC3 Development | RC2 + Corpus V3 | `chakr_1_0_rc3_best.pt` (`61b39ae2`) | Sealed Benchmark Mean: 0.158/3.0 (5.3%), Val Loss: 1.18 | Response masking and semantic tuning achieve first measurable non-zero capability without echo. |
| **NR-050** | Controlled Architecture Scaling Study (17M vs 38M) | Direct SFT on 38M | `chakr_genesis_38m_best.pt` (`76240745`) | Sealed Benchmark: 2.2% vs 5.3% | Scaling parameters without pretraining base degrades convergence; pretraining required. |
| **NR-051** | True Language Pretraining & Controlled Scaling | Scratch (`seed=42`) | `chakr_17m_pretrained_base.pt`, `chakr_38m_pretrained_base.pt`, `chakr_17m_sft_candidate.pt`, `chakr_38m_sft_candidate.pt` | PT Val Loss: 2.8502 (38M) vs 3.0516 (17M); SFT Val Loss: 2.2262 (38M) vs 2.4155 (17M) | Pretraining restores scaling laws; 38M converges faster and lower in loss than 17M. 1M tokens undertrained for 38M factual recall. |
| **NR-052** | Corpus V4, Data Quality & Controlled Language Acquisition | Corpus V4 (`seed=42`) | `chakr_17m_v4_base.pt`, `chakr_38m_v4_base.pt`, `chakr_17m_nr052_sft.pt`, `chakr_38m_nr052_sft.pt` | PT Val Loss: 1.1684 (38M) vs 1.4763 (17M); SFT Val Loss: 1.9964 (38M) vs 2.2376 (17M); Sealed Benchmark: 0.117 (38M) vs 0.092 (17M) | Eliminating template multipliers cut perplexity by >60%; 38M cleanly surpasses 17M in both loss and zero-shot factual recall. |
