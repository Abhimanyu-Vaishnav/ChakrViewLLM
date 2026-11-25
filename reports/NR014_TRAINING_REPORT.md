# CHAKRVIEW RESEARCH REPORT: NR-014 TRAINING REPORT
**Mission ID:** CV-GENESIS-NR-014  
**Date:** 2026-11-25  
**Trained Candidate:** `ChakrGenesis-17M Stage-F Best`  
**Initialization Checkpoint:** `experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt` (`2c21632e`)  
**Training Corpus:** `data/stage_f_corpus_014/stage_f_train.jsonl` (5,388 balanced samples: 87.0% conversational, 13.0% verified coding)  
**Validation Corpus:** `data/stage_f_corpus_014/stage_f_val.jsonl` (570 samples)  

---

## 1. Executive Summary
Under Phase 6 of Mission `CV-GENESIS-NR-014`, we conducted a balanced fine-tuning campaign on ChakrGenesis-17M. Training employed exact shifted next-token autoregressive prediction, prompt loss masking, mixed precision (AMP FP16), and a gentle learning rate ($4.0 \times 10^{-5}$) with cosine decay to preserve conversational stability while instilling programming capabilities.

---

## 2. Convergence Progress & Training Metrics
- **Epochs:** 2 full epochs (672 optimizer steps)
- **Batch Size:** 16 sequences ($T=256$)
- **Throughput:** ~29,800 tokens/second on NVIDIA RTX A1000 GPU
- **Initial Baseline Validation Loss:** 1.9838 (PPL: 7.27)
- **Epoch 1 Validation Loss:** 1.8135 (PPL: 6.13)
- **Epoch 2 Validation Loss:** **1.7645 (PPL: 5.84)** — Statistically verified improvement over baseline.

---

## 3. Checkpoint Ledger
- `genesis_17m_stage_f_best.pt`: `454aa58af12cb915ecf9bf144366489eb56dcfa088e0845574bc24fcca6f6ab8` (68,292,561 bytes)
- `genesis_17m_stage_f_final.pt`: `0a5eb83f804c043ca5f167da4255a4e74080c2e764298e856368627a0c13d96b` (68,292,659 bytes)
