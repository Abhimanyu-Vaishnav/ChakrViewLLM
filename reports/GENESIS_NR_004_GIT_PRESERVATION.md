# GENESIS-NR-004: Git Preservation Report

**Date:** 21 November 2026  
**Mission ID:** CV-GENESIS-NR-004  
**Project:** ChakrView  
**Repository:** `D:\Project\ChakrView`  

---

## 1. Preservation Summary

All uncommitted work across UCIP-28, UCIP-29, Ultimate Brain 001/002, Project Genesis, and Missions NR-001 through NR-003 was audited, staged, and committed cleanly to Git:

- **Branch:** `exp/cbr-01-conversational-breakthrough`
- **Preservation Commit Hash:** `9286592`
- **Commit Message:** `CV-GENESIS-NR-003: preserve tokenizer and neural reconstruction audit`
- **Total Files Committed:** 161 files changed, 76,510 insertions(+).
- **Working Tree Status:** Clean (`git status --short` returned clean).

---

## 2. Large Artifacts Preserved Outside Git (per .gitignore)

In accordance with repository `.gitignore` policies protecting against large binary bloating:
- Checkpoints (`.pt` files) are preserved in their respective directories:
  - `experiments/project_genesis/pretraining/genesis_pretrain_best.pt` (62,204,051 bytes)
  - `experiments/project_genesis/conversation_training/genesis_sft_best.pt` (20,733,097 bytes)
  - `experiments/genesis_neural_recovery_001/checkpoints/genesis_recovery_sft_best.pt` (20,733,719 bytes)
  - `releases/project_genesis_candidate/genesis_sft_best.pt` (20,733,097 bytes)
- Tokenized binary shards (`.bin` files) remain intact under `data/tokenized/stage_c/`.
