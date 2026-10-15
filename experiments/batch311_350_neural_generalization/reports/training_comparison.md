# Training Comparison & Generalization Gate Report — Batches 323–330

**Date:** 2026-10-15  
**Milestone:** Batches 311–350 Neural Generalization  
**Architecture:** ChakrMicro (3,443,136 parameters, 6 layers, $d_{model}=192$, 6 heads, vocabulary 4,096)  
**Hardware:** AMD/Intel x86_64, Windows NT, PyTorch 2.14.0+cpu, 14 execution threads

---

## 1. Experimental Overview

To answer the central scientific question—*Does extended training on a multi-domain target-masked dataset yield genuine generalization or overfit/memorize?*—we compared three controlled training regimes against the frozen canonical step-10,000 baseline:
1. **Canonical Baseline (Step 10,000):** Frozen pretraining baseline.
2. **Candidate Step 70:** 70 steps, linear warmup (15 steps) + cosine decay.
3. **Candidate Step 150:** 150 steps, linear warmup (25 steps) + cosine decay.
4. **Candidate Step 300:** 300 steps, linear warmup (30 steps) + cosine decay.

All candidates trained on the exact same 38-record multi-domain training set (`generalization_train.jsonl`) with strict target loss masking (prompt tokens masked with `-100`).

---

## 2. Quantitative Metric Comparison

| Model Arm | Train Steps | Final Train Loss | Validation Loss (PPL) | Unseen Entities Loss (PPL) | Unseen Templates Loss (PPL) | Clean Termination Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Canonical Baseline** | 0 | N/A | 8.5618 (5228.3) | 8.0865 (3250.4) | 8.3576 (4262.4) | 0.0% (0/20) |
| **Candidate Step 70** | 70 | 4.26 | 5.3643 (213.6) | 4.6910 (108.9) | 4.4552 (86.1) | **100.0% (20/20)** |
| **Candidate Step 150** | 150 | 1.23 | **5.1739 (176.6)** | **4.4785 (88.1)** | 4.4140 (82.6) | 85.0% (17/20) |
| **Candidate Step 300** | 300 | 0.57 | 5.3521 (211.1) | 4.5758 (97.1) | **4.2276 (68.6)** | 80.0% (16/20) |

---

## 3. Scientific Findings & Overfitting Diagnosis

1. **Step 150 Achieves Optimal Generalization Balance:**
   - Both Validation Loss (5.1739 vs 8.5618, **$-3.39$ nats**) and Held-Out Unseen Entities Loss (4.4785 vs 8.0865, **$-3.61$ nats**) reach their global minimum at **Step 150**.
   - Perplexity on unseen technical entities dropped from **3250.4 to 88.1**.
2. **Step 300 Enters Memorization / Overfitting:**
   - At Step 300, training loss dropped to 0.57 nats, but Validation Loss rebounded from 5.1739 to 5.3521 ($+0.18$ nats) and Unseen Entities Loss rebounded from 4.4785 to 4.5758 ($+0.10$ nats).
   - Generative clean termination declined from 100% (Step 70) to 80% (Step 300) as repetitive character loops began reappearing in low-probability completions.
3. **Language Retention Verified:**
   - On standard English, Code, and Hindi continuations, 3-gram repetition remained controlled across all candidates (English: 0.16–0.33, Code: 0.00–0.05, Hindi: 0.00).

---

## 4. Promotion Gate Decision

- **Candidate Step 150 (`candidate_step150.pt`) is Selected** as the strongest neural capability checkpoint:
  - Artifact Path: `experiments/batch311_350_neural_generalization/checkpoints/candidate_step150.pt`
  - SHA-256 Digest: `b6725ce7a4095fe0c551e0b236256d18b55ebd055a865e2f5150e73036384759`
- **Rejection of Step 300:** Step 300 is rejected due to empirical loss rebound on held-out validation and degradation of termination rates.
- **Canonical Preservation:** The canonical step-10,000 checkpoint remains strictly frozen and untouched.
