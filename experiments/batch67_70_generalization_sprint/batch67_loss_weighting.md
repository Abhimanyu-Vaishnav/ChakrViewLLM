# Batch 67 — Comparative Loss-Weighting Experiment Report

## 1. Executive Summary & Scientific Context

In Batch 63–66, we discovered that 88% of target tokens in synthetic arithmetic SFT examples were shared ChatML template tokens (`<|im_end|>\n` + `<EOS>`). The model quickly converged to near-zero loss (0.017) on template tokens, creating an aggregate loss reduction (~0.38) that masked high loss (~2.83) on arithmetic answer tokens.

Batch 67 tests the hypothesis that **loss reweighting** can address this imbalance and promote answer learning. We conducted a controlled 3-way comparative experiment:

1. **Condition A (Baseline)**: Standard uniform loss weighting (weight = 1.0 on all response tokens).
2. **Condition B (Answer-Emphasis)**: 5.0x loss weight on answer-content tokens; 1.0x on ChatML template and termination tokens.
3. **Condition C (Answer-Only Diagnostic)**: 1.0x loss weight on answer-content tokens; 0.0x on template tokens (masked out via target `-100`).

---

## 2. Experimental Setup

- **Base Checkpoint**: Canonical step-10,000 checkpoint (`96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`).
- **Data**: Verified synthetic arithmetic dataset from Batch 60 (Train: 32 examples, SHA: `89299ff5e2d81b143c68a6b6c9aae858b69a486f699c6c90c9855600bfbbd4c5`; Validation: 8 examples; Held-out Benchmark: 8 examples).
- **Optimizer**: AdamW, `lr=1e-4`, `weight_decay=0.01`, `grad_clip=1.0`, 50 steps, batch size 2, seed 42.
- **Normalization**: Weighted loss is normalized by the active weight sum:
  $$\mathcal{L} = \frac{\sum_i w_i \cdot \text{CE}_i}{\sum_i w_i}$$

---

## 3. Quantitative Comparison Across Conditions

| Metric | Condition A: Baseline (Uniform) | Condition B: Answer-Emphasis (5x) | Condition C: Answer-Only Diagnostic (Template 0x) |
|:---|:---:|:---:|:---:|
| **Training Steps** | 50 | 50 | 50 |
| **Initial Loss** | 7.6491 | 7.6491 | 7.6491 |
| **Final Loss** | 0.3533 | 0.8174 | 2.5045 |
| **Elapsed Time (sec)** | 2.87s | 2.89s | 2.85s |
| **TRAIN Exact Match** | 3 / 32 (9.38%) | 3 / 32 (9.38%) | 0 / 32 (0.00%) |
| **TRAIN Mean Answer Loss** | 2.6209 | 2.5702 | **2.5045** |
| **TRAIN Mean Template Loss** | **0.0055** | 0.0205 | 7.5719 (unsupervised) |
| **TRAIN EOS Rate** | 100.0% | 100.0% | **0.0%** (infinite digit loop) |
| **VAL Exact Match** | 0 / 8 (0.00%) | 1 / 8 (12.50%) | 0 / 8 (0.00%) |
| **VAL Mean Answer Loss** | 2.9463 | 2.8053 | **2.6959** |
| **VAL Mean Template Loss** | **0.0096** | 0.0340 | 7.6450 |
| **VAL EOS Rate** | 100.0% | 100.0% | 0.0% |
| **BENCH Exact Match** | 1 / 8 (12.50%) | 1 / 8 (12.50%) | 0 / 8 (0.00%) |
| **BENCH Mean Answer Loss** | 2.8308 | 2.7776 | **2.7553** |
| **BENCH EOS Rate** | 100.0% | 100.0% | 0.0% |

---

## 4. Key Scientific Findings

1. **Answer-Only Loss Destroys Generation Termination**:
   - In Condition C, zeroing template loss caused template loss to stay at ~7.6.
   - At inference, the model generated digits but **never generated `<|im_end|>` or `<EOS>`**, resulting in repetitive runaway digit loops (`'2171717171'`).
   - Consequently, exact match collapsed to 0/32 and EOS rate to 0.0%.
   - **Conclusion**: Template tokens cannot simply be discarded during causal SFT. The model must learn when and how to terminate generation.
2. **Answer-Emphasis Accelerates Answer Loss Reduction While Preserving EOS**:
   - In Condition B (5x weight), answer loss dropped to 2.57 (Train) and 2.80 (Val), outperforming Baseline (2.62 Train, 2.94 Val).
   - Template loss remained very low (0.0205), maintaining 100% EOS completion.
   - Condition B achieved 1/8 validation match (predicting modal answer `'21'`).
3. **Modal Collapse Remains the Primary Obstacle at 50 Steps**:
   - Across both Baseline and Answer-Emphasis, 50 steps at `batch_size=2` was insufficient to learn sample-specific additions; the model converged toward high-probability token combinations (`'27'` for Baseline, `'21'` for Answer-Emphasis).
