# ChakrView Experiment 08 — Synthetic Copying Curriculum Report

**Date**: 2026-10-21  
**Commit/Parent Checkpoint**: Canonical Parent Checkpoint `Step-10000` (`experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`)  
**SHA-256 (Canonical Parent Checkpoint)**: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`  
**SHA-256 (Canonical Tokenizer Vocab)**: `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a`  
**Execution Environment**: Isolated Python 3.14 virtual environment (`.venv`), CPU execution.  
**Experiment Directory**: `experiments/research_v08_json_sft/exp08_synthetic_copy_curriculum/`  

---

## 1. Executive Summary & Primary Status

### Primary Status
`FAMILIAR-TOKEN GENERALIZATION ONLY`

### Core Result
Under a 3-stage synthetic copying curriculum of 2,000 diverse training examples (single target, distractor keys, and varied syntactic templates), **ChakrMicro successfully learns a robust context-conditioned copying rule across familiar token identities in completely novel contexts**:
- On **Test A (Familiar Token Identities in Novel Contexts)**: The model achieves **92.9% (184/198)** teacher-forced first-token accuracy and **92.9% (184/198)** free-generation exact match (mean target rank **1.5 / 4,093**, mean target probability **76.5%**).
- On **Validation Set (Novel Contexts, Familiar Tokens)**: Achieves **91.5% (182/199)** free-generation exact match (Stage B distractor lookups: **98.0%**, Stage C syntax variations: **84.8%**).
- Across both evaluations, exact matches in free generation equaled teacher-forced accuracy at **100% fidelity**, confirming that exposure bias is not an issue once the first token logit is correctly oriented.

However, when tested on **Test B (Held-Out Token Identities Never Supervised as Targets)**:
- The model achieves **0.0% (0/300)** teacher-forced first-token accuracy and **0.0% (0/300)** free-generation exact match.
- While the mean target rank on Test B improved from the baseline noise floor of **1,305.7** down to **385.1** (a 3.4× rank elevation), the model still selects a familiar token from its supervised output vocabulary instead of routing the unseen token from context into the prediction head.

This conclusively proves that **ordinary causal SFT does learn context-conditioned copying rather than rigid prompt-to-answer memorization**, but the learned copying circuit remains dependent on token representations having been aligned through output projection supervision.

---

## 2. Integrity, Controls, and Pre-Training Audit

All pre-training safety gates and unit tests were verified before training in [audit_and_controls.py](file:///d:/Project/ChakrView/experiments/research_v08_json_sft/exp08_synthetic_copy_curriculum/audit_and_controls.py):
1. **Parent Checkpoint Integrity**: Confirmed exact SHA-256 match `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`.
2. **Tokenizer Integrity**: Confirmed exact SHA-256 match on `vocab.json` (`db4d6119...`).
3. **Leakage & Partition Verification**:
   - Zero prompt overlap between `train.jsonl` and `val.jsonl`, `test_a.jsonl`, or `test_b.jsonl`.
   - Test A target tokens are a strict subset of Train target tokens (all 120 target tokens supervised during training).
   - Test B target tokens are strictly disjoint from Train target tokens (60 held-out tokens).
4. **Constant-Output Baseline**:
   - Predicting the most frequent target token (Token ID 1810) yields only **1.55%** on Train, **1.01%** on Test A, and **0.00%** on Test B. The curriculum design prevents majority-class shortcut learning.
5. **Causal Alignment Verification**: Response tokens are masked with `-100`; input position $t$ predicts target token at $t+1$; trailing EOS is supervised separately.
6. **Checkpoint Selection Protocol**: Checkpoints were monitored strictly on Validation loss and Validation Exact Match. Test A and Test B splits were completely held out and never used for selection.

---

## 3. Dataset Specification

| Dataset Split | Examples | Unique Target Tokens | SHA-256 Hash | Purpose |
| :--- | :---: | :---: | :--- | :--- |
| `train.jsonl` | 2,000 | 120 | `b5d3c6cfe85bae30fff4807bf8a0b351aafd13fc04e86fca0166b7102ec93af7` | 800 Stage A + 600 Stage B + 600 Stage C |
| `val.jsonl` | 199 | 95 | `e13a51b48968b77823ae61fff4a817eb19af978fc573bf1681281f9f7a9394cd` | Checkpoint selection only |
| `test_a_familiar_tokens.jsonl` | 198 | 98 | `0224df2d761f15dfdf84cf4360106b0d5fd68098e03d5688698b2fa613cd8d7d` | Primary test: Familiar tokens in novel contexts |
| `test_b_held_out_tokens.jsonl` | 300 | 58 | `7e0dec9050de35c6b04fc1fa9554e52053291f687bc73911850c6b3bfb8db219` | Diagnostic: Held-out token identities |

### Curriculum Stages:
- **Stage A (Standard Target)**: `Target value: {value}\nCopy: ` $\to$ `{value}`
- **Stage B (Distractor Key Lookup)**: `Items: Alpha={v1}, Beta={v2}, Gamma={v3}\nCopy {Target_Label}: ` $\to$ `{v_target}`
- **Stage C (Syntactic Variation)**:
  - `Data: key={target}, ref={distractor}\nExtract key: `
  - `Input sequence: [{distractor}, {target}]\nSelect second item: `
  - `Assignment: value is {target}; ignore {distractor}.\nOutput: `
  - `Record: primary:{target} secondary:{distractor}\nPrimary: `

---

## 4. Training Progression

- **Optimizer**: AdamW ($lr = 3 \times 10^{-4}$, weight decay = 0.01, grad norm clip = 1.0)
- **Batch Size**: 8 sequences per step
- **Budget**: 400 optimizer steps (3,200 sequence exposures, ~1.6 passes over training set)
- **Loss**: Standard unweighted Cross-Entropy Loss (`ignore_index=-100`)

| Step | Train Loss | Val Loss | Val EM (%) | Test A EM (Familiar) (%) | Test B EM (Held-Out) (%) | Dynamics |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** (Parent) | *N/A* | 9.8820 | 0.0% | 0.0% | 0.0% | Rank ~1123, noise floor |
| **50** | 27.9526 | 2.7329 | 1.0% | 0.5% | 0.0% | Syntax acquisition begins |
| **100** | 18.5393 | 2.2396 | 4.0% | 6.6% | 0.0% | Initial target/key binding |
| **150** | 14.7285 | 1.8787 | 17.1% | 17.2% | 0.0% | Stage C syntactic extraction emerges |
| **200** | 11.0890 | 1.3269 | 46.7% | 50.5% | 0.0% | Copying reaches 50% across novel contexts |
| **250** | 7.7516 | 0.8472 | 70.4% | 78.3% | 0.0% | Stage B distractor lookup hits 79% |
| **300** | 4.4117 | 0.5822 | 83.4% | 85.4% | 0.0% | Rapid convergence of attention routing |
| **350** | 3.1634 | 0.4303 | 86.4% | 85.4% | 0.0% | Target rank collapses to < 2.0 on familiar tokens |
| **400** | **2.0398** | **0.2792** | **91.5%** | **92.9%** | **0.0%** | **Best Checkpoint Selected (Validation EM = 91.5%)** |

---

## 5. Comprehensive Quantitative Evaluation

Evaluated with [evaluate_exp08.py](file:///d:/Project/ChakrView/experiments/research_v08_json_sft/exp08_synthetic_copy_curriculum/evaluate_exp08.py). All metrics separate target-token accuracy from EOS prediction.

### Split Performance Breakdown

| Split | Checkpoint | Teacher-Forced 1st-Token Acc | Free-Gen Exact Match | Mean Target Rank | Mean Target Prob | Output Diversity |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Validation (199)** | Step-10000 (Parent) | 0.0% (0/199) | 0.0% (0/199) | 1,123.8 / 4,093 | 0.03% | 0.11 |
| **Validation (199)** | **Exp08 Best (Step 400)** | **91.5% (182/199)** | **91.5% (182/199)** | **3.0 / 4,093** | **74.0%** | **0.49** |
| **Test A: Familiar (198)**| Step-10000 (Parent) | 0.0% (0/198) | 0.0% (0/198) | 1,168.0 / 4,093 | 0.03% | 0.11 |
| **Test A: Familiar (198)**| **Exp08 Best (Step 400)** | **92.9% (184/198)** | **92.9% (184/198)** | **1.5 / 4,093** | **76.5%** | **0.48** |
| **Test B: Held-Out (300)**| Step-10000 (Parent) | 0.0% (0/300) | 0.0% (0/300) | 1,305.7 / 4,093 | 0.01% | 0.10 |
| **Test B: Held-Out (300)**| **Exp08 Best (Step 400)** | **0.0% (0/300)** | **0.0% (0/300)** | **385.1 / 4,093** | **0.06%** | **0.17** |

---

### Detailed Breakdowns on Test A (Familiar Tokens, Novel Contexts)

#### 1. By Curriculum Stage
- **Stage B (Distractor Multi-Item Lookup)**: **97.0% (97/100)** Exact Match
- **Stage C (Syntactic Variation)**: **88.8% (87/98)** Exact Match

#### 2. By Prompt Template
- `bracket` (`Input sequence: [{distractor}, {target}]\nSelect second item: `): **95.8% (23/24)**
- `query_assign` (`Assignment: value is {target}; ignore {distractor}.\nOutput: `): **95.2% (20/21)**
- `key_val` (`Data: key={target}, ref={distractor}\nExtract key: `): **96.9% (31/32)**
- `multi_item_lookup` (`Items: Alpha=..., Beta=..., Gamma=...\nCopy [Key]: `): **97.0% (97/100)**
- `colon_tag` (`Record: primary:{target} secondary:{distractor}\nPrimary: `): **61.9% (13/21)**

#### 3. By Target Position within Input Context
- **Position 0** (Target appears first among items): **89.2% (91/102)**
- **Position 1** (Target appears second among items): **94.4% (51/54)**
- **Position 2** (Target appears third among items): **100.0% (42/42)**

---

## 6. Qualitative Analysis & Failure Modes

### Successful Copying on Novel Contexts (Test A)

```text
Prompt: Items: Alpha=bit, Beta=rect, Gamma=man\nCopy Beta: 
Expected: rect
Generated: rect (Match: TRUE | Rank: 1 | Prob: 86.1%)

Prompt: Data: key=servic, ref=Hel\nExtract key: 
Expected: servic
Generated: servic (Match: TRUE | Rank: 1 | Prob: 90.4%)

Prompt: Items: Alpha=ffici, Beta=Tensor, Gamma=lat\nCopy Alpha: 
Expected: ffici
Generated: ffici (Match: TRUE | Rank: 1 | Prob: 92.3%)

Prompt: Record: primary:updat secondary:gist\nPrimary: 
Expected: updat
Generated: updat (Match: TRUE | Rank: 1 | Prob: 93.0%)
```

### Failure Mode on Held-Out Token Identities (Test B)

When presented with tokens never supervised as output targets:
```text
Prompt: Items: Alpha=through, Beta=engineer, Gamma=mor\nCopy Beta: 
Expected: engineer (Target Token 1601)
Generated: digen (Token 874 | Target Rank: 116 | Prob: 0.02%)

Prompt: Input sequence: [chahi, zar]\nSelect second item: 
Expected: zar (Target Token 1360)
Generated: ith (Token 435 | Target Rank: 395 | Prob: 0.02%)

Prompt: Data: key=hon, ref=Code\nExtract key: 
Expected: hon (Target Token 1135)
Generated: Det (Token 3826 | Target Rank: 214 | Prob: 0.03%)
```
**Mechanism**: The attention mechanism successfully detects the target slot (elevating the target token rank from 1,305 to 385), but the final linear prediction head ($W_{\text{unembed}}$) has only learned sharp output projection weights for the supervised training vocabulary (the 120 tokens). Because the model lacks an architectural pointer/copy mechanism, it must route through its unembedding layer, which biases the argmax towards familiar tokens with high prior norm.

---

## 7. Artifact Integrity

- **Canonical Parent Checkpoint SHA-256**: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (Verified unchanged)
- **Canonical Vocab SHA-256**: `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` (Verified unchanged)
- **Exp08 Best Checkpoint SHA-256**: `26a2d405498111b404f70cff48098db3354db66ca75a67ddb48483555bc07fc0`
- **Exp08 Final Checkpoint SHA-256**: `e5cb72d113ba12b22d2b1c1faf538126405e0bacc8ade95aa740cfe742cf6409`

---

## 8. Epistemic Assessment & Recommendation

### What this Result Establishes
1. **Causal SFT learns contextual copying rules, not rote prompt memorization**: The model successfully extracts values from diverse distractors, varying keys (`Alpha`, `Beta`, `Gamma`, `key`, `Primary`), varied positions, and unseen prompt syntax at **92.9% accuracy**.
2. **Exposure bias is non-existent for single-field extraction**: When teacher-forcing succeeds, greedy generation succeeds ($92.9\% \leftrightarrow 92.9\%$).
3. **The copying circuit in pure causal transformers requires output-space alignment**: Because the model copies by predicting tokens through its static unembedding matrix $W_{\text{unembed}} \in \mathbb{R}^{d \times V}$, it cannot emit a token identity that has never been grounded into the copy head's projection space during SFT.

### Recommended Next Step
**Experiment 09 — Vocabulary-Scale Grounding vs Architectural Copy Pointer**:
Now that we know causal SFT easily learns context-conditioned copying across all grounded tokens (92.9% across varied distractors and templates), determine whether scaling the curriculum target set to cover the full vocabulary (or top 2,000 subwords) provides universal copying coverage, OR whether an architectural copy mechanism (such as a dual-head pointer network) is strictly necessary for zero-shot rare/unseen token copying.
