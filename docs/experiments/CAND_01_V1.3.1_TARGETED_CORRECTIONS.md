# CHAKRVIEW — CAND_01 v1.3.1 TARGETED SCIENTIFIC CORRECTIONS & CLOSURE REGISTER
## Targeted Resolution and Formal Disposition of Verification Findings VER-01 Through VER-05

**Document ID:** `REG-EXP-CAND01-COR-003`  
**Location:** `docs/experiments/CAND_01_V1.3.1_TARGETED_CORRECTIONS.md`  
**Supersedes/Augments:** `docs/experiments/CAND_01_V1.3_CORRECTION_AND_DECISION_REGISTER.md`  
**Candidate ID:** `CAND_01_CONTROLLED_ENTITY_COPYING`  
**Target Milestone:** CAND_01 v1.3.1 Scientific Closure Pass  
**Parent Governance Standard:** `docs/experiments/EXPERIMENT_GOVERNANCE_PROTOCOL_v1.0.md`  
**Status:** `CANONICAL_CLOSURE_REGISTER`  
**Execution Authorization Posture:** **`UNAPPROVED`**  
**Date:** 2026-11-02  

---

## 1. Objective & Scope of Pass v1.3.1

This document provides a formal, traceable scientific resolution and closure record for findings **VER-01 through VER-05** identified during the independent adversarial verification of CAND_01 v1.3. 

This pass operates strictly under **documentation-only constraints**:
- Zero source code changes.
- Zero dataset generation or sample synthesis.
- Zero model training, fine-tuning, inference, calibration, or benchmark execution.
- Zero Git staging, commits, pushes, or history rewrites.
- Absolute preservation of canonical artifacts and protected files (`model.py`, `session.py`, checkpoint `96357705...`, `.gitignore`).
- Absolute avoidance of `experiments/overnight_runs/`.

---

## 2. Detailed Technical Dispositions of Findings VER-01 Through VER-05

### 2.1 VER-01: Two-Way Cluster Bootstrap Algorithmic Specification

#### The Problem Analyzed
The independent verification noted that while v1.3 mandated a "two-way cluster bootstrap resampled simultaneously over unique key token IDs and unique target token IDs ($B=2,000$)", it lacked an explicit algorithmic resampling contract for crossed binary data. In crossed evaluation designs where observations represent pairs $(K_i, V_j)$, independent naive cluster sampling of keys and values generates bootstrap replicates with empty cells or non-constant sample sizes per draw, altering the evaluation estimand.

#### Scientific & Statistical Resolution
1. **Formal Sampling Unit Definition:**  
   The observation unit is a single evaluation instance $y_{m} \in \{0, 1\}$ indexed by prompt $m$, associated with query key token $K(m) \in \mathcal{K}$ and target value token $V(m) \in \mathcal{V}$.
2. **Resampling Algorithm Formulation:**  
   To preserve the crossed structure without degenerate matrix collapse, the bootstrap employs the **multiway cluster resampling scheme** (Cameron, Gelbach, & Miller 2011; Menzel 2021 for crossed networks):
   - In each bootstrap replication $b \in \{1, \dots, B\}$:
     a. Draw a cluster sample $\mathcal{K}^{*(b)}$ with replacement from the set of unique keys $\mathcal{K}$ ($|\mathcal{K}^{*(b)}| = |\mathcal{K}|$).
     b. Draw an independent cluster sample $\mathcal{V}^{*(b)}$ with replacement from the set of unique targets $\mathcal{V}$ ($|\mathcal{V}^{*(b)}| = |\mathcal{V}|$).
     c. For each observed evaluation record $m$, let $c_K(m)$ be the multiplicity of token $K(m)$ in $\mathcal{K}^{*(b)}$, and let $c_V(m)$ be the multiplicity of token $V(m)$ in $\mathcal{V}^{*(b)}$.
     d. Assign the observational weight $w_m^{(b)} = c_K(m) \times c_V(m)$.
     e. Compute the weighted accuracy estimand:
        $$\hat{\theta}^{*(b)} = \frac{\sum_{m=1}^N w_m^{(b)} y_m}{\sum_{m=1}^N w_m^{(b)}}$$
3. **Degeneracy & Replicate Filtering:**  
   A replicate $b$ is flagged as **degenerate** if $\sum_{m=1}^N w_m^{(b)} < 0.5 N$ (severe coverage loss) or if the effective number of unique keys or targets sampled is $< 10$. Degenerate replicates are discarded and re-drawn. The procedure mandates achieving a minimum of **$B_{\text{valid}} = 2,000$ valid replicates**.
4. **Reproducible Seed Policy:**  
   Resampling is governed by fixed deterministic seed: `bootstrap_seed = 999042`.
5. **Confidence Interval Calculation:**  
   Report the $95\%$ percentile bootstrap confidence interval: $[\hat{\theta}^{*}_{(0.025)}, \hat{\theta}^{*}_{(0.975)}]$.
6. **Fallback & Governance Classification:**  
   If empirical evaluation reveals that the crossed network structure in specific task families produces $>10\%$ degenerate draws, the evaluator automatically falls back to **one-way clustered bootstrap grouped by `target_val_token_id`** with robust sandwich standard error reporting.
   - **Status:** **`RESOLVED_WITH_PROVISIONAL_ALGORITHM`** (Algorithm fully specified; empirical stability flagged for pre-execution benchmark validation).

---

### 2.2 VER-02: GLMM Mathematical Formulation Correction

#### The Problem Analyzed
The independent verification identified an incorrect residual term $\epsilon_{ij}$ in `SCIENTIFIC_DESIGN_v1.3` §7.2:
$$\text{logit}(P(Y_{ij} = 1)) = \beta_0 + \beta_{\text{arm}} + u_{\text{key}(i)} + w_{\text{val}(j)} + \epsilon_{ij}$$
In a standard Bernoulli generalized linear mixed model with logit link, the variance of the observation is completely determined by the mean ($p(1-p)$). Adding an additive Gaussian disturbance $\epsilon_{ij}$ specifies an observation-level random effect (OLRE) for overdispersion, which is unidentifiable on single-trial binary data.

#### Scientific & Statistical Resolution
The GLMM equation is formally corrected to the canonical crossed random-effects logistic regression syntax:

$$Y_{ij} \sim \text{Bernoulli}(p_{ij})$$

$$\text{logit}(p_{ij}) = \beta_0 + \beta_{\text{arm}} + u_{\text{key}(i)} + w_{\text{val}(j)}$$

Where:
- $Y_{ij} \in \{0, 1\}$ is the binary retrieval outcome for key $i$ and value $j$.
- $\beta_0$ is the global log-odds intercept.
- $\beta_{\text{arm}}$ is the fixed treatment effect of the experimental arm.
- $u_{\text{key}(i)} \sim \mathcal{N}(0, \sigma_{\text{key}}^2)$ is the random intercept for key token identities.
- $w_{\text{val}(j)} \sim \mathcal{N}(0, \sigma_{\text{val}}^2)$ is the random intercept for value token identities.
- Key and value random effects are assumed mutually independent: $\text{Cov}(u_{\text{key}}, w_{\text{val}}) = 0$.

**Governance & Estimation Posture:**  
- The model is conditional on the crossed random effects.
- It is explicitly acknowledged that the GLMM has **not been fitted or validated** on real model outputs.
- Fitting will be performed via adaptive Gauss-Hermite quadrature or Laplace approximation. If numerical convergence issues arise during benchmark evaluation, the two-way cluster bootstrap serves as the primary pre-registered uncertainty estimator.
- **Status:** **`RESOLVED`**

---

### 2.3 VER-03: Bidirectional Association Isolation & Symmetric Leakage Prevention

#### The Problem Analyzed
`DATASET_SPECIFICATION_v1.3` §4 prohibited forward pair overlap ($\mathcal{P}(\text{Train}) \cap \mathcal{P}(\text{Val}) = \emptyset$). However, it omitted an explicit policy on reverse mappings: whether a pair $(V_1, K_1)$ could appear in validation or testing if $(K_1, V_1)$ was present in training. Because ChakrMicro uses tied input/output embeddings ($W_{\text{out}} = W_{\text{emb}}^T$), learning an association $(A \to B)$ optimizes representation alignment that could leak into reverse associative retrieval $(B \to A)$.

#### Scientific & Statistical Resolution
A formal **Bidirectional Association Exclusion Rule** is enacted across all partitions where relational pairings exist:

1. **Formal Exclusion Definition:**  
   Let $\mathcal{P}_{\text{dir}}(R) = \{(k, v) \mid (k, v) \in R.\text{context\_pairs}\}$ be the set of directed pairs in record $R$.  
   Define the symmetric association closure $\mathcal{P}_{\text{sym}}(R) = \{(k, v), (v, k) \mid (k, v) \in R.\text{context\_pairs}\}$.  
   The dataset generator must strictly enforce:
   $$\left( \bigcup_{R \in \text{Train}} \mathcal{P}_{\text{sym}}(R) \right) \cap \left( \bigcup_{R \in \text{Val Recomb}} \mathcal{P}_{\text{dir}}(R) \right) = \emptyset$$
   $$\left( \bigcup_{R \in \text{Train}} \mathcal{P}_{\text{sym}}(R) \right) \cap \left( \bigcup_{R \in \text{Test Recomb}} \mathcal{P}_{\text{dir}}(R) \right) = \emptyset$$
   $$\left( \bigcup_{R \in \text{Train}} \mathcal{P}_{\text{sym}}(R) \right) \cap \left( \bigcup_{R \in \text{Test Disjoint}} \mathcal{P}_{\text{dir}}(R) \right) = \emptyset$$
   Neither $(k, v)$ nor $(v, k)$ observed during training may appear in validation, test recombination, or disjoint test partitions.
2. **Explicit Scope Restrictions:**
   - **Exclusion from TF-01:** This rule applies strictly to relational key-value bindings ($k \neq v$). It is mathematically not applicable to TF-01 identity-copy tasks (`COPY: X -> X`), which are governed strictly by single-token disjointness.
   - **Disjoint Partition D Preservation:** In Partition D (Test Disjoint-Token Copying), tokens are drawn strictly from $\mathcal{E}_{\text{test}}$, where $\mathcal{E}_{\text{test}} \cap \mathcal{E}_{\text{train}} = \emptyset$. Bidirectional exclusion is trivially satisfied by token-level disjointness.
3. **Combinatorial Recalculation:**
   - With $|\mathcal{E}_{\text{train}}| = 450$ tokens, the total number of unordered token pairs is:
     $$\binom{450}{2} = \frac{450 \times 449}{2} = \mathbf{101,025\ undirected\ pairs}$$
   - Total unordered associations required:
     - Partition A (Train): $\approx 5,000$ unique undirected pairs.
     - Partition B (Val Recomb): $\approx 750$ unique undirected pairs.
     - Partition C (Test Recomb): $\approx 750$ unique undirected pairs.
     - Total required: $\approx 6,500 \ll 101,025$.
   - **Mathematical Feasibility:** Enforcing bidirectional exclusion consumes only $\approx 6.4\%$ of available undirected combinatorial capacity. Partition counts of 4,000 / 500 / 500 / 500 remain 100% mathematically feasible.
4. **Association Identity vs. Formatting:** Association identity is defined strictly at the discrete token ID tuple level, completely invariant to formatting syntax, whitespace, or prompt template variations.
- **Status:** **`RESOLVED`**

---

### 2.4 VER-04: Gradient Calibration Step Definition & Warmup Alignment

#### The Problem Analyzed
The independent verification identified an ambiguity regarding whether "60 steps" referred to micro-batch forward passes or optimizer update steps. In `sft_engine.py`, gradient accumulation is configured with `grad_accum_steps = 4`. If interpreted as micro-batches, 60 micro-batches would yield only 15 optimizer updates, failing to reach the 50-step linear warmup peak.

#### Scientific & Statistical Resolution
The pre-flight calibration protocol is formally codified with exact operational step semantics:

1. **Definitive Step Semantics:**  
   The calibration duration is defined strictly as **60 optimizer update steps**.  
   With micro-batch size $B_{\text{micro}} = 2$ and gradient accumulation $M = 4$:
   $$\text{Total Micro-Batches} = 60 \times 4 = \mathbf{240\ forward/backward\ passes}$$
   $$\text{Total Effective Sequences} = 240 \times 2 = 480\ \text{sequences}$$
2. **Explicit Warmup Traversal:**  
   - Steps 1 through 50: Linear warmup from $LR = 2.0 \times 10^{-6}$ to peak $LR = 1.0 \times 10^{-4}$.
   - Steps 51 through 60: Post-warmup cosine decay initiation at peak learning rate velocity.
   - This guarantees that gradient stability is evaluated under maximum parameter update force.
3. **Exact Measurement Point:**  
   The telemetry metric is defined strictly as the **unclipped global gradient $L_2$ norm**:
   $$\|\mathbf{g}\|_2 = \sqrt{\sum_{i=1}^P \|\nabla_{\theta_i} \mathcal{L}_{\text{accum}}\|_2^2}$$
   Computed **after** all 4 micro-batch gradients are accumulated and averaged, but **immediately prior** to `torch.nn.utils.clip_grad_norm_()`.
4. **Hard-Stop Failure Criteria:**
   - Any non-finite gradient value ($\text{NaN}$ or $\text{Inf}$) at any step triggers immediate run termination and a `CALIBRATION_FAILED_NUMERICAL_INSTABILITY` verdict.
   - If unclipped $\|\mathbf{g}\|_2 > 5.0$ on more than 2 of the final 10 steps (steps 51–60), the run is flagged as `CALIBRATION_FAILED_HIGH_VARIANCE`.
5. **Epistemic Boundaries of Short Calibration:**
   - *What 60 steps can establish:* Numerical stability under float32 CPU execution, absence of immediate gradient explosion, finiteness of target-only loss gradients at peak LR.
   - *What 60 steps cannot establish:* Long-term loss convergence, absence of delayed representation collapse at step 400, final generalization accuracy.
- **Status:** **`RESOLVED`**

---

### 2.5 VER-05: Frequency Stratification Source Correction & Tie-Breaking Policy

#### The Problem Analyzed
`DATASET_SPECIFICATION_v1.3` §2.2 proposed computing token frequencies from the Stage C validation stream (`data/tokenized/stage_c/validation/`). Deriving training entity stratification from an evaluation split introduces a conceptual methodology risk (selection/evaluation bias). Furthermore, the document lacked an explicit deterministic tie-breaking rule for tokens with equal counts.

#### Scientific & Statistical Resolution
The frequency stratification protocol is formally redirected and algorithmically specified:

1. **Definitive Source Specification:**  
   Token frequencies will be computed exclusively from the **Stage C pretraining training stream**:
   $$\text{Path: } \texttt{data/tokenized/stage_c/train/}$$
   Governed by manifest: `data/manifests/stage_c_manifest.json` (Source dataset: `DS-STAGE-C-PRETRAIN`).
   - Zero evaluation splits (`validation` or `test`) will be accessed to construct entity frequency tiers.
2. **Token Count Semantics:**  
   Count $C(t)$ represents the raw occurrence count of discrete token ID $t \in [3, 4092]$ across all training shards.
3. **Filtering Rules (Pre-Stratification):**  
   From the 4,093 active vocabulary items, exclude:
   - Reserved tokens: IDs 0, 1, 2 (`<BOS>`, `<EOS>`, `<PAD>`).
   - Whitespace and newline tokens.
   - Common single-byte punctuation delimiters (`:`, `=`, `-`, `>`, `,`, `.`).
   - High-frequency English syntactic stopwords (`the`, `is`, `at`, `in`, `and`, `to`, `of`).
   - Tokens with zero occurrences ($C(t) = 0$).
4. **Deterministic Percentile Assignment & Tie-Breaking:**  
   - Sort candidate single tokens in descending order of count $C(t)$.
   - **Deterministic Tie-Breaking Rule:** When multiple candidate tokens have identical counts ($C(t_a) = C(t_b)$), sort them strictly by **ascending Token ID**.
   - Partition the top 600 ranked single tokens into three equal-count tiers:
     - **Tier 1 (High Frequency):** Ranks 1 to 200 (Highest pretraining frequency).
     - **Tier 2 (Medium Frequency):** Ranks 201 to 400 (Moderate pretraining frequency).
     - **Tier 3 (Low Frequency):** Ranks 401 to 600 (Lowest pretraining frequency).
5. **Handling Insufficient Members:**  
   If the filtered pool contains fewer than 600 tokens with $C(t) \ge 5$, the tier size $K$ will be dynamically adjusted to $\lfloor N_{\text{filtered}} / 3 \rfloor$, preserving equal tier proportions.
6. **Governance Gating:**  
   In compliance with read-only constraints, zero binary shards were read during this pass. Frequency tiering remains formally classified as **`BLOCKED_PENDING_DATA`** until script execution is authorized.
- **Status:** **`RESOLVED`**

---

## 3. Findings Resolution & Disposition Summary Table

| Finding ID | Title / Vulnerability | Original Document & Section | v1.3.1 Disposition | Technical Action Taken | Pre-Execution Status |
|---|---|---|:---:|---|:---:|
| **VER-01** | Two-Way Cluster Bootstrap Algorithmic Specification | `SCI v1.3` §7.2; `EVL v1.3` §6 | **RESOLVED WITH SPECIFICATION** | Formally defined multiway cluster resampling algorithm ($w_m = c_K \times c_V$, $B=2,000$), seed policy, degeneracy filtering, and one-way fallback. | **RESOLVED** (Provisional algorithm locked) |
| **VER-02** | GLMM Mathematical Formulation Correction | `SCI v1.3` §7.2 | **RESOLVED** | Removed erroneous residual term $\epsilon_{ij}$. Codified canonical Bernoulli crossed random-effects logistic regression syntax. | **RESOLVED** |
| **VER-03** | Bidirectional Association Isolation | `DAT v1.3` §4 | **RESOLVED** | Enacted formal Bidirectional Association Exclusion Rule ($\mathcal{P}_{\text{sym}}(\text{Train}) \cap \mathcal{P}_{\text{dir}}(\text{Val/Test}) = \emptyset$). Proved combinatorial feasibility ($6.4\%$ capacity used). | **RESOLVED** |
| **VER-04** | Gradient Calibration Step Definition | `SCI v1.3` §8.1; `AUD v1.3` §3 | **RESOLVED** | Explicitly defined calibration duration as **60 optimizer update steps** ($240$ micro-batches), guaranteeing warmup peak traversal at step 50. Defined unclipped global $L_2$ norm. | **RESOLVED** |
| **VER-05** | Frequency Stratification Source Correction | `DAT v1.3` §2.2 | **RESOLVED** | Redirected frequency source strictly to `data/tokenized/stage_c/train/`. Formulated deterministic tie-breaking by ascending token ID. Maintained `BLOCKED_PENDING_DATA`. | **RESOLVED** |

---

## 4. Cross-Document Consistency Audit (v1.3.1)

A systematic consistency audit was executed across the CAND_01 documentation suite:

1. **Task-Family Allocations & Feasibility:**  
   - TF-01 (Direct Copy): Exactly 350 Train, 0 Val Recomb, 0 Test Recomb, 50 Test Disjoint. Consistently documented and mathematically verified across all files.
   - TF-02 through TF-08: Evenly distributed across Partitions B, C, and D.
2. **Partition Totals:**  
   - Partition A (Train): Exactly 4,000 records.
   - Partition B (Val Recomb): Exactly 500 records.
   - Partition C (Test Recomb): Exactly 500 records.
   - Partition D (Test Disjoint): Exactly 500 records.
   - Total Corpus: Exactly 5,500 records. Uniform across all documents.
3. **Factorial Arms:**  
   - Arms A, B, C, D defined identically. Arm C uniformly classified as a composite packaging intervention; Arm D uniformly classified as an exploratory historical reference.
4. **Evaluator Contract:**  
   - Coupled $A_{\text{FT}} \ge 75.0\%$ and $A_{\text{FC}} \ge 70.0\%$ [PROVISIONAL] maintained uniformly. Two-token completion contract ($[TARGET, EOS]$) locked across all specifications.
5. **Execution Authorization:**  
   - Uniformly and prominently marked **`UNAPPROVED`** in all headers, sections, checklists, and conclusions.

---

## 5. Implementation Blockers & Pre-Execution Roadmap

The formal gating structure remains intact. CAND_01 execution remains blocked pending the following sequential milestones:

```
[ PRE-EXECUTION GO GATES ]
  ├── 1. Maintainer Sign-off on v1.3.1 Documentation Package ──────────────────────── [CURRENT MILESTONE]
  ├── 2. Separately Authorized Token Frequency Histogram Extraction (GAP-08) ──────── [BLOCKED_PENDING_DATA]
  ├── 3. Separately Authorized 60-Step Pre-Flight Gradient Calibration (GAP-07) ────── [BLOCKED_PENDING_AUTH]
  ├── 4. Dataset Generator Implementation (GAP-01: scripts/generate_cand01_dataset.py) ── [BLOCKED]
  ├── 5. Benchmark Evaluator & Unit Tests (GAP-02 & GAP-03) ───────────────────────── [BLOCKED]
  ├── 6. Non-Interactive Runner Implementation (GAP-04 & GAP-05) ───────────────────── [BLOCKED]
  ├── 7. Python Subshell Environment Verification (GAP-06) ─────────────────────────── [BLOCKED]
  └── 8. Formal Pre-Execution GO Record Sign-off (Protocol v1.0 Section 14) ─────────── [BLOCKED]
```

---

## 6. Document Governance & Final Posture

- **Documentation Readiness:** **`READY_FOR_PRE_EXECUTION_REVIEW`**  
  (All mathematical contradictions, statistical misspecifications, and leakage vulnerabilities are formally resolved.)
- **Implementation Readiness:** **`NOT_READY`**  
  (Gaps 01 through 08 remain unimplemented in code.)
- **Dataset Generation Posture:** **`UNAUTHORIZED`**
- **Calibration Posture:** **`UNAUTHORIZED`**
- **Experiment Execution Posture:** **`UNAPPROVED`**

*(This register forms the authoritative closure record for the CAND_01 v1.3.1 scientific review pass.)*

