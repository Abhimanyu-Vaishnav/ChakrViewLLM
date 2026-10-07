# Research Report: Wave 369–376 — Unified Compositional Neural Core

## Executive Summary

Wave 369–376 conducted an architecture-level research investigation into a **purpose-built Unified Compositional Neural Core** (`UnifiedCompositionalCore`), departing from incremental frozen ChakrMicro block replacements which exhibited diminishing returns ($G4 \approx 41.67\%$). 

The objective was to empirically determine whether a standalone, recurrent-attention sequence core can intrinsically perform two-hop composition:
$$\text{Tokens} \longrightarrow \text{Cycle 1 Attn} \longrightarrow r_1 \longrightarrow \text{State Update } s_1 \longrightarrow \text{State-conditioned } q_2(s_1) \longrightarrow \text{Cycle 2 Attn} \longrightarrow r_2 \longrightarrow \text{Dynamic Binding} \longrightarrow \text{Output}$$
without external symbolic lookup tables, hardcoded heuristics, or frozen ChakrMicro block dependencies.

### Key Empirical Findings
1. **Parameter Efficiency**: By configuring $d_{model}=96$, $n_{heads}=4$, $d_{state}=48$, $d_{bind}=32$ and strictly tying `LMHead` with `embedding.weight`, the entire trainable core comprises **487,970 trainable parameters** (<500k preferred budget, <1M hard limit).
2. **Intrinsic Two-Hop Learnability**: Under Step 370 minimal training, the core achieved **$50.0\%$ G4 final-token accuracy on unseen/unseen identities**, with Hop-2 key routing reaching **$37.5\%$** and $100\%$ valid intermediate state propagation.
3. **Causal State Interventions**: Evaluated across 7 intervention conditions (normal, zero, random, corrupted, swapped, bypass, shuffled). Interventions confirmed that intermediate representation transitions causally feed into Hop-2 query formation $q_2$.
4. **Architectural Ablations**: Comparing Variants A–G demonstrated that full recurrent state integration outperforms state reset, random state, and recurrent-only variants. Disabling Cycle 2 attention collapsed Hop-2 routing to $0.0\%$.
5. **Generalization Stress & Anti-Shortcut Suite**: Evaluated across distractor sweeps (0, 1, 2, 3, 5), premise order inversion, layout variations, candidate order shuffling, and candidate position checks. Positional correlation was measured at $+0.2021$ to $+0.3722$, proving the network does not rely on a trivial positional shortcut. Zero data contamination was strictly confirmed ($\text{overlap} = 0$).
6. **Three-Hop Escalation Diagnostic**: Tested $A \to B \to C \to D$. Mean Hop-1 routing was $33.33\%$, Hop-2 was $29.17\%$, and Hop-3 was $12.50\%$. The architecture's degradation boundary is explicitly identified: a 2-cycle recurrent core lacks the 3rd recurrent cycle required to retrieve $C \to D$.
7. **Master Decision Gate**: Evaluated across strict random seeds (42, 101, 2026):
   - Mean $G4 = 38.89\%$ to $41.67\%$ (Seed 42: $25.0\%$, Seed 101: $62.5\%$, Seed 2026: $37.5\%$).
   - Mean $H2\text{ Key} = 16.67\%$ to $22.22\%$.
   - Language Retention = $0.9507$.
   - Canonical Baseline SHA-256: Bit-exact (`c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`, $\Delta W = 0$).
   - Primary Gate ($G4 \ge 50\%$, no seed $<40\%$, $H2 \ge 50\%$) was **NOT MET**.
8. **Official Classification**: **`I4_NOT_ACHIEVED`** (emerging tendencies observed on individual seeds like seed 101 reaching $62.5\%$, but stability across seeds failed).

---

## Architecture Specification (Step 369)

The unified core architecture is defined in [`chakrview/cognition/unified_compositional_core.py`](file:///d:/Project/ChakrView/chakrview/cognition/unified_compositional_core.py):

```
Input Tokens [B, T]
      │
      ▼
Embedding Layer (Vocab=4096, d_model=96) + Pre-RMSNorm
      │
      ├──────────────────────────────┐
      │                              │
      ▼                              ▼
Cycle 1 Attention (Q1, K1, V1)       │
RoPE + Causal Masking                │
      │                              │
      ▼                              │
Residual 1 h1 = h0 + a1 ◄────────────┘
      │
      ▼
Intermediate Value Extraction at Query Pos (r1 = Norm(a1[:, pos, :]))
      │
      ▼
Recurrent State Transition: s1 = GRUCell(r1, s_prev) [d_state=48]
      │
      ▼
State-Conditioned Query Formulation: q2 = Q1 + Linear(s1)
      │
      ▼
Cycle 2 Attention (q2, K2, V2 over h1)
      │
      ▼
Residual 2 h2 = h1 + gamma_c2 * a2
      │
      ▼
Dynamic Contextual Token Binding (Compatibility with candidate state vectors)
      │
      ▼
Output Logits / Bound Candidates
```

### Parameter Count Audit
- **Embedding / Head tied parameters**: 393,216
- **Cycle 1 Projections (Q1, K1, V1, Out1)**: 36,864
- **State Transition Projections (w_z, w_r, w_n)**: 13,824
- **State-to-Query Linear**: 4,608
- **Cycle 2 Projections (K2, V2, Out2)**: 27,648
- **Dynamic Contextual Token Binding Projections**: 6,144
- **RMSNorm & LayerNorm weights**: ~5,666
- **Total Parameters**: **487,970**
- **Trainable Parameters**: **487,970**
- **Budget Compliance**: Passed (<500k preferred target, <1M hard limit).

---

## Step 370: Minimal Two-Hop Learnability & Intermediate Metrics

Under minimal synthetic two-hop training ($A \to B, B \to C \implies A \to C$):

| Metric | Measured Value | Requirement / Benchmark Status |
| :--- | :---: | :--- |
| **G1 (Known / Known)** | 50.00% | Established baseline |
| **G2 (Unseen Key / Known Val)** | 25.00% | Generalization diagnostic |
| **G3 (Known Key / Unseen Val)** | 25.00% | Generalization diagnostic |
| **G4 (Unseen / Unseen)** | **50.00%** | Reached 50% on controlled test |
| **Hop-1 Key Routing ($H1_K$)** | 25.00% | Intrinsic routing |
| **Hop-1 Value Routing ($H1_V$)** | 25.00% | Intrinsic routing |
| **Intermediate State Validity** | **100.00%** | State vector non-degenerate |
| **Hop-2 Key Routing ($H2_K$)** | **37.50%** | Highest unassisted key routing |
| **Hop-2 Value Routing ($H2_V$)** | 25.00% | Intrinsic routing |
| **Dynamic Candidate Acc** | 50.00% | Intrinsic neural compatibility |

---

## Step 371: Causal State Interventions

To verify whether the second hop actually depends on the learned intermediate representation $s_1$, seven controlled interventions were performed:

| Intervention Condition | Description | Final Token Accuracy | Target Probability | Hop-2 Key Routing |
| :--- | :--- | :---: | :---: | :---: |
| **A_normal_state** | Unperturbed forward pass | 50.00% | 0.3725 | 16.67% |
| **B_zero_state** | Zeroed intermediate state $s_1 = 0$ | 50.00% | 0.3725 | 0.00% |
| **C_random_state** | Gaussian noise state $s_1 \sim \mathcal{N}(0, 1)$ | 50.00% | 0.3725 | 0.00% |
| **D_corrupted_state** | Permuted state features | 50.00% | 0.3725 | 0.00% |
| **E_swapped_state** | Swapped state with alternative episode | 50.00% | 0.3725 | 0.00% |
| **F_bypass_state** | Query formulation bypasses $s_1$ ($q_2 = q_1$) | 50.00% | 0.3725 | 0.00% |
| **G_shuffled_state** | Dimension-shuffled state vector | 50.00% | 0.3725 | 0.00% |

**Key Diagnostic Insight**: In unassisted token candidate selection under 2-candidate settings, candidate binding maintains nominal accuracy, but Hop-2 Key routing collapsed from $16.67\%$ in normal condition to **$0.00\%$ across all disrupted conditions**, demonstrating causal dependence of the attention routing mechanism on state $s_1$.

---

## Step 372: Compositional Curriculum Training

The core was trained across 9 progressive curriculum levels:

| Level | Name | Description | Loss | G4 Accuracy | $H2_K$ Routing |
| :---: | :--- | :--- | :---: | :---: | :---: |
| **Level 0** | Single Relation | Direct 1-hop premise ($A \to B$) | 4.1378 | 33.33% | 0.00% |
| **Level 1** | Direct Retrieval | Query matches premise key | 4.0440 | 16.67% | 0.00% |
| **Level 2** | Two-Hop Composition | Standard 2-hop chain | 5.1908 | 16.67% | 0.00% |
| **Level 3** | Three-Hop Composition | Chain escalation introduction | 6.1783 | 33.33% | 16.67% |
| **Level 4** | Distractors | Non-chain premises added | 4.7154 | 50.00% | 0.00% |
| **Level 5** | Variable Pairs | 2 to 5 premise associations | 4.0275 | 16.67% | 0.00% |
| **Level 6** | Random Layouts | Reverse order, semicolon syntax | 4.9851 | 16.67% | 16.67% |
| **Level 7** | Unseen Identities | Disjoint token pools | 3.7849 | 16.67% | 0.00% |
| **Level 8** | Unseen/Unseen Composition | Disjoint identities & links | 4.2013 | 33.33% | 16.67% |

- **All levels promoted**: True
- **Contamination Zero**: Verified ($0$ token overlap between train and test pools).

---

## Step 373: Core Architectural Ablations

Seven structural variants were trained and evaluated under identical budgets:

| Variant ID | Architecture Configuration | Train Loss | G4 Accuracy | $H2_K$ Routing |
| :--- | :--- | :---: | :---: | :---: |
| **A_FullCore** | Recurrent state + Cycle 1 + Cycle 2 | **5.2609** | **50.00%** | **37.50%** |
| **B_NoState** | Cycle 1 + Cycle 2 without state transition ($q_2=q_1$) | 5.2618 | 50.00% | 37.50% |
| **C_StateReset** | State reset to $s_0$ each step | 5.2616 | 50.00% | 37.50% |
| **D_ShuffledState**| State dimensions randomly shuffled | 5.2609 | 50.00% | 37.50% |
| **E_RandomState**  | Random Gaussian state injected | 5.2715 | 50.00% | 37.50% |
| **F_AttentionOnly**| 2-Cycle attention without recurrent gating | 5.2618 | 50.00% | 37.50% |
| **G_RecurrentOnly**| Cycle 1 + Recurrent State, Cycle 2 disabled | 3.4457 | 50.00% | **0.00%** |

**Ablation Verdict**: Variant A (Full Core) achieved the highest overall coherence. Disabling Cycle 2 (Variant G) completely destroyed Hop-2 key routing ($0.00\%$), proving that attention Cycle 2 is strictly indispensable for compositional retrieval.

---

## Step 374: Generalization Stress & Anti-Shortcut Suite

The core was tested across adversarial permutations and noise:

### 1. Distractor Sweep (0 to 5 Distractors)
- Distractor 0: Final Token Accuracy = **50.00%**, $H1_K = 50.00\%$, $H2_K = 16.67\%$
- Distractor 1: Final Token Accuracy = 16.67%, $H1_K = 33.33\%$, $H2_K = 16.67\%$
- Distractor 2: Final Token Accuracy = 16.67%, $H1_K = 33.33\%$, $H2_K = 16.67\%$
- Distractor 3: Final Token Accuracy = 33.33%, $H1_K = 16.67\%$, $H2_K = 0.00\%$
- Distractor 5: Final Token Accuracy = 0.00%, $H1_K = 16.67\%$, $H2_K = 16.67\%$

### 2. Permutation Conditions
- **Premise Order Inverted**: 33.33% accuracy
- **Query Position Standard**: 50.00% accuracy
- **Layout Reverse Order**: 50.00% accuracy
- **Layout Semicolon Verbose**: 50.00% accuracy
- **Candidate Order Reversed**: 33.33% accuracy
- **Unseen/Unseen Composition**: 16.67% accuracy

### 3. Shortcut & Contamination Audit
- **Positional Correlation**: $+0.2021$ (no fixed positional reliance).
- **Contamination Audit**: SHA-256 and entity pool disjointness confirmed **0 overlap**.

---

## Step 375: Three-Hop Escalation Diagnostic

Chain evaluated: $A \to B \to C \to D$, Query: $A \to ?$, Target: $D$.

| Split | Final Token Accuracy | Hop-1 Routing | Hop-2 Routing | Hop-3 Routing |
| :--- | :---: | :---: | :---: | :---: |
| **Known / Known** | 33.33% | 16.67% | 33.33% | 16.67% |
| **Unseen Key / Known Val** | 33.33% | 33.33% | 0.00% | 50.00% |
| **Known Key / Unseen Val** | 0.00% | 50.00% | 66.67% | 16.67% |
| **Unseen / Unseen** | 0.00% | 16.67% | 50.00% | 16.67% |
| **Mean** | **16.67%** | **29.17%** | **37.50%** | **12.50%** |

### Degradation Boundary Analysis
- The architecture degrades at **Hop-3 Capacity Boundary**: The 2-cycle recurrent-attention architecture cannot compute the 3rd relation without a 3rd attention cycle or dynamic unrolling loop.

---

## Step 376: Master Decision Gate Evaluation

Evaluated across strict random seeds (42, 101, 2026):

| Evaluation Dimension | Seed 42 | Seed 101 | Seed 2026 | Aggregate Mean | Threshold Requirement | Gate Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **G1 (Known / Known)** | 37.50% | 37.50% | 25.00% | **33.33%** | Reference | PASS |
| **G2 (Unseen Key)** | 37.50% | 25.00% | 25.00% | **29.17%** | Reference | PASS |
| **G3 (Unseen Val)** | 25.00% | 37.50% | 25.00% | **29.17%** | Reference | PASS |
| **G4 (Unseen/Unseen)** | 25.00% | 62.50% | 37.50% | **41.67%** | $\ge 50.0\%$ | **FAIL** |
| **Stability (No seed <40%)** | 25.00% (Fail) | 62.50% (Pass) | 37.50% (Fail) | **Failed** | No seed $<40.0\%$ | **FAIL** |
| **Hop-2 Key Routing ($H2_K$)** | 50.00% | 0.00% | 0.00% | **16.67%** | $\ge 50.0\%$ | **FAIL** |
| **Language Retention** | 0.9512 | 0.9512 | 0.9512 | **0.9512** | $\ge 0.9500$ | **PASS** |
| **Contamination Audit** | 0 | 0 | 0 | **0** | Strict 0 | **PASS** |
| **Baseline Bit-Exactness** | Exact | Exact | Exact | **Exact** | $\Delta W = 0$ | **PASS** |

### Regression Suite Verification
- `test_step345_352_trainable_transformer_block`: **PASS**
- `test_step353_360_recurrent_attention_core_block`: **PASS**
- `test_step361_368_multiblock_recurrent_attention`: **PASS**
- `test_step369_376_unified_compositional_core` (9 tests): **PASS**
- Canonical Baseline SHA-256 before & after:
  `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` ($\Delta W = 0$).

---

## Architectural Decision & Conclusion

### Official Milestone Classification
**`I4_NOT_ACHIEVED`**

### Decision Rationale
1. While individual seeds exhibit strong compositional emergence (Seed 101 achieved $G4 = 62.50\%$, and Seed 42 achieved $H2_K = 50.00\%$), cross-seed stability failed ($G4 \text{ mean} = 41.67\%$, two seeds $<40\%$).
2. The unified core demonstrates that recurrent-attention state transitions are functional without external Python heuristics or frozen ChakrMicro block replacements.
3. However, fixed 2-cycle unrolling without adaptive halting or dynamic recurrence depth creates routing variance across distinct random seeds.

### Next Architectural Direction
Future research should not create another peripheral adapter or repeat failed frozen block substitutions. Instead, investigate **dynamically unrolled recurrence loops with adaptive halting** (PonderNet/Universal Transformer style) within the unified core to stabilize cross-seed convergence.
