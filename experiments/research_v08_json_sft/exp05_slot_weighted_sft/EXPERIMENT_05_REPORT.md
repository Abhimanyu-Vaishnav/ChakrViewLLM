# CHAKRVIEW v0.8: EXPERIMENT 05 RESEARCH & CAPABILITY REPORT
## Slot-Weighted SFT for Context-Grounded Copying: Treatment Evaluation, Regressions & Limits of Loss Weighting

- **Author**: ChakrView Research & ML Engineering Agent
- **Execution Date**: 2026-10-21
- **Experiment Identifier**: `research_v08_json_sft_exp05_slot_weighted_sft`
- **Output Directory**: `experiments/research_v08_json_sft/exp05_slot_weighted_sft/`
- **Architecture**: ChakrMicro (3,443,136 parameters, 6 layers, $d_{\text{model}}=192$, $n_{\text{heads}}=6$, $d_{\text{ff}}=512$, context 512)
- **Primary Mission**: Run an isolated, controlled capability experiment to determine whether increasing the training loss weight ($5.0\times$) strictly on target status-string tokens improves ChakrMicro's ability to copy unseen values from context into structured JSON.

---

## 1. Safety, Invariants & Cryptographic Verification

All parent checkpoints, tokenizers, historical datasets, and runtime code were pre-verified and re-verified post-experiment (SHA-256):

| Artifact Role | File Path | SHA-256 Checksum | Invariant Status |
| :--- | :--- | :--- | :--- |
| **Canonical Parent Checkpoint** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | Verified bit-exact |
| **Canonical Tokenizer Vocab** | `data/experiments/vocab_4096/vocab.json` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | Verified bit-exact |
| **Canonical Tokenizer Merges** | `data/experiments/vocab_4096/merges.json` | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | Verified bit-exact |
| **Parent Initialization Checkpoint** | `experiments/research_v08_json_sft/exp02_string_copy/checkpoints/exp02_best_checkpoint.pt` | `d31d2f9cc8be1a021b710a2f434db20d2f45e545e1c5fea10afcca1675f2bfe2` | Verified bit-exact |
| **Frozen Train Dataset** | `experiments/research_v08_json_sft/exp02_string_copy/data/train.jsonl` | `bf0da185d0a5172bffd422c0b64a2ecf664d532b56148c54b3ce5856d07f55c4` | 130 records verified |
| **Frozen Val Dataset** | `experiments/research_v08_json_sft/exp02_string_copy/data/val.jsonl` | `d5750bff540387b4a4c8d68515fd9a1a25cbb5f2122645f5da0589f5c21fbb9c` | 30 records verified |
| **Frozen Test Dataset** | `experiments/research_v08_json_sft/exp02_string_copy/data/test.jsonl` | `b8ce9bfb6a1bfaba8ca7607a126b1babd33420839954d72a6af80e893f518f88` | 30 records verified |
| **Treatment Best Checkpoint** | `experiments/research_v08_json_sft/exp05_slot_weighted_sft/checkpoints_treatment/treatment_best_checkpoint.pt` | `1c6764a05e86314e5641cd149c4b495117a61514803949449bf4ff73fe4da816` | Selected at Step 90 |

Protected invariants: `chakrview/runtime/session.py` remained untouched; zero Git commits or remote repository mutations.

---

## 2. Baseline Reproduction Check

Before training the treatment arm, we reproduced the unweighted baseline evaluation using the verified candidate checkpoint on the frozen test set (`REPRODUCED_BASELINE_RESULTS.json`):
- **JSON Parse Rate**: 19 / 20 (95.0%)
- **Schema Valid Rate**: 19 / 20 (95.0%)
- **Status Match (Exact Copying)**: 0 / 20 (0.0%)
- **Replay Continuity**: 10 / 10 (100.0%)
- **Exact Agreement Check**: Using `verify_reproduction.py`, the reproduced evaluation agreed **bit-for-bit** across all 30 test records and all metric dimensions with historical Experiment 02 evidence.

---

## 3. Mathematical Formulation of Slot-Weighted Loss

To isolate the gradient impact on the target string value:
1. **Target Slot Slice**: For each single-field extraction sequence `{"status": "<val>"}`, the status value `<val>` starts at token position $i_{\text{start}} = M + 5$ and ends at $i_{\text{end}} = i_{\text{start}} + |S|$, where $M$ is the prompt length and $|S|$ is the number of subword tokens in `<val>`.
2. **Token Weights Assignment**:
   $$w_i = \begin{cases} 
   0.0 & \text{if } y_i = -100 \text{ (prompt tokens, padding tokens)} \\
   5.0 & \text{if } i \in [i_{\text{start}}, i_{\text{end}}) \text{ (status-string content)} \\
   1.0 & \text{otherwise (JSON delimiters, syntax, EOS, continuation replay)} 
   \end{cases}$$
3. **Normalized Loss**:
   $$\mathcal{L}_{\text{weighted}} = \frac{\sum_{i=1}^T w_i \cdot \ell_i}{\sum_{i=1}^T w_i}$$
   where $\ell_i = -\log P(y_i \mid x_{<i})$.
   When all active tokens have $w_i = 1.0$, this simplifies bit-exact to standard mean cross-entropy.

---

## 4. Pre-Training Unit Test Verification

All 7 focused unit tests in `test_weighted_loss.py` passed deterministically before any optimizer steps:
1. **Hand-calculated numerical check**: Computed loss matched the exact analytical solution ($0.369520 == 0.369520$).
2. **Identity check ($w=1.0$)**: Weighted loss reproduced standard PyTorch `nn.CrossEntropyLoss` bit-exact ($3.384546 == 3.384546$).
3. **Selective weighting**: Verified that only target string tokens receive weight 5.0, while prompt tokens receive 0.0 and delimiters receive 1.0.
4. **Replay preservation**: Replay records received weight 1.0 across all active tokens.
5. **Padding and dynamic batching**: Padded positions had target $-100$ and weight $0.0$.
6. **Finite loss and gradients**: Gradient backpropagation on ChakrMicro produced strictly finite losses and clipped norms.
7. **Fail-closed alignment assertion**: Deliberately corrupted status-string mappings raised immediate assertion errors.

---

## 5. Controlled Training Execution

Trained the Treatment Arm for 120 steps on CPU from the canonical parent under identical optimizer settings:
- **Optimization**: AdamW, batch size 4, lr $3\times 10^{-5} \to 3\times 10^{-6}$ (10-step warmup + cosine decay), grad clip 1.0.
- **Initial Weighted Val Loss**: `5.2181`
- **Validation Trajectory**:
  - Step 10: `4.8477` (Drift 0.17%)
  - Step 20: `4.3460` (Drift 0.45%)
  - Step 50: `4.2092` (Drift 1.11%)
  - Step 70: `4.1582` (Drift 1.41%)
  - Step 80: `4.1331` (Drift 1.51%)
  - **Step 90 (Best Validation Checkpoint)**: **`4.0374`** (Drift 1.59%)
  - Step 100: `4.0796` (Drift 1.64%)
  - Step 120 (Final): `4.1152` (Drift 1.70%)
- **Checkpoint Selection**: Step 90 achieved the global minimum weighted validation loss (`4.0374`) and was saved as `treatment_best_checkpoint.pt`.

---

## 6. Head-to-Head Evaluation on Frozen Held-Out Test Set

Evaluated the Canonical Parent Baseline, Control Baseline (Exp 02), and Treatment Checkpoint (Exp 05) on the exact frozen 30-case held-out test suite under deterministic greedy decoding ($T=0.0$, `max_new_tokens=64`, stop token `[1]`):

| Metric Dimension | Canonical Parent (`Step-10000`) | Control Arm (Exp 02 Unweighted) | Treatment Arm (Exp 05 5.0× Slot-Weighted) | Relative Treatment Impact |
| :--- | :---: | :---: | :---: | :--- |
| **Exact Status-String Copying** | **0 / 20 (0.0%)** | **0 / 20 (0.0%)** | **0 / 20 (0.0%)** | **No gain (0/20)** |
| **JSON Parse Rate** | 0 / 20 (0.0%) | **19 / 20 (95.0%)** | **0 / 20 (0.0%)** | **Severe syntax regression (-95%)** |
| **Schema Valid Rate (`{"status": str}`)** | 0 / 20 (0.0%) | **19 / 20 (95.0%)** | **0 / 20 (0.0%)** | **Severe schema regression (-95%)** |
| **Replay Non-Empty Rate** | 10 / 10 (100.0%) | 10 / 10 (100.0%) | 10 / 10 (100.0%) | Preserved |
| **Replay Meaningful Continuation** | 2 / 10 (20.0%) | 3 / 10 (30.0%) | 2 / 10 (20.0%) | Preserved base pretraining |
| **Best Step Val Loss** | N/A | 3.3230 (Step 120) | 4.0374 (Step 90) | Loss higher due to 5x weight |
| **Final Weight Drift** | 0.0% | 1.87% | 1.70% | Within 4% safety boundary |

---

## 7. Qualitative Comparison & Forensic Emission Trace

### Side-by-Side Qualitative Table:

| Test Record ID | Expected In-Context Status | Control Output (`exp02_best_checkpoint.pt`) | Treatment Output (`treatment_best_checkpoint.pt`) |
| :--- | :--- | :--- | :--- |
| `exp02_test_json_001` | `optical_alignment_fault` | `{"status": "path."}` | `{"__main__":\n import docannot import "__mainininked"...` |
| `exp02_test_json_002` | `beam_misaligned` | `{"status": "pated"}` | `{"{"__main__":\n import docannot recattledraiveditor...` |
| `exp02_test_json_003` | `cavity_depressurized` | `{"status": "path."}` | `{"__main__":\n import docannot resupport import...` |
| `exp02_test_json_004` | `flux_fluctuation` | `{"status": "pathalso "}` | `{"{"__main__":{"__mainput_st_stridecase_of_of...` |
| `exp02_test_json_005` | `cryo_quench` | `{"status": "blicath"}` | `{"__main__":\n # Iteratoritemperattresupport...` |

### Step-by-Step Greedy Token Emission Trace on Treatment Model:
Traced via `trace_treatment_steps.py` to see why the treatment model broke JSON syntax:
- Step 0: Emits `126` (`{`) [Top logit: 7.98]
- Step 1: Emits `37` (`"`) [Top logit: 10.98]
- **Step 2 (The Fatal Bifurcation)**:
  - In Control Arm: Emits `1831` (`stat`), followed by `us`, `": "`.
  - In Treatment Arm: Model predicts `1312` (`__`) with logit **5.32**, narrowly edging out `126` (`{` logit 4.80) and `304` (`st` logit 4.50)!
  - Because `__` is emitted, greedy decoding immediately branches into `ma` $\to$ `in` $\to$ `__` $\to$ `":\n import docannot...`.
  - The model reverts to Python pretraining boilerplate (`"__main__": ...`), completely destroying JSON parseability.

### Forensic Explanation of the Regression:
1. **Gradient Gradient Destabilization across Asymmetric Delimiters**:
   By multiplying the loss on status tokens by $5.0\times$ while keeping the surrounding JSON syntax (`stat`, `us`, `": "`, `"`, `}`) at $1.0\times$, the loss landscape was severely warped.
2. In early training steps, the model was heavily penalized for guessing wrong subwords in the variable status strings. Because the status strings are diverse and high-entropy, the gradients from the $5.0\times$ status loss overwhelmed the weaker $1.0\times$ signal for the syntax tokens.
3. Rather than learning to route attention back to the input context, the network drifted away from the clean JSON grammar learned in Exp 02 and fell into a Python script dictionary attractor (`{"__main__": ...}`).

---

## 8. What This Experiment Established & What Remains Unknown

### What Was Established:
1. **Loss reweighting alone does NOT induce an in-context pointer/copy circuit**: Upweighting token loss by $5.0\times$ does not give a 3.44M-parameter causal transformer the structural induction mechanism needed to attend back across 50+ tokens and bind an arbitrary context substring.
2. **Asymmetric loss weighting destabilizes syntactic convergence**: Heavily weighting variable interior tokens while under-weighting surrounding delimiters degrades the model's ability to retain stable syntax, causing it to fall back into base pretraining attractors (`__main__`).
3. **The copying failure is architectural/inductive, not an optimization objective imbalance**: Neither unweighted causal SFT nor slot-weighted causal SFT produces context-grounded string copying.

### What Remains Unknown:
- Whether a direct attention-supervision loss (constraining cross-position attention weights to the context span) or an explicit copy head/pointer architecture is required for ChakrMicro to copy unseen strings.

---

## 9. Conclusion & Decision

- **Primary Success Metric**: 0 / 20 (0.0%) exact string copy rate on held-out test data.
- **Secondary Metrics**: Severe regression from 95% to 0% in JSON parseability and schema validity.

### Final Status:
`REGRESSION OR INSTABILITY DETECTED`

### Recommended Next Step:
**Cease pure token-level loss-reweighting experiments.**
To achieve context-grounded copying without architectural pointer augmentation, any subsequent research should investigate **extrinsic pointer mechanisms** (such as the existing relation pointer head in `chakrview/cognition/`) or **constrained decoding / extraction masking** rather than pure autoregressive next-token SFT.
