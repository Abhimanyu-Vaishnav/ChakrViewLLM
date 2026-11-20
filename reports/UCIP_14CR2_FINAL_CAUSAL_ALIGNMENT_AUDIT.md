# UCIP-14C-R2: Final Causal Target Alignment & Indexing Audit
## Independent Mathematical and Implementation Verification of Assistant-Only Causal Loss

**Mission ID:** UCIP-14C-R2  
**Project:** ChakrView  
**Repository:** `D:\Project\ChakrView`  
**Active Branch:** `exp/cbr-01-conversational-breakthrough`  
**Starting Commit:** `562d099` (verified)  
**Execution Mode:** Independent Mathematical & Implementation Audit  
**Training Status:** **ZERO TRAINING CONDUCTED (STRICTLY PROHIBITED)**  
**Dataset Construction:** **ZERO SFT DATASET GENERATED (PROHIBITED)**  
**Final Scientific Verdict:** **VERIFIED**  

---

## 1. Executive Summary: Resolution of the Indexing Ambiguity

In the previous UCIP-14C-R audit, an apparent contradiction was noted:
> *If logit position 5 predicts the first assistant token `Hel`, but the reported target sequence places `Hel` at target index 5, does the implementation perform standard runtime causal shifting (`logits[:, :-1]` vs `targets[:, 1:]`), which would pair logit position 5 with target position 6 rather than position 5?*

### The Resolution:
1. **ChakrView Uses Pre-Shifted Next-Token Target Pairing in the Dataset Loader:**
   ChakrView's `sft_engine.py` (lines 33–59, `build_causal_sequence`) constructs target sequences that are **already pre-shifted** by exactly 1 position relative to the causal input sequence:
   $$\text{target\_seq}[i] = \text{next\_token}(\text{input\_seq}[i])$$
   Therefore, at loss evaluation time, the model executes a **direct pointwise cross-entropy loss**:
   $$\mathcal{L} = \text{CrossEntropyLoss}(\text{logits}[i], \text{target\_seq}[i])$$
   **No runtime causal shifting (`logits[:, :-1]` vs `targets[:, 1:]`) is performed in `sft_engine.py`.**
2. **Mathematical Equivalence to Standard Runtime Causal Shifting (HuggingFace / PyTorch convention):**
   In the standard runtime shift convention:
   - Full sequence: $[x_0, x_1, x_2, \dots, x_N]$ (length $N+1$).
   - Full labels: $[y_0, y_1, y_2, \dots, y_N]$ (length $N+1$).
   - Runtime shift slices: $\text{logits} = \text{model}(x)[:, :-1]$ (length $N$) and $\text{shift\_labels} = y[:, 1:]$ (length $N$).
   - Index $i$ of `shift_labels` is $y[i+1]$.
   
   In ChakrView's pre-shifted convention:
   - $\text{target\_seq}[i]$ is constructed directly as $y[i+1]$ during dataloading.
   - Both approaches produce **100% bit-exact loss values, identical active loss positions, and identical gradient norms**.
3. **No Implementation Defect Exists:**
   The production code in `chakrview/training/sft_engine.py` is **mathematically sound and fully correct**. The apparent contradiction was purely a terminology distinction between *runtime shifted slicing* versus *pre-shifted dataloader construction*.

---

## 2. Source Code Inspection & Line Numbers

### A. Dataloader Sequence Construction (`chakrview/training/sft_engine.py`, lines 33–59)
```python
def build_causal_sequence(
    prompt_tokens: List[int],
    response_tokens: List[int],
    bos_id: int = 0,
    eos_id: int = 1,
    ignore_index: int = -100,
    mask_prompt: bool = True,
) -> Tuple[List[int], List[int]]:
    input_seq = [bos_id] + prompt_tokens + response_tokens
    if mask_prompt:
        target_seq = ([ignore_index] * len(prompt_tokens)) + response_tokens + [eos_id]
    else:
        target_seq = prompt_tokens + response_tokens + [eos_id]
    assert len(input_seq) == len(target_seq)
    return input_seq, target_seq
```
- Line 53: `input_seq` has length $1 + P + R$.
- Line 55: `target_seq` has length $P + R + 1 = 1 + P + R$.
- Line 58: `assert len(input_seq) == len(target_seq)` proves both tensors have identical sequence length.

### B. Training Loss Execution (`chakrview/training/sft_engine.py`, lines 354–355)
```python
logits = self.model(x)
loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
```
- `logits` shape: `[B, T, V]`
- `y` (`target_seq`) shape: `[B, T]`
- `self.loss_fn`: `nn.CrossEntropyLoss(ignore_index=-100)`
- **Observation:** Notice that `logits` is NOT sliced (`[:, :-1]`) and `y` is NOT sliced (`[:, 1:]`). Direct element-wise pairing $(\text{logits}[:, i], y[:, i])$ is executed.

### C. Pretraining Loss Execution (`chakrview/training/dataset.py`, lines 96–97 and `loss.py`, line 53)
In Stage C pretraining, the binary token stream also pre-shifts targets during chunking:
```python
# dataset.py lines 96-97:
input_ids = torch.from_numpy(chunk[:-1].astype(np.int64))
target_ids = torch.from_numpy(chunk[1:].astype(np.int64))

# loss.py line 53:
loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=self.ignore_index)
```
Across both pretraining and SFT, ChakrView consistently adheres to the **pre-shifted dataset pairing convention**.

---

## 3. Three-Way Verification: Detailed Causal Alignment Table

### Test Case:
- **Prompt:** `User: Hi` (Tokens: `[88, 835, 324, 75, 108]`, length $P=5$)
- **Response:** `Hello` (Tokens: `[3860, 317]`, length $R=2$)
- Followed by 2 padding tokens (`<PAD>=2`, target `-100`).

| Index $i$ | Input ID $X_i$ | Input Text | Unshifted Label $Y^{\text{raw}}_i$ | Pre-Shifted Target $Y_i$ | Target Text | Active Loss? | Logit Gradient Norm | Causal Prediction Meaning |
| :---: | :---: | :--- | :---: | :---: | :--- | :---: | :---: | :--- |
| **0** | `0` | `'<BOS>'` | `-100` | `-100` | `<IGNORE>` | **NO** | `0.000000` | Prompt token masked |
| **1** | `88` | `'U'` | `-100` | `-100` | `<IGNORE>` | **NO** | `0.000000` | Prompt token masked |
| **2** | `835` | `'ser'` | `-100` | `-100` | `<IGNORE>` | **NO** | `0.000000` | Prompt token masked |
| **3** | `324` | `': '` | `-100` | `-100` | `<IGNORE>` | **NO** | `0.000000` | Prompt token masked |
| **4** | `75` | `'H'` | `-100` | `-100` | `<IGNORE>` | **NO** | `0.000000` | Prompt token masked |
| **5** | `108` | `'i'` | `-100` | `3860` | `'Hel'` | **YES** | **`0.666552`** | **Logit from final prompt token predicts first assistant token** |
| **6** | `3860` | `'Hel'` | `3860` | `317` | `'lo'` | **YES** | **`0.666569`** | Logit from assistant token predicts next assistant token |
| **7** | `317` | `'lo'` | `317` | `1` | `'<EOS>'` | **YES** | **`0.666555`** | Logit from final assistant token predicts terminal `<EOS>` |
| **8** | `2` | `'<PAD>'` | `1` | `-100` | `<IGNORE>` | **NO** | `0.000000` | Padding token masked |
| **9** | `2` | `'<PAD>'` | `-100` | `-100` | `<IGNORE>` | **NO** | `0.000000` | Padding token masked |

---

## 4. Rigorous Comparison: Method A vs. Method B

```mermaid
flowchart TD
    subgraph Method A: Runtime Causal Shift (HF Convention)
        A1["Input: [BOS, P0, P1, P2, P3, P4, R0, R1, EOS] (len 9)"] --> A2["Model Forward -> Logits (len 9)"]
        A2 --> A3["Shift Logits: logits[:, :-1] (len 8)"]
        A4["Labels: [-100, -100, -100, -100, -100, -100, R0, R1, EOS] (len 9)"] --> A5["Shift Labels: labels[:, 1:] (len 8)"]
        A3 --> A6["F.cross_entropy(shift_logits, shift_labels)"]
        A5 --> A6
    end

    subgraph Method B: Pre-Shifted Dataloader (ChakrView Convention)
        B1["Input: [BOS, P0, P1, P2, P3, P4, R0, R1] (len 8)"] --> B2["Model Forward -> Logits (len 8)"]
        B3["Target: [-100, -100, -100, -100, -100, R0, R1, EOS] (len 8)"]
        B2 --> B4["F.cross_entropy(logits, target)"]
        B3 --> B4
    end

    A6 -.->|Bit-Exact Loss & Gradients| EQ["PROVEN MATHEMATICALLY IDENTICAL"]
    B4 -.->|Bit-Exact Loss & Gradients| EQ
```

### Exact Equivalence Properties:
1. In Method A, `shift_labels = labels[:, 1:]`. Therefore:
   - `shift_labels[5] == labels[6] == 3860` (`'Hel'`).
   - `shift_logits[5] == logits[5]` (which processed sequence tokens up through index 5, `'i'`).
2. In Method B, `target_seq[5] == 3860` (`'Hel'`).
   - `logits[5]` processed input tokens up through index 5 (`'i'`).
3. **Conclusion:** Method B's `target_seq` is identically equal to Method A's `shift_labels`. The dataloader performs the shift upfront rather than allocating temporary slices in memory during training.

---

## 5. Multi-Turn Dialogue Verification

In multi-turn conversations, preceding turns function as causal prompt history.
For dialogue:
`User: Name?\nAssistant: Alice\nUser: Repeat\nAssistant: ` $\to$ `Alice`

- **Pre-Shifted Method B:**
  - `input_seq`: `[0] + [User: Name?...Assistant: ] + [Alice]`
  - `target_seq`: `[-100] * len(prompt_tokens) + [Alice] + [1]`
- **Runtime-Shifted Method A:**
  - `full_seq`: `[0] + [User: Name?...Assistant: ] + [Alice] + [1]`
  - `labels`: `[-100] * (1 + len(prompt_tokens)) + [Alice] + [1]`
  - `shift_labels = labels[1:]`
- **Result:** `shift_labels` matches `target_seq` with 100% parity across all token positions. All preceding turns generate zero gradient, and the final response turn generates active gradients.

---

## 6. Test Suite Execution & Results

An isolated test suite (`experiments/ucip14/tests/test_causal_target_alignment_r2.py`) was implemented and executed:
1. `test_01_method_a_standard_runtime_causal_shift`: **PASSED**
2. `test_02_method_b_preshifted_target_construction`: **PASSED**
3. `test_03_exact_mathematical_equivalence`: **PASSED** (loss diff = 0.0, grad diff = 0.0)
4. `test_04_real_model_forward_gradient_isolation`: **PASSED** (on actual `ChakrMicro` forward/backward)
5. `test_05_multi_turn_causal_target_alignment`: **PASSED**

Combined UCIP-14 test suite: `pytest experiments/ucip14/tests/`: **13/13 tests passing**.

---

## 7. Protected Artifact Checksums & Final Verdict

| Artifact | Expected SHA-256 | Actual SHA-256 | Status |
| :--- | :--- | :--- | :---: |
| `chakrview/runtime/session.py` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **MATCH ($\Delta W = 0$)** |
| Canonical Baseline (Seed 42) | `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` | `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` | **MATCH** |
| Base Checkpoint `checkpoint_step_800.pt` | `8759ccb4a70989b79c0b4b26876da1d47989ebc61f39831750b6acf0455b9943` | `8759ccb4a70989b79c0b4b26876da1d47989ebc61f39831750b6acf0455b9943` | **MATCH** |

### Selected Verdict: **VERIFIED**

**Justification:**
The actual production implementation in `chakrview/training/sft_engine.py` is verified to be mathematically and causally sound. It implements a pre-shifted target convention that is 100% bit-exact to standard runtime causal shifting, perfectly isolating loss to assistant responses and EOS while placing zero gradient on prompt tokens and padding. Zero training occurred, and all protected artifacts remain unmodified.
