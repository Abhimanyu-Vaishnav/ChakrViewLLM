# UCIP-14C: Design Consistency & Assistant-Only Loss Verification Report
## Corrective Engineering Audit on Instruction Alignment Architecture

**Mission ID:** UCIP-14C  
**Project:** ChakrView  
**Repository:** `D:\Project\ChakrView`  
**Active Branch:** `exp/cbr-01-conversational-breakthrough`  
**Starting Commit:** `d564bb6` (verified)  
**Execution Mode:** Corrective Engineering Audit  
**Training Status:** **ZERO TRAINING CONDUCTED (STRICTLY PROHIBITED)**  
**Final Scientific Verdict:** **VERIFIED DESIGN**  

---

## 1. Executive Summary

In accordance with the UCIP-14C charter, an independent corrective engineering audit was conducted on the UCIP-14 instruction alignment architecture deliverables. All implementation-level ambiguities, token-level causal alignments, gradient flows, dataset allocations, replay ratio mathematics, benchmark quarantine procedures, and execution security constraints were rigorously verified and resolved.

### Core Audit Findings:
1. **Assistant-Only Masked Loss Formally Verified:** Evaluated token-by-token alignment on `User: Hi` $\to$ `Hello`. Verified that BOS is at index 0, prompt tokens predict target `-100`, the final prompt token predicts the first response token, all response tokens predict subsequent response tokens, the final response token predicts `<EOS>=1`, and padding tokens predict `-100`. Gradient backpropagation confirmed that **prompt and padding positions receive exactly 0.0 logit gradient norm**, while all response and EOS positions receive active gradients ($>0.66$).
2. **Replay Ratio Consistency Resolved:** The nominal "80% SFT / 20% Replay" target and the concrete "26 SFT + 6 Replay per 32 batch" implementation were mathematically reconciled: $26/32 = 81.25\%$ SFT and $6/32 = 18.75\%$ Replay. A microbatch structure of $13 \text{ SFT} + 3 \text{ Replay}$ with gradient accumulation of 2 achieves this integer composition cleanly, eliminating fractional batch slicing.
3. **Exact 15,000 Dataset Record Allocation:** Exact integer target, minimum, and maximum counts were defined across all 18 categories, summing to **exactly 15,000 records** (Train: 12,000, Val: 1,500, Dev: 1,500). Non-overlapping language allocations were established: English (8,250), Hindi Devanagari (4,500), Hinglish (1,500), Sanskrit (750).
4. **Benchmark Quarantine & Historical Prompt Exclusion:** Established the quarantine protocol for the 500-probe sealed evaluation set. Historical UCIP prompts are strictly barred from being recycled. Zero contamination is enforced via automated n-gram overlap checks and cryptographic manifest registry.
5. **Python Execution Security Controls Defined:** Audited the existing evaluation harness and documented the crucial distinction between subprocess execution and OS security containment. Established mandatory AST pre-screening, module whitelisting, timeout enforcement (3.0s), memory capping, and isolated ephemeral workspaces.

---

## 2. Phase 1: Assistant-Only Masked Loss Verification

### A. Deterministic Token-by-Token Sequence Trace
For the test pair:
- **Prompt:** `User: Hi`
- **Response:** `Hello`

Under ChakrMicro's frozen BPE tokenizer (`data/experiments/vocab_4096`), prompt tokens are `[88, 835, 324, 75, 108]` (`['U', 'ser', ': ', 'H', 'i']`), and response tokens are `[3860, 317]` (`['Hel', 'lo']`).

Sequence construction via `build_causal_sequence`:
- `input_seq`: `[0, 88, 835, 324, 75, 108, 3860, 317, 2, 2]`
- `target_seq`: `[-100, -100, -100, -100, -100, 3860, 317, 1, -100, -100]`

| Index | Input Token ID | Decoded Input | Target Token ID | Decoded Target | Active Loss? | Logit Gradient Norm |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | `0` | `<BOS>` | `-100` | `<IGNORE>` | **NO** | `0.000000` |
| **1** | `88` | `'U'` | `-100` | `<IGNORE>` | **NO** | `0.000000` |
| **2** | `835` | `'ser'` | `-100` | `<IGNORE>` | **NO** | `0.000000` |
| **3** | `324` | `': '` | `-100` | `<IGNORE>` | **NO** | `0.000000` |
| **4** | `75` | `'H'` | `-100` | `<IGNORE>` | **NO** | `0.000000` |
| **5** | `108` | `'i'` | `3860` | `'Hel'` | **YES** | `0.666489` |
| **6** | `3860` | `'Hel'` | `317` | `'lo'` | **YES** | `0.666571` |
| **7** | `317` | `'lo'` | `1` | `<EOS>` | **YES** | `0.666498` |
| **8** | `2` | `<PAD>` | `-100` | `<IGNORE>` | **NO** | `0.000000` |
| **9** | `2` | `<PAD>` | `-100` | `<IGNORE>` | **NO** | `0.000000` |

### B. Mathematical Causal Alignment Proof:
1. `input_seq[5]` is token `'i'` (final prompt token). It attends to all preceding prompt tokens (`<BOS> User: Hi`) and its causal prediction target is `target_seq[5] == 3860` (`'Hel'`), the first token of the assistant response.
2. `input_seq[6]` is token `'Hel'`. It attends to the prompt and itself, predicting `target_seq[6] == 317` (`'lo'`).
3. `input_seq[7]` is token `'lo'`. It attends to all preceding tokens, predicting `target_seq[7] == 1` (`<EOS>`).
4. **Zero Off-by-One Shift:** The causal shift is mathematically exact. The prompt receives zero gradient; the assistant response and terminal EOS receive full gradient signal.
5. **Multi-Turn Compatibility:** In multi-turn sequences (e.g. `User: ... \n Assistant: ... \n User: ... \n Assistant: `), all preceding turns receive target `-100`, isolating loss to the final assistant response.

---

## 3. Phase 2: Replay Ratio Consistency Resolution

### The Discrepancy:
- Documentation cited nominal "80% SFT / 20% Replay".
- Batch layout specified "26 SFT sequences + 6 Stage C replay sequences per 32-sequence batch".
- Exact ratio: $26 / 32 = 0.8125$ ($81.25\%$) and $6 / 32 = 0.1875$ ($18.75\%$).

### Analysis of Scheduling Strategies:
1. **Fixed Integer Batch Composition (Recommended):**
   - Each effective batch of 32 consists of 26 SFT sequences and 6 Replay sequences.
   - Sliced across 2 microbatches of 16 (gradient accumulation = 2): each microbatch draws **13 SFT sequences and 3 Replay sequences** ($13 \times 2 = 26$, $3 \times 2 = 6$).
   - **Advantage:** Every single optimizer step contains an identical, smooth regularization signal. Prevents gradient oscillation.
2. **Alternating Batch Scheduling (4:1):**
   - 4 batches of pure SFT (32 seqs each = 128 seqs) followed by 1 batch of pure Stage C (32 seqs).
   - Exactly $80.0\%$ SFT and $20.0\%$ Replay.
   - **Disadvantage:** Introduces gradient shock on step 5, as loss and gradient norms differ sharply between assistant-only masked loss and full pretraining loss.
3. **Separately Weighted Loss Aggregation:**
   - $\mathcal{L} = \alpha \mathcal{L}_{\text{SFT}} + \beta \mathcal{L}_{\text{Replay}}$.
   - **Disadvantage:** Requires tuning an arbitrary scalar hyperparameter ($\alpha, \beta$) rather than letting token frequencies reflect natural data mixtures.

### Resolution:
Adopt **Fixed Integer Batch Composition** (13 SFT + 3 Replay per microbatch; 26 SFT + 6 Replay per effective batch of 32). The nominal label is formally clarified as **81.25% SFT / 18.75% Foundational Replay**.

---

## 4. Phase 3: Exact 15,000 Dataset Record Specification

The 15,000 target count refers strictly to **unique supervised training examples** (one prompt $\to$ one gold assistant response turn).

### Complete Allocation Table (Sum = Exactly 15,000):

| Category ID | Category Name | Target % | Target Records | Min Records | Max Records |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `CONV` | Conversation & Greetings | 6.0% | **900** | 720 | 1,125 |
| `INST` | Instruction Following | 8.0% | **1,200** | 960 | 1,500 |
| `QA` | Question Answering | 7.0% | **1,050** | 840 | 1,312 |
| `CTX` | Context Retention (Multi-turn) | 6.0% | **900** | 720 | 1,125 |
| `COPY` | Exact Copying | 4.0% | **600** | 480 | 750 |
| `BIND` | Entity Binding | 5.0% | **750** | 600 | 938 |
| `ARITH` | Arithmetic & Numerical Reasoning | 6.0% | **900** | 720 | 1,125 |
| `LOGIC` | Deductive Logic & Syllogisms | 5.0% | **750** | 600 | 938 |
| `EXPL` | Conceptual Explanations | 7.0% | **1,050** | 840 | 1,312 |
| `PYTHON` | Python Syntax & Coding | 8.0% | **1,200** | 960 | 1,500 |
| `ENG` | English Linguistic | 6.0% | **900** | 720 | 1,125 |
| `HIN` | Hindi (Devanagari) | 8.0% | **1,200** | 960 | 1,500 |
| `HING` | Hinglish | 4.0% | **600** | 480 | 750 |
| `SAN` | Sanskrit | 3.0% | **450** | 360 | 562 |
| `UNCERT` | Uncertainty & Abstention | 5.0% | **750** | 600 | 938 |
| `FORMAT` | Formatting Constraints | 4.0% | **600** | 480 | 750 |
| `CORR` | Error Correction | 4.0% | **600** | 480 | 750 |
| `CLARIF` | Clarification Request | 4.0% | **600** | 480 | 750 |
| **TOTAL** | **18 Categories** | **100.0%** | **15,000** | **12,000** | **18,747** |

### Orthogonal Language Allocation:
- **English (`en`):** 8,250 records (55.0%)
- **Hindi Devanagari (`hi`):** 4,500 records (30.0%)
- **Hinglish (`hi-en`):** 1,500 records (10.0%)
- **Sanskrit (`sa`):** 750 records (5.0%)
- **Total:** Exactly 15,000 records (100.0%).

### Partitioning Splits:
- **Training Set (80%):** 12,000 records.
- **Validation Set (10%):** 1,500 records.
- **Dev Evaluation Set (10%):** 1,500 records.
- **Sealed Final Evaluation Set:** 500 quarantined probes (completely separate).

---

## 5. Phase 4: Benchmark Quarantine Protocol

1. **Independent Probe Authoring:** All 500 sealed evaluation probes must be authored independently. **Recycling historical prompts from UCIP-10E, UCIP-10E-R2, or UCIP-11 is strictly prohibited.**
2. **Quarantine Directory:** Stored strictly in `experiments/ucip14/benchmarks/sealed/benchmark_ucip14_sealed.json`.
3. **Automated Leakage Audit:** Prior to training, an n-gram overlap audit script checks all training records against the sealed suite. Any matching record is dropped.
4. **Dataloader Hard Lock:** The training dataloader raises an immediate `PermissionError` if any path matching `*sealed*` is supplied.

---

## 6. Phase 5: Python Evaluation Security & Execution Containment

Inspection of `experiments/ucip10e_r2/evaluators/strict_evaluators_r2.py` confirmed that code execution is currently handled via raw `subprocess.run([sys.executable, tmp_path], timeout=5)`.

### Security Vulnerabilities of Raw Subprocess Execution:
- Subprocesses inherit host filesystem read/write access.
- Subprocesses inherit local networking capabilities.
- Untrusted code generated by the model could execute deletion commands or bind ports.

### Mandatory Future Security Controls:
1. **AST Pre-Execution Screening:** Parse code with `ast.parse`. Reject any code importing `os`, `sys`, `subprocess`, `shutil`, `socket`, `requests`, `urllib`, `ctypes`, or accessing `__import__`.
2. **Process Isolation Flags:** Execute Python with isolated mode: `python.exe -I -B -s temp_code.py`.
3. **Execution Limits:** Strict 3.0-second timeout ceiling; 256 MB memory cap.
4. **Ephemeral Sandboxing:** Execute in dedicated `tempfile.TemporaryDirectory` with immediate cleanup in a `finally:` block.

---

## 7. Deliverables & Documentation Updated

- Updated: [UCIP_14_DATASET_SPECIFICATION.md](file:///d:/Project/ChakrView/docs/UCIP_14_DATASET_SPECIFICATION.md) (15,000 exact table, language matrix, record definitions).
- Updated: [UCIP_14_TRAINING_STRATEGY.md](file:///d:/Project/ChakrView/docs/UCIP_14_TRAINING_STRATEGY.md) (Exact 26/6 replay math, token weighting).
- Updated: [UCIP_14_EVALUATION_PROTOCOL.md](file:///d:/Project/ChakrView/docs/UCIP_14_EVALUATION_PROTOCOL.md) (Historical benchmark ban, Python containment controls).
- Created: [test_assistant_only_loss.py](file:///d:/Project/ChakrView/experiments/ucip14/tests/test_assistant_only_loss.py) (**3/3 passing token-level and gradient tests**).
- Created: [UCIP_14C_DESIGN_CONSISTENCY_AUDIT.md](file:///d:/Project/ChakrView/reports/UCIP_14C_DESIGN_CONSISTENCY_AUDIT.md).

---

## 8. Final Verdict & Authorization Gate

- **Protected Hashes Verified:**
  - `chakrview/runtime/session.py` SHA256: `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` (**MATCH, $\Delta W = 0$**).
  - Base checkpoint candidate (`checkpoint_step_800.pt`) SHA256: `8759ccb4a70989b79c0b4b26876da1d47989ebc61f39831750b6acf0455b9943` (**MATCH**).
- **Training Statement:** **ZERO training, fine-tuning, optimizer updates, or checkpoint modifications occurred.**
- **Verdict:** **VERIFIED DESIGN**
- **Readiness:** The design specifications and implementation mechanics are fully reconciled, verified, and ready for **UCIP-14R (Dataset Construction & Independent Quality Audit)**.
