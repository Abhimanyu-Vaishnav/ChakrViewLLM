# CBR-SFT-04 PHASE 1.5 — INDEPENDENT PRE-TRAINING AUDIT & EXECUTION READINESS REPORT

**Audit Date:** 2026-11-04  
**Audit Classification:** Independent Pre-Training Verification Audit (Read-Only)  
**Parent Mission:** CBR-01 / CBR-SFT-04  
**Repository:** `D:\Project\ChakrView`  
**Active Branch:** `exp/cbr-01-conversational-breakthrough` (`257801b4c0fcc3017bb450b104baee5971ad530a`)  
**Parent Target Checkpoint:** CBR-SFT-03B Step 200 (`297dcc60f39cadd6bae9a5bd969b1a7d831845d7a67823b30c194a67ef75b5ac`)  

---

## 1. Executive Summary & Gate Verdict

An exhaustive, line-by-line independent pre-training audit of the **CBR-SFT-04** dataset (160 examples), splits (144 train / 16 validation), and independent held-out evaluation suite (80 tasks) was conducted. The audit verified dataset structural validity, exact entity grounding accuracy, conversational multi-turn integrity, zero prompt leakage, tokenizer provenance, architecture context window specifications, and parent checkpoint cryptographic lineage.

### Gate Verdict:
**`READY FOR PHASE 2`**

No blockers, data corruption, or leakage were identified. The dataset and evaluation suite adhere to the canonical architecture constraints of ChakrMicro and are fully validated for controlled SFT execution.

---

## 2. Invariant & Safety Verification

All project safety rules and non-destructive boundaries were preserved:
- **Git Branch & HEAD:** `exp/cbr-01-conversational-breakthrough` at commit `257801b4c0fcc3017bb450b104baee5971ad530a`.
- **Zero Destructive Actions:** No `git reset`, `git clean`, or history modifications were performed.
- **Protected Files Integrity (100% Bit-Exact SHA-256 Match):**
  - `chakrview/brain/model.py`: `1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361`
  - `chakrview/runtime/session.py`: `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67`
  - `Canonical Step 10k Checkpoint`: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
  - `Parent CBR-SFT-02 Step 50`: `592553c3e27b654997e8f5df61e94fe1de25bbf577391df6c1a87f974730bf9d`
  - `Parent CBR-SFT-03B Step 200`: `297dcc60f39cadd6bae9a5bd969b1a7d831845d7a67823b30c194a67ef75b5ac`
  - `Alternate CBR-SFT-03B Step 300`: `e540bd2364113d7eaf271c3495d882921f4421e59659f19f06cd7d590f74710a`
- **Isolation Boundaries:** `experiments/overnight_runs/` was not accessed, enumerated, or modified. `D:\AI\SmartProjectMemory` was completely untouched.
- **Zero Model Modification:** Zero training steps, weight updates, or checkpoint writes were executed.

---

## 3. Dataset Quality Findings (All 160 Examples Inspected)

A comprehensive inspection of all 160 examples across JSONL formatting, role structuring, naturalness, factual claims, entity extraction, and abstention was performed:

### 3.1 JSONL & Field Schema
- All 160 examples parse cleanly as valid single-line JSON.
- Every record includes mandatory fields: `id`, `track`, `category`, `prompt`, `response`.
- Prompts strictly adhere to the prompt framing contract: starting with `User: ` and terminating with `\nAssistant:`.
- Responses strictly preserve SentencePiece subword leading spaces (starting with ` `) to ensure seamless token concatenation with no spacing artifacts.

### 3.2 Track Breakdown & Temporal Balance
- **Track A (Temporal Greetings - 40 examples):**
  - Morning (`CBR4-GREET-MORN-0001` to `0010`): Exactly 10 examples.
  - Afternoon (`CBR4-GREET-AFTN-0001` to `0010`): Exactly 10 examples.
  - Evening (`CBR4-GREET-EVEN-0001` to `0010`): Exactly 10 examples.
  - Generic / Timeless (`CBR4-GREET-GEN-0001` to `0010`): Exactly 10 examples.
  - **Audit Finding:** Temporal bias from prior runs (which over-indexed on morning greetings) is completely resolved by an exact 1:1:1:1 balance.
- **Track B (Identity, Role, Limits & Constraints - 30 examples):**
  - Examples (`CBR4-IDENT-0001` to `0030`) accurately state ChakrView's indigenous development, 3.44M parameter size, custom 4,096 BPE tokenizer, and native commodity CPU execution.
  - Explicitly refutes third-party foundation ties (negative constraints against OpenAI ChatGPT, Anthropic Claude, Meta LLaMA, Mistral).
  - Explicitly states role boundaries (user vs. assistant) and epistemic limits (acknowledging uncertainty rather than hallucinating).
- **Track C (Entity Grounding, Copying & Abstention - 50 examples):**
  - Exact entity copying (`CBR4-GROUND-0001` to `0040`): 40 technical and factual entity extraction queries (IPs, port numbers, commit hashes, file paths, personal names, and metrics).
  - **Verbatim Verification:** Every non-abstention entity target was verified programmatically against its source context. In 100% of cases, the target value is present verbatim in the provided context text.
  - Explicit abstentions (`CBR4-GROUND-0041` to `0050`): 10 examples where the prompt asks for information omitted from the context (e.g., passwords, birthdates, unstated compiler flags). The target responses correctly express bounded abstention ("That information is not provided in the context.").
- **Track D (Multi-Turn Dialogue Pairs - 40 examples):**
  - Evaluated in Section 4 below.

---

## 4. Multi-Turn Integrity Findings (All 40 Pairs Inspected)

All 40 dialogue pairs (`CBR4-MTURN-0001` to `0040`) were evaluated against multi-turn conversational criteria:

1. **Dependency on History:**
   - In all 40 cases, Turn 2 directly refers to context introduced in Turn 1 (e.g., referencing tracking codes, project names, numerical variables, user preferences, or error codes).
   - Turn 2 cannot be resolved without access to Turn 1.
2. **Context Serialization:**
   - Serialized as: `User: {u1}\nAssistant: {a1}\nUser: {u2}\nAssistant:` with target completion ` {a2}`.
   - Preserves complete conversational history in the prompt prefix, training the causal attention mask to attend across previous dialogue turns.
3. **Partition Preservation:**
   - In the train/val split (36 train, 4 val), multi-turn pairs were split as atomic units. No dialogue pair was fragmented across splits.

---

## 5. Data Leakage & Evaluation Independence Findings

An independent cross-partition audit was conducted across Training (144), Validation (16), and Held-Out Evaluation (80):
- **Exact Prompt Collisions:**
  - Train $\cap$ Validation: **0**
  - Train $\cap$ Evaluation: **0**
  - Validation $\cap$ Evaluation: **0**
- **Normalized / Near-Duplicate Collisions:**
  - Case-insensitive, punctuation-stripped, and whitespace-normalized prompt hashes yielded **0 collisions** between Train/Val and Evaluation.
- **Evaluation Independence:**
  - The 80 held-out evaluation tasks test identical structural capabilities (Track A temporal precision, Track B identity boundaries, Track C entity copying, Track D conversational recall) using completely disjoint phrasings, independent entities (different IP blocks, filenames, project names, and numbers), and novel contexts.
- **Evaluator Robustness:**
  - `DualScorer` in `evaluation/cbr_sft_04_suite/evaluator.py` executes both legacy regex and semantic evaluation.
  - Successfully resolves legacy regex false rejections on punctuation-delimited entities (e.g. `10.0.4.12.`) and semantic abstention paraphrases while strictly enforcing mutual exclusivity on temporal greetings.

---

## 6. Architecture, Tokenizer & Context Window Investigation

### 6.1 Context Window Resolution: 256 vs. 512
- **Investigation:** The Phase 1 summary referenced a 256-token context window, whereas the canonical architecture specification defines `max_seq_len: 512`.
- **Finding:**
  - Inspection of `chakrview/brain/config.py` confirms:
    ```python
    max_seq_len: int = 512
    ```
  - Inspection of the tokenized CBR-SFT-04 dataset reveals:
    - Min sequence length: **32 tokens**
    - Average sequence length: **77.3 tokens**
    - Max sequence length: **154 tokens**
  - **Conclusion:** All CBR-SFT-04 sequences (maximum 154 tokens) fit comfortably within both 256 and the full canonical 512 context window. The 256 reference in Phase 1 notes was an empirical upper bound filter from historical Exp34 runs, but the canonical architecture and runtime natively support the full 512 tokens. No configuration changes are required or permitted.

### 6.2 Tokenizer Provenance
- Tokenizer path: `data/experiments/vocab_4096`
- Format: `chakrview_bpe` version `0.1.0`
- Vocab size: Exactly **4,096**
- Special tokens: `<BOS>: 0`, `<EOS>: 1`, `<PAD>: 2`
- Checksums verified:
  - `merges.json` SHA-256: `7498d92adeef7c9e64d4d1bbd5b4b23f8774692d868742e3b98e30c584c4403f`
  - `vocab.json` SHA-256: `e0f11c0795fa0ddd366f54a17355e5ceaaf7d12fc73881677dc122eaae6a999e`
- Confirmed: This is the official canonical ChakrMicro 4096-vocabulary tokenizer, not an experimental replacement.

---

## 7. Parent Checkpoint Lineage & Training Readiness

- **Designated Parent Checkpoint:** CBR-SFT-03B Step 200
- **Location:** `experiments/cbr01_sft/runs/cbr_sft_03b/sft_checkpoint_step_200.pt`
- **Verified SHA-256:** `297dcc60f39cadd6bae9a5bd969b1a7d831845d7a67823b30c194a67ef75b5ac`
- **Lineage Chain:**
  1. Canonical Step 10,000 (`96357705...`)
  2. CBR-SFT-01 Step 100
  3. CBR-SFT-02 Step 50 (`592553c3...`)
  4. CBR-SFT-03B Step 200 (`297dcc60...`) $\longrightarrow$ **Target Base for CBR-SFT-04**

### Proposed Training Hyperparameters for Phase 2:
- **Optimization:** AdamW ($\beta_1=0.9, \beta_2=0.95$), weight decay 0.01.
- **Learning Rate Schedule:** Peak $1.5 \times 10^{-4}$ with 20-step linear warmup from $5.0 \times 10^{-5}$ and cosine decay to $1.0 \times 10^{-5}$.
- **Batching:** Micro-batch size 4, gradient accumulation steps 2 (effective batch size 8).
- **Target Steps:** 200–300 steps with checkpointing every 25 steps.
- **Output Directory:** `experiments/cbr01_sft/runs/cbr_sft_04/` (isolated from historical runs).
- **Accidental Overwrite Prevention:** The training script must write exclusively to the new `cbr_sft_04` directory, with assertions validating that neither base canonical nor parent checkpoint files can be opened in write mode.

---

## 8. Exact File Hashes Audited

| Artifact | File Path | SHA-256 Hash | Status |
| :--- | :--- | :--- | :---: |
| **Model Code** | `chakrview/brain/model.py` | `1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361` | MATCH |
| **Runtime Code** | `chakrview/runtime/session.py` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | MATCH |
| **Canonical Checkpoint** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | MATCH |
| **CBR-SFT-02 Step 50** | `experiments/cbr01_sft/runs/cbr_sft_02/sft_checkpoint_step_50.pt` | `592553c3e27b654997e8f5df61e94fe1de25bbf577391df6c1a87f974730bf9d` | MATCH |
| **CBR-SFT-03B Step 200** | `experiments/cbr01_sft/runs/cbr_sft_03b/sft_checkpoint_step_200.pt` | `297dcc60f39cadd6bae9a5bd969b1a7d831845d7a67823b30c194a67ef75b5ac` | MATCH |
| **CBR-SFT-03B Step 300** | `experiments/cbr01_sft/runs/cbr_sft_03b/sft_checkpoint_step_300.pt` | `e540bd2364113d7eaf271c3495d882921f4421e59659f19f06cd7d590f74710a` | MATCH |
| **Full SFT Dataset** | `data/sft/cbr_sft_04/dataset.jsonl` | `dbea739883aa2459e63976c1b0cc948d34473c1fb987aa2b954bee884a395cbe` | MATCH |
| **Train Split (144)** | `data/sft/cbr_sft_04/splits/cbr_sft_04_train_144.jsonl` | `335ea5d9456d455119019c91dda900b4f8dc06e7195577419dc8c047cf4baaf1` | MATCH |
| **Val Split (16)** | `data/sft/cbr_sft_04/splits/cbr_sft_04_val_16.jsonl` | `331abe2077e6478d25215e6b9b49d002fa9af0068fa1866f960494c9f653f1d9` | MATCH |
| **Held-Out Eval (JSON)** | `evaluation/cbr_sft_04_suite/held_out_cbr_sft_04_80.json` | `0f20ce4cb1a21e6c382346ef27137f3747bf62c1240c49cc3a32f643eec4b036` | MATCH |
| **Held-Out Eval (JSONL)** | `evaluation/cbr_sft_04_suite/evaluation_cases.jsonl` | `382f95e1498672b57929540d0e6860550174ee641c229b6f28b0aa34740d7a69` | MATCH |
| **Evaluator Engine** | `evaluation/cbr_sft_04_suite/evaluator.py` | `82f9cceba68e21ecf425b0fbf8d79a2958416ca255476a6cf06a86c8f62c6ea0` | MATCH |

---

## 9. Comprehensive Classification of Audit Requirements

| Audit Requirement | Classification | Evidence & Notes |
| :--- | :---: | :--- |
| **Preserve all invariants & safety rules** | **PASS** | No destructive Git commands, hashes verified bit-exact, sensitive directories unaccessed. |
| **JSONL structure & field validation** | **PASS** | 160/160 records valid; correct user/assistant fields; leading space preserved. |
| **Track A temporal balance** | **PASS** | Exactly 10 morning, 10 afternoon, 10 evening, 10 generic. Zero skew. |
| **Track B identity & constraint integrity** | **PASS** | Indigenous CPU-first architecture verified; negative constraints against third-party models validated. |
| **Track C entity extraction & abstention** | **PASS** | 40/40 non-abstention values verified verbatim in context; 10/10 abstentions justified. |
| **Track D multi-turn context retention** | **PASS** | 40/40 dialogue pairs genuinely dependent on Turn 1; serialized with full conversational framing. |
| **Data leakage & evaluation independence** | **PASS** | 0 exact collisions, 0 normalized collisions across train, val, and held-out 80 tasks. |
| **Dual-scorer validation** | **PASS** | Resolves punctuation and semantic abstention issues; enforces temporal exclusivity. |
| **Architecture & tokenizer provenance** | **PASS** | Canonical `max_seq_len: 512` verified; 4,096 BPE tokenizer integrity confirmed. |
| **Parent checkpoint lineage** | **PASS** | CBR-SFT-03B Step 200 hash `297dcc60...` verified bit-exact. |
| **Training configuration safety** | **PASS** | Memory-safe micro-batching, isolated destination directory, protected file write guards. |

---

## 10. Final Gate Determination

### Verdict: **`READY FOR PHASE 2`**

The CBR-SFT-04 dataset, validation partition, independent evaluation suite, and execution environment are audited, verified, and free of defects. Phase 1.5 is concluded.

*Awaiting explicit human authorization to initiate Phase 2 training.*

