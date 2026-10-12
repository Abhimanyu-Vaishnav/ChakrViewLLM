# Batch 75 — Repository & Pretraining Pipeline Audit

## 1. Audit Scope & Objective

Before conducting synthetic pretraining and curriculum experiments, Batch 75 independently audits:
1. Current Git working tree, commit history, remotes, and user file preservation.
2. Canonical architectural invariants, expected parameter counts, and baseline checkpoint SHA-256 digests.
3. Test suite execution and pass counts.
4. Pretraining vs SFT pipeline mechanics (causal loss masking, sequence packing, document boundaries, optimizer restoration).
5. Baseline arithmetic performance on the audited single-digit addition benchmark.
6. Execution hardware environment and device selection behavior.

---

## 2. Git & Working Tree Verification

- **Branch**: `dev/v0.6-v0.7-validation`
- **HEAD Commit**: `f5a72b325e933c221e8cbd9e20a7ac05ddd07e0d`
  - Message: `feat(sprint): complete Batches 71-74 reproducibility audit, single-digit arithmetic benchmark, controlled learning evaluation, and rationale hypothesis testing`
  - Committed Files: 17 files (+9901 insertions, -0 deletions).
- **Preserved User Modification**:
  - `chakrview/runtime/session.py` (lines 317–332 converting turn history to `ConversationTemplate` ChatML format).
  - Confirmed present as unstaged modification. Preserved strictly untouched and never staged.
- **Remote Configuration**:
  - `origin`: `https://github.com/Abhimanyu-Vaishnav/ChakrViewLLM.git`
  - No push or merge into `main` has been executed or authorized.

---

## 3. Canonical Architecture & Checkpoint Invariants

All canonical ChakrMicro invariants were independently computed and verified:
- **Total Parameters**: **3,443,136** (exact match).
- **Architecture**: 6 transformer layers, $d_{model}=192$, $n_{heads}=6$, $head\_dim=32$, $hidden\_dim=512$, $vocab\_size=4096$, $max\_seq\_len=512$, RMSNorm, RoPE, SwiGLU, tied embeddings.
- **Step-10,000 Checkpoint**:
  - Path: `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
  - Expected SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
  - Computed SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
  - Verdict: **BIT-EXACT MATCH**.
- **Fresh Initialization SHA-256**:
  - Computed with `torch.manual_seed(42)` and named parameters UTF-8 sorted:
  - Expected: `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`
  - Computed: `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`
  - Verdict: **BIT-EXACT MATCH**.
- **Regression Test Suite**:
  - Collected: **101 tests** across historical batches 33 through 74.
  - Passed: **101 tests (100%)**.
  - Failed: **0 tests**.

---

## 4. Hardware & Resource Environment

- **Execution Hardware**: Intel x86_64 CPU (Windows PowerShell).
- **PyTorch Version**: `2.14.0+cpu`.
- **CUDA / GPU Acceleration**: `torch.cuda.is_available() == False`.
- **System Memory**: Total 31.64 GB RAM, Available ~8.93 GB RAM.
- **Device Policy**: All experiments strictly execute on CPU with bounded thread/memory footprints. No GPU or distributed execution is claimed.

---

## 5. Pretraining vs SFT Causal Loss Mechanics

- **Pretraining Loss**: Next-token prediction across all non-pad tokens ($P(\text{tok}_t \mid \text{tok}_{<t})$) using cross-entropy with `ignore_index=2` (PAD).
- **SFT Loss**: Prompt-masked next-token prediction using `ignore_index=-100`. Only assistant response tokens and EOS contribute to gradients.
- **Prompt/Response Contract**: `chakrview/training/prompt_contract.py` unifies sequence packing into `input_seq = [BOS] + p_toks + r_toks` and `target_seq = [-100]*len(p_toks) + r_toks + [EOS]`, ensuring input at index $|P|$ predicts target $R_0$.
