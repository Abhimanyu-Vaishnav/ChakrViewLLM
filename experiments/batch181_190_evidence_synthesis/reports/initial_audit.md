# Initial Audit Report: Batches 181–190
## Baseline Reproduction, Failure Diagnostics & Experimental Invariants

### 1. Repository State & Canonical Invariants
- **Git Commit Baseline:** `77a9a91` (`feat(batch171-180): add experimental neural pointer learning and cognitive execution`) on `dev/v0.6-v0.7-validation`.
- **Protected File Preservation:** `chakrview/runtime/session.py` remains strictly uncommitted and unaltered.
- **Canonical Model:** ChakrMicro v0.1 (3,443,136 unique parameters, $V=4096$, $d=192$, $L=6$, $H=6$).
- **Canonical Checkpoint:** `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`.
- **SHA-256 Digest:** `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (Exact match).
- **Compute Environment:** Windows x64, 31.6 GB RAM, 20 logical CPU threads, CUDA unavailable (CPU-only execution verified).

### 2. Reproduction of Previous Results
- Canonical Base Checkpoint (Arm A): Evaluated zero-shot on 5 held-out test facts. Produced 20% Exact Match due to an accidental substring hit (`Boron` -> `5` inside table formatting string `|2005|`).
- SFT Next-Token Cross-Entropy (Arm B): 0% Exact Match; collapsed into high-frequency token repetition.
- Auxiliary ChakrPointerHead (Arm C): 40% Exact Match (correctly copied `5` for Boron and `8` for Oxygen from context).
- Checkpoint Lineage: `experiments/batch171_180_pointer_learning/checkpoints/pointer_head_arm_c.pt` (24,961 trainable parameters).

### 3. Deep Diagnosis of Remaining 60% Pointer Failures
1. **Multi-Token Name Truncation:**
   - Case `Neptune -> Triton`: Generated `Nepublic args Nepton\nNepu`. The greedy pointer step selected a capitalized token derived from the query rather than the relation target in the context.
2. **Numeric Prefix Overlap:**
   - Case `Gold -> 1064 C`: Generated `64 ChmeCl6l6l6\n64 C`. Copied sub-token `64` instead of full sequence `1064 C` because greedy search failed to maintain sequence coherence across multiple BPE tokens.
3. **Hindi Multibyte Tokenization:**
   - Case `चांदी -> विद्युत सुचालक`: Fallback to table template tokens due to BPE unicode byte fragment boundaries in greedy decoding.
4. **Greedy Horizon Myopia:**
   - In greedy decoding ($k=1$), a single sub-optimal first token prevents the pointer mechanism from completing multi-token entities.
   - **Remedy:** Beam search decoding ($k=2..4$) with causal pointer constraints and sequence-level length normalization.
