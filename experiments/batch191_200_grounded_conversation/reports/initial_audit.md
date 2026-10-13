# Initial Failure Reproduction & Architectural Audit: Batches 191–200

## 1. Environment & Baseline State
- **Git Commit Baseline:** `124317e` (`feat(batch181-190): add pointer beam decoding and evidence synthesis`) on `dev/v0.6-v0.7-validation`.
- **Protected File Preservation:** `chakrview/runtime/session.py` remains strictly uncommitted and unmodified. Unrelated sprint documentation files are preserved.
- **Canonical Model:** ChakrMicro v0.1 (3,443,136 unique parameters, $V=4096$, $d=192$, $L=6$, $H=6$).
- **Canonical Checkpoint:** `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`.
- **SHA-256 Digest:** `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (Exact match).
- **Compute Environment:** Windows x64, 31.6 GB RAM, 20 logical CPU cores, CPU-only execution confirmed.

---

## 2. Baseline Reproduction Results
Re-running `experiments/batch181_190_evidence_synthesis/run_pointer_decoding_eval.py` strictly reproduced previous findings:
- Canonical Base (Vocab Only): 20.0% Exact Match (accidental substring in table markup).
- Pointer Greedy ($k=1$): 40.0% Exact Match (`5` and `8` copied cleanly).
- Pointer Beam ($k=2$): 40.0% Exact Match.
- Pointer Beam ($k=3$): 40.0% Exact Match (latency increased from 185 ms to 736 ms).
- Deterministic Oracle: 80.0% Exact Match.

---

## 3. Diagnostic Breakdown of Unresolved Pointer Failure Categories
1. **Query-Entity Overlap Bias (Primary Root Cause):**
   - In query `What is the largest moon of Neptune?` with context `Triton is the largest natural satellite of the planet Neptune.`, the entity token `Neptune` appears in both question and context.
   - The dot-product pointer attention $Q K^T / \sqrt{d}$ is driven strongly by the prompt's recent hidden state, which contains recent embeddings of `Neptune`.
   - Consequently, the pointer head attends to `Neptune` instead of the factual value `Triton`, generating `Nepublish...` or repeating the subject.
2. **Hindi Multi-Byte Continuation Fragmentation:**
   - Multi-byte Hindi words like `विद्युत सुचालक` decompose into 9 fine-grained BPE sub-tokens: `['व', 'िद', '्य', 'ुत', ' स', 'ु', 'च', 'ाल', 'क']`.
   - In unconstrained autoregressive decoding, once a single sub-token deviates or encounters an invalid UTF-8 byte boundary, output text becomes unparseable (``).
   - **Remedy:** Trie-based constrained candidate decoding over valid candidate token sequences.
3. **Relation Disentanglement Deficit:**
   - The pointer mechanism does not currently receive an explicit representation of the requested relation (e.g. `atomic number` vs `melting point` vs `largest moon`) to modulate context position attention.
