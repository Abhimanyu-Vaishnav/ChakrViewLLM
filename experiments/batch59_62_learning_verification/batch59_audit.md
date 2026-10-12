# Batch 59 Audit: Baseline & Training Route Independent Verification

## 1. Git & Repository Status
- **Current Branch**: `dev/v0.6-v0.7-validation`
- **Initial HEAD**: `e59d349a15c898c8d8f0709b1fbf788647efb99e`
- **Working Tree**: Preserved pre-existing uncommitted modification in `chakrview/runtime/session.py`.
- **Commit History**:
  - `e59d349` — `feat(sprint): complete Batches 55-58 enforced trust gates, verified micro-benchmark, and generation diagnostics`
  - `d624bee` — `feat(sprint): complete Batches 51-54 causal alignment audit, dataset trust gates, and clean learning pilot`
  - `01f0c52` — `docs(batch47-50): add final report and manifest for Batches 47-50`

## 2. Canonical Baseline Invariant Verification
All canonical invariants verified bit-exact:
- **Total Parameters**: `3,443,136`
- **Vocabulary Size**: `4,096`
- **Configuration**: 6 layers, `d_model=192`, `n_heads=6`, `hidden_dim=512`, `max_seq_len=512`
- **Fresh Initialization SHA-256**: `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`
- **Canonical Step 10,000 Checkpoint SHA-256**: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`

## 3. End-to-End Training Route & Trust Gate Audit
We audited the path from dataset JSONL file to model weight updates:

### Identified Gap in Batch 55-58 Code
- `SFTDataset._load` previously checked `rec.get("review_status")`. If a JSONL file omitted `review_status` entirely, it was treated as `None` and bypassed the `PENDING_HUMAN_REVIEW` check.
- `train_clean.jsonl` from Batch 51-54 did not have explicit `review_status` fields in its lines, even though the manifest recorded them as `PENDING_HUMAN_REVIEW`.

### Resolution & Hardening
1. In `chakrview/training/sft_engine.py`:
   - `SFTDataset` now computes and stores `self.file_sha256`.
   - `SFTDataset` blocks any record tagged with `data_role: HELD_OUT_BENCHMARK` or `BENCHMARK` from training (`ValueError`).
   - `save_checkpoint` records `dataset_mode`, `allow_exploratory_opt_in`, `enforce_trust_gate`, `train_file`, `val_file`, `train_file_sha256`, and `val_file_sha256` in the saved checkpoint payload.
2. Verified that all 77 regression tests pass.
