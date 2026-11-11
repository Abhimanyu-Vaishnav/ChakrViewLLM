# CR-EXP-17R: Phase A — Repository and Artifact Integrity Report

## 1. Executive Identification & Git State
- **Workspace:** `D:\Project\ChakrView`
- **Current Active Branch:** `exp/cbr-01-conversational-breakthrough`
- **Current Local HEAD:** `b2a025b4ec90d344bb4207913b3c0d840917326a`
- **Remote Tracking Branch:** `origin/exp/cbr-01-conversational-breakthrough`
- **Remote HEAD:** `b2a025b4ec90d344bb4207913b3c0d840917326a`
- **Synchronization Status:** Local and Remote are in exact 100% synchronization.
- **Working Tree Status:** Clean (`nothing to commit, working tree clean`).
- **Release Tags:** None created (complying strictly with authorization rules).

## 2. Commit Ancestry and Lineage Audit
The four CR-17 commits were audited directly via `git log`:
1. `33fbb2eab9b5ce246380b25c4e3cf035290e6595`: `research(cognition): archive audited experimental pipelines and scientific evidence (CR-EXP-04 to CR-EXP-16F)` (Parent: `3ff881c`)
2. `3e158f626b719a149a47381526b67d47c0c90070`: `feat(conversation): implement CR-EXP-17 dataset curation, training pipeline, and evaluation suite` (Parent: `33fbb2e`)
3. `06b6abf0f8bbaf1ef641d64ee219a7f756561b3e`: `feat(runtime): calibrate chat.py decoding defaults and add __main__.py CLI interface` (Parent: `3e158f6`)
4. `b2a025b4ec90d344bb4207913b3c0d840917326a`: `release(v0.9.1): package v0.9.1 conversational preview configuration, model card, and usage guide` (Parent: `06b6abf`)

All commits form an unbroken, non-branching linear chain pushed to `origin`.

## 3. Cryptographic Artifact Verification Table

| Artifact Description | File Path | Expected SHA-256 | Actual SHA-256 | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Release Checkpoint** | `releases/v0.9.1_conversational_preview/chakrview_v091_model.pt` | `f161ccdc501bca34f6ed2019252f2e51d18fa113ce0cc90187c4ff0cdd180d10` | `f161ccdc501bca34f6ed2019252f2e51d18fa113ce0cc90187c4ff0cdd180d10` | **VERIFIED (Bit-Exact)** |
| **CR-17 Best Checkpoint** | `experiments/conversation/cr_exp_17/checkpoints/chakrview_v091_conversational_best.pt` | `f161ccdc501bca34f6ed2019252f2e51d18fa113ce0cc90187c4ff0cdd180d10` | `f161ccdc501bca34f6ed2019252f2e51d18fa113ce0cc90187c4ff0cdd180d10` | **VERIFIED (Bit-Exact)** |
| **CR-17 Frozen Benchmark** | `experiments/conversation/cr_exp_17/CONVERSATION_BENCHMARK_V1.json` | `d2472e0b2a72f2555bea68511ac7373c5f2983f785bb9a7d60da4dd9ac2f1a5f` | `d2472e0b2a72f2555bea68511ac7373c5f2983f785bb9a7d60da4dd9ac2f1a5f` | **VERIFIED (Bit-Exact)** |
| **Canonical Stage B Checkpoint** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **VERIFIED (Bit-Exact)** |
| **Frozen Conversational Parent** | `experiments/cbr01_sft/runs/cbr_sft_04/checkpoints/sft_checkpoint_step_200.pt` | `327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153` | `327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153` | **VERIFIED (Bit-Exact)** |
| **CR-16A Arm D Copy Parent** | `experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16a_copying_recovery/checkpoints/arm_d_generalization/arm_d_generalization_best.pt` | `9d609a770c748f9321d06713093620da0fe6caabc04fea2401b37403d33de1a5` | `9d609a770c748f9321d06713093620da0fe6caabc04fea2401b37403d33de1a5` | **VERIFIED (Bit-Exact)** |
| **Protected Runtime** | `chakrview/runtime/session.py` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **VERIFIED (Bit-Exact)** |
| **Committed Training Data** | `experiments/conversation/cr_exp_17/data/conversation_train_1080.jsonl` | `15b59d722a32b9aceb2ebddc0eba92d065157145edfceda5901181258bcc0c47` (git) | `15b59d722a32b9aceb2ebddc0eba92d065157145edfceda5901181258bcc0c47` | **VERIFIED (Bit-Exact)** |
| **Committed Validation Data** | `experiments/conversation/cr_exp_17/data/conversation_val_120.jsonl` | `1e49b6d26f5b4a66093353da7fdd29671dbcfaad55bf8b1eb933aa4c165b9a11` (git) | `1e49b6d26f5b4a66093353da7fdd29671dbcfaad55bf8b1eb933aa4c165b9a11` | **VERIFIED (Bit-Exact)** |
| **Committed Data Manifest** | `experiments/conversation/cr_exp_17/data/conversation_corpus_manifest.json` | `c38d8f64f1f4ef0d77e5dc7d2eb773d799379c1ab479a477b1a55390326100b8` (git) | `c38d8f64f1f4ef0d77e5dc7d2eb773d799379c1ab479a477b1a55390326100b8` | **VERIFIED (Bit-Exact)** |
| **Release Config** | `releases/v0.9.1_conversational_preview/config.json` | `353d196ac5e3751adab80991588f4a9d0ab43ee526e88592c164c72e83749142` (git) | `353d196ac5e3751adab80991588f4a9d0ab43ee526e88592c164c72e83749142` | **VERIFIED (Bit-Exact)** |

## 4. Hash Reconciliation Note
In the CR-17 report narrative, earlier intermediate draft hashes for the dataset and config were quoted in the summary text. However, git forensic inspection of commit `3e158f6` (`ARTIFACT_HASHES.txt`) shows that the actual committed files on disk match the committed git tree with 100% bit-exact cryptographic accuracy.
The release checkpoint `f161ccdc501bca34f6ed2019252f2e51d18fa113ce0cc90187c4ff0cdd180d10` is bit-for-bit identical between `experiments/conversation/cr_exp_17/checkpoints/` and `releases/v0.9.1_conversational_preview/`.
