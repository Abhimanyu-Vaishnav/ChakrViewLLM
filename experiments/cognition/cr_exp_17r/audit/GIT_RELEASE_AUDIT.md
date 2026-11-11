# CR-EXP-17R: Phase L — Git & Release Integrity Report

## 1. Git Repository State Audit
- **Repository Path:** `D:\Project\ChakrView`
- **Active Branch:** `exp/cbr-01-conversational-breakthrough`
- **Local HEAD Commit:** `b2a025b4ec90d344bb4207913b3c0d840917326a`
- **Remote Tracking:** `origin/exp/cbr-01-conversational-breakthrough`
- **Remote HEAD Commit:** `b2a025b4ec90d344bb4207913b3c0d840917326a`
- **Synchronization:** 100% in sync (`ahead 0, behind 0`).
- **Release Tags:** Zero tags created (strict compliance with authorization).

## 2. Verification of the 4 CR-17 Commits
The commit ancestry was audited and verified to form a linear, non-destructive sequence:
1. `33fbb2e`: `research(cognition): archive audited experimental pipelines and scientific evidence (CR-EXP-04 to CR-EXP-16F)`
   - Files: 545 files changed, 2,135,803 insertions.
   - Preserves all preceding cognitive experiments.
2. `3e158f6`: `feat(conversation): implement CR-EXP-17 dataset curation, training pipeline, and evaluation suite`
   - Files: 27 files changed, 28,748 insertions.
   - Includes benchmark, dataset, training harness, and evaluation logs.
3. `06b6abf`: `feat(runtime): calibrate chat.py decoding defaults and add __main__.py CLI interface`
   - Files: 2 files changed, 83 insertions.
   - Adds CLI entrypoint and calibrated sampling defaults.
4. `b2a025b`: `release(v0.9.1): package v0.9.1 conversational preview configuration, model card, and usage guide`
   - Files: 3 files changed, 130 insertions.
   - Adds model card, run instructions, and json config.

## 3. Protected Files Audit
The protected runtime file `chakrview/runtime/session.py` was inspected and verified to be untouched:
- Expected SHA-256: `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67`
- Actual SHA-256:   `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67`
- Status: **VERIFIED (Bit-Exact)**

## 4. Release Files Tracking Status
- Model binary `chakrview_v091_model.pt` is intentionally ignored by `.gitignore` (standard repository policy for model weights).
- Model configuration `config.json`, `MODEL_CARD.md`, and `HOW_TO_RUN.md` are tracked by Git.
- Model binary SHA-256 is cryptographically anchored in `config.json`, `MODEL_CARD.md`, and `ARTIFACT_HASHES.txt`.
