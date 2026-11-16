# CR-EXP-19: Training Precheck Report

## 1. Mandatory Preconditions Checklist
- [x] **Runtime History Tests:** 8/8 tests passed via `test_runtime_history.py`. Dynamic prompt packing within 384 tokens verified.
- [x] **Dataset Integrity Tests:** Disjointness verified. Exact duplicate pairs: 0. Benchmark contamination: 0/200 on Context, 0/100 on Conversation.
- [x] **Frozen Benchmarks:** `CR19_CONTEXT_200.json` (`3203fd77...`) and `CR19_CONVERSATION_100.json` (`1d230274...`) frozen and hashed.
- [x] **Training Formatter & Masking:** Verified with target-only supervision and prompt token masking (`-100`).
- [x] **Checkpoint Lineage:** Initialized from exact CR-17 release checkpoint `releases/v0.9.1_conversational_preview/chakrview_v091_model.pt` (`f161ccdc501bca34f6ed2019252f2e51d18fa113ce0cc90187c4ff0cdd180d10`).
- [x] **Resource & Execution Budget:** 2 arms $\times$ 3 seeds = 6 runs. 300 optimizer steps per run (effective batch size 8). Total steps: 1,800. Estimated compute time: ~10-15 minutes on GPU/CPU environment.

Precheck Status: **ALL PRECONDITIONS SATISFIED — PROCEEDING TO CONTROLLED TRAINING**

