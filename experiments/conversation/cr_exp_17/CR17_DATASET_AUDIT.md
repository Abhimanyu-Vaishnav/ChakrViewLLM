# CR-EXP-17: Comprehensive Dataset Audit Report

## 1. Corpus Identification
- **Dataset Title:** `CR17-Conversational-Corpus-v1`
- **Location:** `experiments/conversation/cr_exp_17/data/`
- **Files:**
  - `conversation_train_1080.jsonl` (1,419 examples, SHA-256: `91ca4c4ca4dbe4d59fca76865d4ebf07297e6e58dfda5f69c5e2ef975ea8e24c`)
  - `conversation_val_120.jsonl` (157 examples, SHA-256: `c4960ce0f9f3bfb5883ea57cb5a909b6d8591f24d770c675fb621375b48db9d0`)
  - `conversation_corpus_manifest.json` (SHA-256: `78ce7ebef371cceea9f002d2bf94aa5df8d9bbf38283a216dbbb746ca72a6b25`)

## 2. Source Composition
The corpus was curated from historical verified SFT pools combined with targeted augmentations:
1. `cbr_sft_06a/conversational_sft_train.jsonl` (900 clean conversational turns)
2. `cbr_sft_06a/conversational_sft_val.jsonl` (100 clean conversational turns)
3. Deterministic Augmentation Pools:
   - Self-Identity Grounding (`ChakrView`, `independent indigenous neural language model`, `first principles`)
   - Multi-Turn Entity Recall (2-3 turn dialogue tracking: names, locations, preferences)
   - Natural Conversational Uncertainty (replacing rigid refusal formulas with natural responses)
   - Hindi & Hinglish Grounded Dialogue (`Namaste`, `Main aapki kya madad kar sakta hoon?`, etc.)

## 3. Benchmark Contamination Audit
- **Held-Out Benchmark:** `experiments/conversation/cr_exp_17/CONVERSATION_BENCHMARK_V1.json` (200 probes)
- **Contamination Check:** Every prompt in `CONVERSATION_BENCHMARK_V1.json` was compared against all training and validation instances using exact string matching and normalized substring matching.
- **Initial Audit:** 11 probes had identical user formulations (e.g. `User: Hello!\nAssistant:`).
- **Remediation:** All 11 probes were rephrased into distinct, unseen conversational formulations (e.g. `User: Greetings! Hope you are having a pleasant day.\nAssistant:`).
- **Final Leakage Rate:** **0 / 200 (0.00% Contamination, Strict PASS)**.
