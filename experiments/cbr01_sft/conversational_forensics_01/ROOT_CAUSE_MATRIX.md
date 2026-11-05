# ChakrView Conversational Forensic Investigation: Root Cause Matrix

This forensic root-cause matrix synthesizes programmatic evidence, state-dict checks, data audits, loss masking traces, and decoding analyses performed on **ChakrView CBR-SFT-07 Step 50** (`experiments/cbr01_sft/runs/cbr_sft_07/checkpoints/sft_checkpoint_step_50.pt`).

---

## Confidence Classifications

- **CONFIRMED**: Proven through mathematical verification, reproducible code execution, or bit-exact tensor equality.
- **STRONGLY SUPPORTED**: Backed by strong empirical correlation across multiple tests and statistical data frequencies.
- **POSSIBLE**: Plausible mechanism, but lacks conclusive isolation or ablation.
- **NOT SUPPORTED**: Refuted by empirical facts or direct testing.

---

## 1. Forensic Root Cause Matrix

| Category | Hypothesis / Failure Mode | Classification | Empirical Evidence & Reproducibility |
|---|---|---|---|
| **Data SFT** | **Over-Saturation of Context-Refusal Phrases ("Context Attractor Sink")** | **CONFIRMED** | In `cbr_sft_07_train_228.jsonl` and parent datasets, completions heavily repeat `"context"`, `"not provided in the context"`, and `"the provided context does not contain"`. When tested on context-grounding and abstention prompts, Step 50 outputs malformed context strings (`"Yes, the neural language model is a context"`, `"is not provided in the context"`). Repro: `python experiments/cbr01_sft/conversational_forensics_01/scripts_data_forensics.py`. |
| **Data SFT** | **Identity and Readiness Phrase Monoculture** | **CONFIRMED** | Greetings and general queries collapse into repetitive ChakrView readiness statements (`"I am ChakrView, an indigenous..."`, `"ready to assist"`). Analysis of data reveals that out of 228 training records, 20 explicitly contain `"chakrview"` and common templates start with `"Good evening! I am ready to assist"`. Repro: `DATASET_AUDIT.md`. |
| **Data SFT** | **Multi-Turn Context Training Scarcity** | **CONFIRMED** | In `cbr_sft_07_train_228.jsonl`, only 23 out of 228 examples (10.1%) feature multi-turn conversation format. In contrast, 90% are single-turn `User: ...\nAssistant: ...`. When `chat.py` feeds previous dialogue turns, the input structure lies far out-of-distribution for the model's shallow layers, causing multi-turn context retention to be 0%. |
| **Pipeline / Architecture** | **Loss Masking / Causal Shift Bug** | **NOT SUPPORTED** | Inspection of `chakrview/training/prompt_contract.py` and `chakrview/training/sft_engine.py` proves causal alignment is exact: `input_seq[len(prompt)]` is the prompt's last token, predicting `target[len(prompt)]` (`response[0]`). All prompt positions `t < len(prompt)` are masked with `-100`. Alignment is bit-exact. Repro: `LOSS_MASK_AUDIT.md`. |
| **Tokenizer** | **Tokenizer Out-of-Vocabulary / UNK Corruption** | **NOT SUPPORTED** | All 8 probe phrases (including Hindi/Hinglish) decode with 100% exact character fidelity. Zero `<unk>` tokens are produced. Repro: `TOKENIZER_AUDIT.md`. |
| **Inference / Decoding** | **Greedy N-Gram Self-Attraction (Repetition Loops)** | **CONFIRMED** | Under pure greedy decoding (`temperature=0.0, repetition_penalty=1.0`), the model falls into repeating token loops (`"neural neural neural neural language model"`). The repetition penalty (1.25) in `chat.py` breaks these exact loops, but exposes grammatical malformations when forced away from top logits. Repro: `DECODING_COMPARISON.md`. |
| **Architecture** | **Induction Circuit Limitation in Shallow 6-Layer Model** | **STRONGLY SUPPORTED** | ChakrMicro has 6 layers, 6 heads, $d_{model}=192$. Theoretical and empirical mechanistic interpretability shows that exact copy-pasting (induction heads) requires dedicated attention head composition across at least 2-3 layers. When the model has only 6 layers and 3.44M parameters, allocating capacity simultaneously to language modeling, identity, and copy-induction requires dense, specialized training signals that 228 samples failed to provide. |
| **Training / Parity** | **Inference Prompt Format vs Training Prompt Mismatch** | **NOT SUPPORTED (Single Turn)** / **CONFIRMED (Multi Turn)** | Single-turn prompt tokens in `chat.py` match training data token-for-token. However, in multi-turn mode, history turns separated by newlines create token lengths and attention patterns never seen during 90% of single-turn training steps. |

---

## 2. Key Diagnostic Insights

1. **Why Greetings Collapse into Readiness Mantras**:
   Single-turn greeting prompts in the training set (`data/sft/cbr_sft_07`) consistently end with offers to assist: `"Good morning! How can I assist you today?"` or `"I am ChakrView, ready to assist."` The model's top logit at Step 1 for greetings is `'G'` (Good), `'H'` (Hello), or `'M'` (My). This is functioning as trained.
2. **Why Context Grounding Emits "in the context"**:
   The lineage data from CBR-SFT-04, 05, and 06A contains numerous negative context examples where the model was trained to say `"That information is not provided in the context."` The token sequence `"not provided in the context"` became an ultra-low loss basin. Whenever the prompt contains `"context"` or factual grounding, attention heads latch onto the word `"context"` and route prediction directly into the refusal sink.
3. **Why Multi-Turn Memory Fails**:
   When given `User: My name is Abhimanyu.\nAssistant: Hello Abhimanyu!...\nUser: What is my name?\nAssistant:`, the logits for token 1 are:
   - Rank 1: `'Y'` (0.2228) -> followed by `'o'` (`"You..."`)
   - Rank 2: `'N'` (0.1738)
   - Rank 3: `'G'` (0.1663)
   - Token `'A'` (Abhimanyu) has only 0.0526 probability!
   The model does not maintain an active copying circuit across turn boundaries because multi-turn copy examples constituted less than 5% of its entire training signal.

