# WP 861–865: Pointer Failure Audit & Root-Cause Analysis

## Executive Summary

An audit of Batches 781–860 was conducted to evaluate why the token-level pointer-generator (`ChakrPointerMicro`) failed to improve end-to-end plan accuracy on the frozen benchmark ($N=70$), remaining at **6/70 (8.6%)** across all systems, and recorded 0/42 exact copies on the argument fidelity evaluation.

Through manual inspection of individual generation traces and mechanistic decomposition of the architecture and training pipeline, the root failure modes have been isolated and confirmed.

---

## 1. Questions Investigated & Verified

### 1.1 How Prompt Tokens are Represented
- The full input sequence is encoded via `ChakrMicro`'s causal transformer layers.
- Prompt representations are obtained as `prompt_states = hidden_states[:, :prompt_len, :]` from the final RMSNorm layer ($d_{\text{model}} = 192$).

### 1.2 How the Pointer Distribution is Computed
- For any output decoding step $t$, the query vector is $q_t = W_q h_t$ and prompt keys are $k_i = W_k h_i$.
- Scaled dot-product scores: $s_{t, i} = \frac{q_t^T k_i}{\sqrt{d_{\text{model}}}}$, normalized with softmax over $i \in [0, \text{prompt\_len})$.

### 1.3 How the Generation/Copy Gate is Trained
- $p_{\text{gen}} = \sigma(W_{\text{gate}} [h_t; c_t])$ where $c_t = \sum_i P_{\text{ptr}}(i) h_i$.
- The gate is trained via standard Negative Log-Likelihood (NLL) over the combined marginal distribution:
  $$P(w) = p_{\text{gen}} P_{\text{vocab}}(w) + (1 - p_{\text{gen}}) \sum_{i: x_i = w} P_{\text{ptr}}(i)$$

### 1.4 Whether Pointer Scores are Supervised with Correct Source Positions
- **No direct pointer supervision existed.** The loss only supervised the token identity $w$, allowing the model to scatter probability to *any* token matching $w$ or simply let $p_{\text{gen}} \to 1.0$ (which averaged 0.85–0.92 during training).
- As a consequence, the pointer head never received position-specific grounding signals to disambiguate repeated tokens or select continuous span intervals.

### 1.5 How Copied Tokens are Converted into Output Text
- In token-level generation, the model samples/greedily picks a token ID from the marginal distribution $P(w)$.
- The selected token is decoded by the BPE tokenizer. It is **not** copied as a slice of raw source text.

### 1.6 Whether BPE Subwords are Selected Independently or as a Contiguous Span
- **Independently per autoregressive step.**
- In ChakrView's vocabulary (size 4,096), multi-digit numbers and phrases are frequently split:
  - `'600'` $\to$ `['6', '00']` (tokens 57, 340)
  - `'605'` $\to$ `['6', '0', '5']` (tokens 57, 51, 56)
  - `'800'` $\to$ `['800']` (single token 3749)
  - `'Decentralized...'` $\to$ 18 subword tokens.
- **Empirical Confirmation:**
  - On Item 0 (`600`): The model generated `6+3` instead of `600*5` because after emitting `'6'`, the grammar checker allowed arithmetic operators (`+`, `-`, `*`), so the model generated `+` instead of the second subword `'00'`.
  - On Item 5 (`605`): The model generated `65+3` (emitting `'6'`, then `'5'`, dropping `'0'`).
  - On Item 30 (`800`): Because `'800'` is a single atomic BPE token (3749), the model successfully emitted `800+3`!
  - On Item 48 (text chain): The model generated `'Decs s s s s s...'` looping infinitely across subwords until truncating.

### 1.7 Whether Repeated Values are Distinguished by Source Position
- The pointer head scores prompt tokens independently without positional span context. When the same digit appears multiple times (e.g., `14` in the prompt and `14` as an instruction count), the attention is diluted across all occurrences.

### 1.8 Whether the Constrained Decoder Uses the Pointer Distribution Correctly
- The constrained decoder applies token masks $M \in \{-\infty, 0\}$ over the log-probabilities of $P(w)$.
- Because $P(w)$ contains the scattered pointer probability, legal tokens that exist in the prompt are boosted. However, the mask evaluates *syntactic grammar validity of the next token*, not *semantic completeness of the operand span*.

### 1.9 Evaluation Consistency Across Systems
- Systems 1–4 used identical prompt formatting (`User: ...\nPlan:\n`), maximum generation length (160 tokens), and identical parsers.

### 1.10 Why Pointer-Constrained Syntax Validity was Lower than Baseline
- Baseline Constrained Syntax: **52.9%** (37/70).
- Pointer Constrained Syntax: **45.7%** (32/70).
- **Cause:** When pointer probabilities injected unexpected subword tokens (like partial text fragments or isolated digits) into the logits, the generation occasionally steered into dead-end grammar branches or hit the token length limit before emitting `<END_PLAN>` (observed in Item 48 where text looping exhausted the budget).

---

## 2. Definitive Conclusion

The token-level pointer generator suffers from a structural **subword BPE misalignment bottleneck**:
1. Human arguments (numbers, entities, strings) are semantic multi-character concepts.
2. Token-level pointers operate on subwords, where grammar masking permits structural operators before a multi-token literal is complete.
3. Therefore, argument grounding **must be formulated at the span level** (identifying character/token start and end indices of the complete entity), which can then be directly bound to plan argument slots.
