# ChakrView Experiment 10 — Controlled Multi-Token Span Copying Design

**Date**: 2026-10-21  
**Experiment**: Research-Only Multi-Token Span Copying Prototype (`exp10_span_pointer`)  
**Parent Checkpoint**: Canonical Parent Checkpoint `Step-10000` (`experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`)  
**Base Architecture**: ChakrMicro (3.44M parameters, 6 layers, $d_{\text{model}}=192$, context 512)  

---

## 1. Problem Statement & Motivation

Experiment 09 proved that:
- Pure vocabulary-projected next-token SFT fails on unseen token identities (0.7% exact match).
- An isolated single-token pointer head achieved **100.0% position accuracy** across all evaluation splits and broke the zero-shot barrier on held-out tokens (93.7% exact match).

However, Experiment 09.1 established that:
- The 6.3%–10.7% exact-match gap in the single-token pointer was caused entirely by **BPE Delimiter Boundary Merges**: when words appear adjacent to syntax punctuation (e.g. `primary:mode` or `Beta=caus, `), the BPE tokenizer merges the word with delimiters into multi-token sequences (`mo` + `de `, or `c` + `au` + `s, `).
- Real-world structured extraction (such as JSON key-value extraction, variable names, URLs, filepaths, and numerical values) frequently consists of **contiguous multi-token spans**.

**Experiment 10 Mission**:
Extend the pointer approach from single-token copying to **contiguous multi-token span copying** using a dual-boundary pointer head (predicting start and end position pairs $[s, e]$ where $s \le e \le s + L_{\max}$), enabling 100% extraction fidelity across both single-token and multi-token values, including those adjacent to punctuation.

---

## 2. Theoretical Architecture of the Span Pointer

Given an input prompt sequence $X = [x_0, x_1, \dots, x_{T-1}]$:
1. **Backbone**: Passes $X$ through ChakrMicro's transformer blocks to obtain contextual hidden representations $H \in \mathbb{R}^{T \times d_{\text{model}}}$.
2. **Query Conditioning**: Uses the query state $h_{\text{query}} = H[T-1] \in \mathbb{R}^{d_{\text{model}}}$ at the final prompt position.
3. **Dual Pointer Projections**:
   - Start Scoring: $\text{Score}_{\text{start}}(i) = \frac{(W_{q,\text{start}} h_{\text{query}})^T (W_{k,\text{start}} H[i])}{\sqrt{d}}$
   - End Scoring: $\text{Score}_{\text{end}}(j) = \frac{(W_{q,\text{end}} h_{\text{query}})^T (W_{k,\text{end}} H[j])}{\sqrt{d}}$
4. **Constrained Masking & Ineligible Filtering**:
   - Ineligible prompt tokens (overhead, delimiters, prompt template boilerplate) are masked with $-\infty$.
   - Valid span constraint: $s \le e \le s + L_{\max}$ (where $L_{\max} = 6$ tokens).
5. **Loss Function**:
   $$\mathcal{L} = \mathcal{L}_{\text{start}} + \mathcal{L}_{\text{end}} = \text{CE}(\text{Score}_{\text{start}}, s^*) + \text{CE}(\text{Score}_{\text{end}}, e^*)$$
6. **Execution / Extraction**:
   - Identifies $(\hat{s}, \hat{e}) = \operatorname{argmax}_{s \le e} (\text{Score}_{\text{start}}(s) + \text{Score}_{\text{end}}(e))$.
   - Extracts sub-sequence: $\hat{Y} = [x_{\hat{s}}, x_{\hat{s}+1}, \dots, x_{\hat{e}}]$.
   - Decodes: $\text{Output} = \text{Tokenizer.decode}(\hat{Y})$.

---

## 3. Dataset Design & Task Categories

The dataset consists of 2,400 training examples, 300 validation examples, and 600 test examples (300 Test A familiar tokens/spans, 300 Test B held-out values):
- **Category A**: Single-token regression control (clean single tokens).
- **Category B**: Multi-token values (2 to 4 tokens: e.g. compound words, software identifiers like `auth_token`, `max_iter`, `learning_rate`).
- **Category C**: Punctuation-adjacent spans (directly following colons `:`, equals `=`, brackets `[]`, or preceding commas `,`).
- **Category D**: Distractor values with overlapping subwords.
- **Category E**: Repeated values at multiple source locations.
- **Category F**: Novel contexts with randomized distractor keys.
- **Category G**: Held-out target identities (values whose subwords never appeared as supervised targets in training).

### Ground-Truth Span Derivation Guarantee:
- Spans are derived strictly by sub-sequence matching on the **actual encoded token sequence** $[0] + \text{encode}(prompt)$.
- Every example is asserted: $\text{decode}([x_{s^*}, \dots, x_{e^*}]).\text{strip}() == \text{target\_value}.\text{strip}()$.
- Any example that cannot be verified to match the ground truth span under tokenizer round-trip is marked invalid and reported.

---

## 4. Evaluated Baselines & Comparison

1. **Deterministic Lookup Baseline**: Direct string extraction using regex/key pattern matching (non-neural upper bound).
2. **Single-Token Pointer Baseline**: The Experiment 09 single-token pointer architecture (evaluating its failure mode on multi-token spans).
3. **Span-Pointer Prototype**: The newly trained factorized start/end pointer head.

---

## 5. Predeclared Evaluation Metrics

For every split and category:
1. **Span Exact Match (Span EM)**: $\hat{s} = s^* \land \hat{e} = e^*$.
2. **Token Sequence Match (Token EM)**: $[x_{\hat{s}}, \dots, x_{\hat{e}}] == [x_{s^*}, \dots, x_{e^*}]$.
3. **Decoded Value Match (Value EM)**: $\text{decode}([x_{\hat{s}}, \dots, x_{\hat{e}}]).\text{strip}() == \text{target\_value}.\text{strip}()$.
4. **Start Position Accuracy**: $\hat{s} == s^*$.
5. **End Position Accuracy**: $\hat{e} == e^*$.
6. **Invalid Span Rate**: Percentage of spans where $\hat{e} < \hat{s}$.
