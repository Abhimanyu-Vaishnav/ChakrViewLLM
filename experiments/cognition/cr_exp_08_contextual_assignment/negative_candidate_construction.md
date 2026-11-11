# CR-EXP-08: NEGATIVE CANDIDATE CONSTRUCTION SPECIFICATION
## Controlled Pairwise Negative Sampling Rules for Arm C

To satisfy the scientific invariants of Stage 2 and Stage 5 for **Arm C (Counterfactual Pairwise Assignment Objective)**, all negative candidates must adhere to deterministic, symmetric, and length-matched construction rules.

---

### 1. Core Negative Candidate Principles

1. **Semantic Validity**: Every negative candidate must be a real, orthographically correct entity of the identical entity class (e.g., Indian cities for location relations). No corrupt, garbage, or non-word tokens are permitted.
2. **Contextual Incorrectness**: The negative candidate must be strictly false under the current subject-value assignment. If the context maps `Ravi -> Delhi`, `"Delhi"` is forbidden as a negative candidate.
3. **Exact Token-Length Matching**: To prevent the pairwise margin penalty from exploiting sequence-length artifacts (as demonstrated in CR-EXP-07.2), the negative candidate must have the exact same BPE token count as the positive candidate:
   - For 3-token positives (`Delhi`, `Jaipur`, `Pune`, `Surat`, `Kanpur`, `Nagpur`, `Indore`, `Patna`), the negative candidate MUST be chosen from the length-3 pool.
   - For 4-token positives (`Mumbai`, `Bengaluru`, `Bhopal`, `Nashik`), the negative candidate MUST be chosen from the length-4 pool.
   - For 5-token positives (`Chennai`, `Kolkata`, `Hyderabad`, `Lucknow`), the negative candidate MUST be chosen from the length-5 pool.
4. **Counterbalancing & Uniform Sampling**: Negative candidates must not over-index on any single entity (e.g., never use `"Jaipur"` as the negative for all prompts). Negatives are cyclically or uniformly selected across the matching length bucket.
5. **No Negative Pairs for Non-Relational Dialogue**: Conversational preservation examples (greetings, polite answers, instruction following) do NOT participate in the pairwise objective ($L_{\text{pair}} = 0$) and are trained exclusively on standard autoregressive language modeling loss ($L_{\text{LM}}$).

---

### 2. Available Entity Length Pools in `vocab_4096`

- **Length 3 Pool (8 Cities)**:
  - `Delhi` (`[71, 904, 453]`)
  - `Jaipur` (`[77, 1000, 2382]`)
  - `Pune` (`[83, 371, 104]`)
  - `Surat` (`[86, 326, 278]`)
  - `Kanpur` (`[78, 286, 2382]`)
  - `Nagpur` (`[81, 397, 2382]`)
  - `Indore` (`[896, 546, 287]`)
  - `Patna` (`[83, 278, 709]`)

- **Length 4 Pool (4 Cities)**:
  - `Mumbai` (`[80, 120, 550, 1000]`)
  - `Bengaluru` (`[3803, 288, 326, 120]`)
  - `Bhopal` (`[69, 572, 115, 288]`)
  - `Nashik` (`[81, 427, 453, 110]`)

- **Length 5 Pool (4 Cities)**:
  - `Chennai` (`[70, 107, 275, 709, 108]`)
  - `Kolkata` (`[78, 1142, 110, 278, 100]`)
  - `Hyderabad` (`[75, 124, 762, 412, 413]`)
  - `Lucknow` (`[79, 425, 110, 768, 122]`)

---

### 3. Verification & Validation Standard

Every example generated for Arm C records:
- `y_positive`: The true contextual target string.
- `y_positive_token_ids`: Exact BPE token sequence.
- `y_negative`: The counterbalanced incorrect candidate string.
- `y_negative_token_ids`: Exact BPE token sequence.
- `token_count_match`: A strict boolean asserting `len(y_positive_token_ids) == len(y_negative_token_ids)`.
If a sample fails the length-match condition, it is rejected before training.

