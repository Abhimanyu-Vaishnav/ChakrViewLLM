# CR-EXP-07.1 OUTPUT-PRIOR HYPOTHESIS REASSESSMENT
## Disentangling Expected Syntax (Leading Whitespace) from Semantic Prior Bias

- **Investigation**: CR-EXP-07.1 (Answer-Position & Token-Conditional Probability Audit)
- **Target Question**: Is the reported dominance of Token ID 35 (`' '`) at the `Answer:` delimiter a genuine conversational output prior or simply the expected whitespace token preceding the answer?
- **Audited Checkpoints**:
  - Frozen Parent (`sft_checkpoint_step_200.pt` — `327118ce...`)
  - Conversational Pilot (`conversational_pilot_best_val.pt` — `ea6ee570...`)
  - Control Arm A (`control_arm_best_val.pt` — `729d5812...`)
  - Treatment Arm B (`treatment_arm_best_val.pt` — `2d8cea93...`)

---

### 1. The Core Empirical Finding: Two Distinct Phenomenological Layers

CR-EXP-07 claimed:
> *"The output head is dominated by a static conversational prior (Token 35 `' '` at $P > 0.73$) that prevents contextual information from being decoded into the target entity."*

CR-EXP-07.1 resolves this into two distinct, quantifiable components:

#### Layer 1: The Whitespace Artifact (Resolved)
- When prompted with `Answer:` (ending with token `61` `':'`), the standard linguistic syntax of the training corpus requires a space before any following word.
- Token ID 35 (`' '`) is emitted with $P = 77.5\% - 86.2\%$ across the conversational models precisely because **a space is the syntactically correct delimiter separator**.
- Comparing the probability of Token 35 (`' '`) against the first token of a city (e.g. `'D'` in `"Delhi"`, Token 71) was a **methodological confounder**: it pitted a generic whitespace delimiter token against a semantic word token.
- **Classification of the Whitespace Component**: **REFUTED / RECLASSIFIED AS SYNTACTIC CONVENTION**. Token 35 at `Answer:` is simply normal whitespace insertion.

#### Layer 2: The Semantic Prior Bottleneck (Confirmed)
- When the expected trailing whitespace is supplied in the prompt (`Answer: `, terminating in merged token `2550` `'er: '`), the next token to be generated is now strictly the semantic continuation.
- Under this correct prefix, what does the model emit?
  - Does it emit `"Delhi"`, `"Jaipur"`, or any contextual city name? **NO.**
  - Across all conversational checkpoints, the top-1 predicted token is **Token 13 (`'\n'`) at $P \approx 25\%$**, followed by **Token 276 (`', '`) at $P \approx 10-16\%$**, followed by conversational punctuation and uppercase sentence-starters like `'The '` (Token 898).
  - Greedy decoding from `Answer: ` generates:
    `"\n\nShandard the context.\n\nN"`
  - Candidate city tokens (e.g., `'D'`, Token 71) achieve ranks between 9 and 73 with individual probabilities $P < 0.0002$.
- Furthermore, under teacher-forced conditional log probability scoring across candidate entities (`Delhi`, `Jaipur`, `Mumbai`, `Chennai`, `Kolkata`), the model's ranking of cities is **static**:
  - Candidate city log probabilities differ by up to **$9.0$ nats** based on static unigram/bigram token frequencies in the pre-training vocabulary (e.g., `"Delhi"` and `"Chennai"` have massive static likelihood advantages over `"Mumbai"` or `"Jaipur"`).
  - Contextual reassignment (e.g., moving Ravi from Delhi to Jaipur) produces a real, measurable context boost of only **$\approx 0.32$ nats** in favor of the contextual city.
  - Because the static token frequency advantage ($3.8$ to $9.0$ nats) is over an order of magnitude larger than the contextual attention boost ($0.32$ nats), the model **never re-ranks the contextual city to top-1**.
- **Classification of the Semantic Prior Component**: **VERIFIED**. The model's output distribution over semantic tokens is dominated by static token frequencies and conversational dialogue continuation habits, drowning out the weak contextual signal.

---

### 2. Hypothesis Reassessment Classification Matrix

| Hypothesis / Claim | Status | Detailed Empirical Rationale |
|---|---|---|
| **Claim A: Token 35 at `Answer:` represents conversational filler** | **REFUTED** | Token 35 is ASCII space (`' '`), which is standard English orthography after a colon delimiter. It is not an "epistemic conversational prior", but a syntactic separator. |
| **Claim B: Supplying trailing whitespace restores entity generation** | **REFUTED** | Prompting with `Answer: ` does NOT cause the model to output the target city. Instead, it emits `'\n'`, `', '`, and conversational fragments. Target entity probability remains $<0.0002$. |
| **Claim C: Semantic output distribution is dominated by static priors** | **VERIFIED** | Teacher-forced scoring across candidate cities demonstrates that static city unigram priors dominate by $3.8 - 9.0$ nats, whereas contextual binding modulates the log probability by only $0.15 - 0.44$ nats. |
| **Claim D: Causal activation intervention modulates candidate likelihood** | **SUPPORTED** | Intervening at Layers 0–4 with true entity activations shifts candidate log likelihoods by $+0.20$ to $+0.45$ nats specifically in favor of the donor city, with $0.0000$ shift on control tokens. |
| **Claim E: Relational failure is a mixed architectural and prior bottleneck** | **VERIFIED** | The failure is dual: (1) Weak attention binding in the residual stream (yielding only $\approx 0.3$ nats of contextual preference), and (2) An overpowering static output distribution favoring frequent dialogue tokens. |

---

### 3. Conclusion

The claim that *"the model output is blocked by Token 35 as a conversational prior"* was an artifact of comparing whitespace against semantic tokens. 

However, **the deeper scientific conclusion holds true with even greater precision**: once whitespace is properly factored out, the model still fails completely to generate the contextual entity because **static semantic priors overpower the weak contextual binding signal by more than 10 to 1 in log probability space**.

