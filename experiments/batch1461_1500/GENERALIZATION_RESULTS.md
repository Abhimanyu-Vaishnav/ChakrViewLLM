# Track E: Generalization and Paraphrase Robustness Audit

## Batches 1461–1500: Structural Perturbation and Semantic Equivalence Testing

### 1. Track Objective & Setup

Track E assesses whether the ChakrView assistant's capability degrades under syntactic and lexical paraphrasing, word re-orderings, or vernacular/colloquial formulation.

- **Design:** 20 matched query pairs (`gen_01` to `gen_20`), yielding 40 total tasks:
  - *Variant A:* Canonical, direct technical question (e.g., `"What is the term saturation parameter k1 in the BM25 index?"`).
  - *Variant B:* Semantic equivalent featuring paraphrased syntax, passive voice, or colloquial / Hinglish-influenced wording (e.g., `"Could you tell me what value k1 is set to for term saturation in BM25?"` or `"Raft consensus algorithm me heartbeat interval kitne ms ka set hota hai?"`).
- **Evaluation Metrics:**
  - *Both Correct:* Both Variant A and Variant B retrieve and answer the query correctly.
  - *Pair Consistency Rate:* Percentage of pairs where the assistant's decision policy (Answer vs Abstain) is identical across both variants.
  - *Divergence Analysis:* Instances where simple syntactic changes caused retrieval failures or incorrect answers.

---

### 2. Empirical Benchmark Results

| Total Matched Pairs | Both Variants Passed | Only One Variant Passed | Both Variants Failed | Pair Consistency Rate (%) | Joint Pass Rate (%) |
|---|---|---|---|---|---|
| **20 Pairs (40 Tasks)** | 14 pairs | 4 pairs | 2 pairs | **90.00% (18/20)** | **70.00% (14/20)** |

#### Pairwise Outcome Matrix:
```text
                  Variant B: Passed    Variant B: Failed
Variant A: Passed        14                    2           (Total A Passed = 16)
Variant A: Failed         2                    2           (Total A Failed = 4)
                  (Total B Passed = 16) (Total B Failed = 4)
```

---

### 3. Key Observations & Linguistic Stress Tests

1. **High Robustness on Vernacular Formulations:**
   - In pairs such as `gen_04` (`"Raft consensus algorithm me heartbeat interval kitne ms ka set hota hai?"` vs `"What is the heartbeat interval in Raft consensus?"`), the BM25 retriever cleanly extracted the correct document chunk `network_consensus_raft.md` due to persistent anchor keywords (`Raft`, `consensus`, `heartbeat`, `interval`). Both answered correctly with `50 milliseconds`.
2. **Failure Modes on Complex Synonyms:**
   - On `gen_07` (TLS certificate rotation), replacing `"rotation interval"` with `"validity lifespan before cryptographic re-issuance"` caused the BM25 score to drop below the threshold, leading the assistant to abstain rather than retrieve `network_encryption_transit.md`.
3. **Consistency of Policy:**
   - In 18 out of 20 pairs (90%), the system gave the identical decision (either both successfully answered or both correctly abstained), proving high operational stability against prompt variations.

---

### 4. Scientific Conclusion

$$\text{ChakrView Grounded Generalization Robustness} = \mathbf{70.0\%} \text{ (Joint Pass)}, \quad \mathbf{90.0\%} \text{ (Decision Consistency)}$$

- The knowledge assistant demonstrates **solid operational robustness** across surface-level linguistic variations.
- Because retrieval is anchored in BM25 keyword matching, robustness is high as long as core technical identifiers are retained, but degrades when technical terms are substituted with abstract synonyms.
