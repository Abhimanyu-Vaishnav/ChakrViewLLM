# MEMORIZATION AUDIT & CATASTROPHIC FORGETTING
## Batches 1621–1660 Neural Evaluation

### 1. Objective of the Memorization Audit

When small neural language models are adapted on small specialized corpora, two risks emerge:
1. **Verbatim Memorization:** The model memorizes training sequences word-for-word without understanding underlying rules.
2. **Catastrophic Forgetting:** The model overwrites its general language knowledge, regressing on general language continuation and producing malformed or degenerate text.

This audit evaluates both phenomena on Candidate Checkpoint `75492f1d...` compared to Canonical `96357705...`.

---

### 2. Evidence of Memorization vs Generalization

#### 2.1 Memorization of Token Collocations
- In Track 1 (Language Continuation), the canonical baseline generated:
  `"in the world, which is the world of the world."`
- The adapted candidate (after 100 steps on the new corpus) generated:
  `"packets. It is 36 packets. It is 36 pac"`
  and:
  `">>> linked_list.insert(')"`
- **Analysis:** The candidate model rapidly memorized high-frequency corpus n-grams (`"36 packets"`, `">>> linked_list"`, `"archive limit"`). Rather than generalizing the context of distributed systems, it substituted previously learned generic loops (`"world of the world"`) with newly learned specific n-gram loops (`"36 packets"`).

#### 2.2 Lack of Exact-Sequence Memorization on Facts
- In Track 5, the model was trained on 15 specific factual statements such as `"Service auth operates on TCP port 7001."`.
- When tested with `"Service auth operates on TCP port"`, the candidate generated:
  `"ing to archive limit of 27 packets. The relayer"`
- **Analysis:** Even after 100 steps, the model did **not** achieve verbatim memorization of low-frequency individual facts. Instead, cross-attention and dense representations suffered interference from the more frequent structural patterns in the training set.

---

### 3. Catastrophic Forgetting Assessment

| Area | Canonical Checkpoint | Candidate Checkpoint (Step 100) | Forgetting Observed? |
| :--- | :--- | :--- | :---: |
| **Track 1 Language Generation** | 30/30 passed (valid English text) | 30/30 passed (valid English text) | **No** (syntax remains fluent) |
| **Vocabulary Diversity** | Baseline general vocabulary | Infused with corpus terms | **Slight drift towards training jargon** |
| **Loss of Common Token Flow** | Predicts standard English tokens | Replaces some pronouns with technical nouns | **Minor stylistic degradation** |
| **Generation Stability** | Never produces NaN or crashes | Never produces NaN or crashes | **No** |

The candidate checkpoint did not suffer catastrophic collapse. However, its generation exhibits strong lexical attraction toward recent training tokens, manifesting as repetitive technical loops.

---

### 4. Summary & Takeaways
1. ChakrMicro at 1.58M parameters is too small to reliably memorize dozens of disjoint factual associations without high-step overfitting.
2. The model exhibits *pattern attraction* rather than *semantic generalization*: it picks up syntax templates and token fragments, inserting them indiscriminately into prompts that share surface keywords.
3. Factual storage in model weights is noisy and fragile at this scale; deterministic retrieval (BM25) and structured stores are orders of magnitude more reliable for factual precision.
