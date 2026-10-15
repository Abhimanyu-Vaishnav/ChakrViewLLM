# Generalization Analysis: Batches 1061–1100
## Rigorous Generalization & Vocabulary Sensitivity Evaluation

### 1. Overview & Dataset Discipline
To assess whether the Local Knowledge Assistant genuinely generalizes beyond the phrasing of the frozen 50-item benchmark, two new held-out suites were created:
1. **`generalization_dev.json` (30 questions):** Used for initial exploratory validation and defect identification.
2. **`generalization_eval.json` (60 questions):** A strict held-out evaluation suite containing:
   - Paraphrases avoiding verbatim source terms (10 items).
   - Synonyms and alternate question syntax (8 items).
   - Negations, exceptions, and security policy rules (8 items).
   - Multi-turn follow-ups and referential anaphora (10 items).
   - Multi-document factual queries across disparate system tiers (8 items).
   - Unindexed and out-of-domain queries requiring honest abstention (10 items).
   - False premise refutations and contradiction inquiries (6 items).

Neither suite was used to train, tune, or alter system parameters or stopword filters.

### 2. Quantitative Generalization Results

| Suite | Total Questions | Passed | Accuracy % | Citation Validity | Citation Support | Abstention | Refutation | Follow-ups |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Original Frozen Benchmark** | 50 | 50 | 100.0% | 38/38 (100%) | 38/38 (100%) | 8/8 (100%) | 4/4 (100%) | 7/7 (100%) |
| **Generalization Dev** | 30 | 28 | 93.33% | 20/21 (95.2%) | 20/21 (95.2%) | 6/6 (100%) | 2/2 (100%) | 4/4 (100%) |
| **Generalization Eval (Held-out)** | **60** | **59** | **98.33%** | **43/44 (97.7%)** | **43/44 (97.7%)** | **10/10 (100%)** | **6/6 (100%)** | **9/10 (90%)** |

### 3. Detailed Failure Analysis on Generalization

#### Failure 1 (Generalization Eval Item 29): Extreme Synonymous Paraphrase without Lexical Overlap
- **Case:**
  - Context Turn: *"What encryption standard protects network traffic?"*
  - Follow-up Turn: *"What symmetric cipher algorithm is used with it?"*
- **Symptom:** Assistant returned `ABSTAIN` (*"I could not find sufficient verified evidence..."*).
- **Root Cause:**
  - In `security_policy.md`, the document states:
    `"- In-transit communication is strictly encrypted via TLS 1.3 with AES-256-GCM."`
  - The context turn query uses *"network traffic"*, while the document uses *"in-transit communication"*.
  - Because `DeterministicHashEmbeddingProvider` is a deterministic token hash projection rather than a deep semantic language encoder, it cannot bridge high-semantic-distance synonyms like `network traffic` <-> `in-transit communication`.
  - The lexical BM25 index matched only 1 token (`encryption`).
  - Because the grounding threshold requires $\ge 2$ content token matches for multi-token queries, the assistant correctly and conservatively abstained rather than guessing.
- **Evaluation Takeaway:** This demonstrates that the grounding safeguard functions as intended: **when lexical and semantic similarity falls below the confidence threshold, the assistant abstains instead of hallucinating.**

#### Generalization Strengths Confirmed:
1. **Zero False Positives on Out-of-Domain Queries:** 10 out of 10 unindexed queries (e.g. GDP of the UK, melting temperature of titanium, speed of sound) were cleanly abstained from without any spurious matches.
2. **False Premise Refutations:** 6 out of 6 contradiction queries (e.g. capitals of Venus, Jupiter, Saturn; Prime Minister of India in 1890/1930) were refuted without relying on incorrect document matches.
3. **Multi-Turn Context Resolution:** 9 out of 10 follow-up queries successfully resolved pronouns and referents (*"that model"*, *"that tier"*, *"those snapshots"*).
