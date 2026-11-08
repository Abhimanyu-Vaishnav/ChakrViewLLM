# CR-EXP-14 Phase 7: Qualitative & Quantitative Failure Analysis

## 1. Overview of Evaluated Failure Modes

Comprehensive qualitative examination of the detailed generation records (`candidate_a_sealed_detailed.json`, `candidate_b_sealed_detailed.json`, `candidate_c_sealed_detailed.json`) demonstrates that the failure of dynamic entity-value binding in ChakrMicro stems from four distinct mechanistic failure modes:

1. **Hallucinatory Subject Substitution (Dominant Mode)**
2. **Subword Fragmentation & Representation Degradation**
3. **Temporal Recency Decay Across Multi-Turn Separators**
4. **Structural Delimiter Memorization vs. Transfer Failure**

---

## 2. Detailed Breakdown of Primary Failure Modes

### Failure Mode 1: Hallucinatory Subject Substitution
Across 100% of the state-tracking queries in the sealed benchmark, when asked `"Where is [Unseen Entity] currently located?"`, the model replaces the query entity with high-frequency training names.
- **Example (Candidate A):**
  - Context: `Abhay moved to Dhanbad.`
  - Query: `Where is Abhay currently located?`
  - Generation: `'Ankita is currently in Amritsh is currently in Amritsar'`
  - Error: Replaces `Abhay` with `Ankita` and `Dhanbad` with `Amritsar` (both prominent in the training set).
- **Example (Candidate B):**
  - Context: `Chetan moved to Firozabad.`
  - Query: `Where is Chetan now?`
  - Generation: `'Amita is currently in Aurandabad.'`
  - Error: Replaces `Chetan` with `Amita` and `Firozabad` with `Aurandabad`.

### Failure Mode 2: In-Distribution Conditioning vs. Out-of-Distribution Collapse
When diagnostic tests were executed using in-distribution training entities:
- Query on training entity `Aaditya` after moving to `Bhopal`:
  - Generation: `'is currently in Bhopal.'`
- The model successfully recalled the updated city `Bhopal` when the entity was an in-distribution training token!
- However, when two entities were introduced into the same dialogue context (`Aaditya` and `Aakash`), querying `Aaditya` produced `'Anita is currently in Amritsar.'`, while querying `Aakash` produced `'Pratch, is currently in Bhopal.'`.
- This demonstrates that the model did not learn a generalizable **Induction Head** (copy query entity $\to$ search previous context for matching entity $\to$ extract corresponding city value). Instead, it learned a **distributed associative lookup table** indexed by specific token embeddings.

### Failure Mode 3: Subword Tokenization Mismatch
In ByteBPE (vocabulary size 4096), familiar training names have unified or compact tokenizations (e.g. `Ananya` $\to$ `['An', 'anya']`), whereas unseen benchmark entities are split into fragmented subwords:
- `Abhay` $\to$ `['A', 'bha', 'y ']` (3 tokens)
- `Bimla` $\to$ `['B', 'im', 'la']` (3 tokens)
- `Chetan` $\to$ `['Ch', 'et', 'an']` (3 tokens)
- `Damodar` $\to$ `['D', 'am', 'od', 'ar']` (4 tokens)
Because rotary position embeddings (RoPE) and self-attention heads at $d_{\text{model}}=192$ must propagate multi-token span representations across 6 layers without pre-trained semantic priors, subword fragments fail to form bound entity representations.

### Failure Mode 4: Delimiter Overfitting (Candidate C)
Candidate C, trained with structured delimiters (`<ent>...</ent>` and `<val>...</val>`), achieved the lowest training loss ($0.5707$) and preserved the highest abstention rate ($88\%$). However, on natural language benchmark prompts lacking explicit XML tags, Candidate C generated malformed fragments:
- Generation: `'Anandrucknow.Madurai<isia.'`
- The delimiters acted as rigid syntactic triggers rather than invariant semantic scaffolding.
