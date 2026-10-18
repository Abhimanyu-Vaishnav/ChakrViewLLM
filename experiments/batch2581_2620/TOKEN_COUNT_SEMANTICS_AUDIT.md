# TOKEN-COUNT SEMANTICS AUDIT
**Batch:** 2581–2620  
**Repository:** ChakrViewLLM (`dev/v0.6-v0.7-validation`)  
**Investigated Assets:**
- `experiments/batch2021_2060/SPLIT_V2_MANIFEST.json`
- `experiments/batch2101_2140/CORPUS_V2_CLEANED_MANIFEST.json`
- `experiments/batch2141_2180/cpt_runner.py`
- `release/chakrview-v0.5.0/data/experiments/vocab_4096/` (Canonical BPE Tokenizer)

---

### 1. Disambiguation of 4 Distinct Token Quantities

In previous audit batches, different token statistics were occasionally referenced interchangeably. We formally define and distinguish four separate token metrics:

1. **Raw Encoded Tokens ($T_{\text{raw}}$):**
   - The exact length of the token ID list produced by encoding the entire document text with the canonical tokenizer: $\text{len}(\text{tokenizer.encode}(\text{doc.text}))$.
   - For the 782 training documents, $T_{\text{raw}} = \mathbf{147,476}$.
   - This represents the total semantic mass in tokens without padding, sliding, or context truncation.
2. **Tokens Counted by the Manifest ($T_{\text{manifest}}$):**
   - In `SPLIT_V2_MANIFEST.json`, `split["train"]["token_count"]` is declared as $\mathbf{147,476}$.
   - In `CORPUS_V2_CLEANED_MANIFEST.json`, each document has a `token_count` field equal to its raw encoded length.
   - Therefore, $T_{\text{manifest}} \equiv T_{\text{raw}}$. The manifest explicitly records raw token counts.
3. **Chunked Target Tokens for Static Evaluation ($T_{\text{eval\_targets}}$):**
   - For sequential, non-stochastic loss evaluation (as in `PER_DOCUMENT_LOSSES.jsonl` or sliding validation), sequences longer than context window $C=256$ are chunked into contiguous windows of stride $S=255$ and context length $C=256$, where target tokens are shifted by 1 ($N-1$ targets per window of $N$ tokens).
   - Across the 782 training documents, this tiling yields exactly $\mathbf{146,694}$ target tokens.
   - Across the 290 held-out test documents, this yields exactly $\mathbf{64,769}$ target tokens (compared to 65,665 raw tokens).
4. **Tokens Presented to the Actual Training Pipeline ($T_{\text{train\_processed}}$):**
   - Inspection of `experiments/batch2141_2180/cpt_runner.py` (lines 180–204) reveals:
     - Training samples batches stochastically with replacement from `tokenized_train` using Python's `random.Random(seed)`.
     - For sequences $> 257$ tokens, a random window of 257 tokens is selected: `s_idx = rng_sample.randint(0, len(seq) - 257)`.
     - Target tokens per micro-batch sample are $256$ (or $\text{len}(seq) - 1$ if shorter, padded to 256 with target $-100$).
     - The training loop runs for $S_{\text{max}} = 200$ steps with `batch_size = 4` and `grad_accum = 2` (effective batch size 8).
     - At each step, $8 \times 256 = 2,048$ target tokens are processed.
     - Over 200 steps, $T_{\text{train\_processed}} = 200 \times 2,048 = \mathbf{409,600}$ training target tokens processed (sampling with replacement across the $147\text{k}$ token corpus).
   - Crucially: **The CPT training pipeline does NOT perform a single static 1-epoch pass over the 146,694 target tokens**. It samples random 256-token sub-sequences dynamically.

---

### 2. Reconciliation Summary

| Quantity | Scope | Count | Semantic Interpretation |
| :--- | :--- | :--- | :--- |
| **Raw Encoded Tokens** | Train Split (782 docs) | **147,476** | Pure BPE encoded sequence lengths |
| **Manifest Declared Tokens** | Train Split | **147,476** | Equal to sum of raw encoded tokens |
| **Static Chunked Targets** | Train Split | **146,694** | Tiled evaluation targets (stride 255, context 256) |
| **Training Steps $\times$ Batch Targets** | CPT Run (200 steps) | **409,600** | $200 \text{ steps} \times 8 \text{ batch} \times 256 \text{ tokens/seq}$ |
| **Manifest Declared Tokens** | Test Split (290 docs) | **65,665** | Pure BPE encoded sequence lengths |
| **Static Chunked Targets** | Test Split | **64,769** | Exact target tokens evaluated in held-out benchmark |
