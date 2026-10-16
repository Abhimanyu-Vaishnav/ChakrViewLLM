# Proposition Extraction Audit

## Batches 1541–1580: Multi-Entity Attribute Disambiguation & Cross-Talk Prevention

### 1. Audit Motivation & Failure Cases

In Batches 1501–1540, simple keyword proximity in `_extract_attribute_from_chunk()` occasionally suffered from **attribute cross-talk** and **false entity binding**:
1. **Adjacent Parameter Lines:** In `storage_bm25_index.md`, line 4 states `k1: 1.5` and line 5 states `b: 0.75`. Matching generic words like `"parameter"` could mistakenly extract `1.5` for parameter `b`.
2. **Document Title Collision:** The header `# BM25 Lexical Knowledge Index` was occasionally parsed as an attribute line due to the colon or number `25`.
3. **Multi-Entity Subscription Plans:** In `billing_subscription_tiers.md`, `Starter Tier: $150`, `Professional Tier: $450`, and `Dedicated Cluster Tier: $12,000` appear in the same section. Inverted queries risked binding `$150` to the wrong tier.

---

### 2. Disambiguation Architecture

The upgraded proposition extractor enforces:
1. **Markdown Header Protection:** Lines beginning with `#` are treated strictly as document/section headers, never as key-value attribute lines.
2. **Colon Boundary Enforcement (`Prefix: Suffix`):**
   - The attribute phrase must match the text *before* the colon (`prefix`).
   - The numerical or categorical value is extracted *strictly after* the colon (`suffix`).
3. **Multi-Keyword Conjunctive Matching:**
   - Multi-word attributes (e.g. `maximum node cluster diameter`, `data transmission port`) require all or high-ratio keyword presence in the attribute prefix rather than accepting partial single-word hits.
4. **Unit and Normalization Safety:**
   - Units (`ms`, `bytes`, `$`, `%`) are normalized via typed parsing functions before equality comparison.
