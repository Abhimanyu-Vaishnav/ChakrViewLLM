# SOURCE-WISE EVALUATION & SANSKRIT DRIFT DIAGNOSIS

**Batch:** 2261–2300  
**Focus:** Independent Verification of Source-Wise Cross-Entropy and Classical Sanskrit Drift  
**Data Source:** `experiments/batch2261_2300/EVALUATION_AUDIT.json` (Fresh Independent Run)  

---

## 1. Disaggregated Source-Wise Performance Breakdown

The held-out test split comprises **4 distinct source families** across 290 documents and 64,769 tokens. The comparison between Canonical Baseline (Step 10,000) and CPT Candidate Seed 42 (Step 125) reveals:

| Source Family Identifier | Domain Classification | Document Count | Token Count | Baseline Token CE | Seed 42 Token CE | Delta ($\Delta$) | Baseline PPL | Seed 42 PPL |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `gutenberg_loney_coordinate_geometry` | Classical Geometry (English) | 150 | 24,482 | 4.4315 | **4.1495** | **-0.2820** | 84.06 | **63.40** |
| `wikipedia_en_information_theory` | Mathematics & Tech (English) | 70 | 15,912 | 4.9157 | **4.4737** | **-0.4420** | 136.41 | **87.68** |
| `wikipedia_hi_sanskrit_sahitya` | Classical Literature (Hindi) | 9 | 4,798 | 3.9203 | **3.9057** | **-0.0146** | 50.42 | **49.69** |
| `wikipedia_sa_bhagavad_gita` | Classical Sanskrit Philosophy | 61 | 19,577 | **4.3202** | 4.3978 | **+0.0776** | **75.20** | 81.27 |
| **All Test Sources Combined** | Full Held-Out Benchmark | **290** | **64,769** | **4.4789** | **4.2861** | **-0.1928** | **88.14** | **72.68** |

---

## 2. Independent Verification of the Sanskrit Drift Claim

The prior report claimed that Classical Sanskrit cross-entropy exhibited a drift of approximately **+0.076 to +0.077** for Seed 42.

### Audit Findings:
1. **Mathematical Replication:**
   $$\Delta_{\text{Bhagavad Gita}} = 4.3978 - 4.3202 = \mathbf{+0.0776}$$
   The claim is **100% verified and mathematically exact**.
2. **Proper Granularity Distinction:**
   - `wikipedia_sa_bhagavad_gita` (Devenagari Sanskrit) regressed by **+0.0776 CE** (perplexity increased from 75.20 to 81.27).
   - `wikipedia_hi_sanskrit_sahitya` (Hindi essays about Sanskrit literature) improved slightly by **-0.0146 CE** (perplexity decreased from 50.42 to 49.69).
   - It is vital not to conflate Hindi literature with pure Sanskrit verse: the regression is strictly confined to Devanagari Classical Sanskrit text.
3. **Root Cause Analysis:**
   - In `experiments/batch2101_2140/CORPUS_V2_CLEANED_MANIFEST.json`, `wikipedia_sa_bhagavad_gita` is assigned **100% to the test split**.
   - The CPT training split contained English literature (Tagore), Python algorithms (heapq), TCP specifications (RFC 793), Hindi literature (Premchand), and Hindi chemistry, but **zero pure Sanskrit documents**.
   - As the neural weights adjusted to English and Hindi technical corpora over 125 steps, unreinforced Sanskrit syntactic and morphological transition matrices underwent mild representational drift (+0.0776 CE).

---

## 3. Proposal for Controlled Follow-up Curriculum Experiment (Batches 2301+)

To eliminate Sanskrit drift without causing regression in geometry or information theory, we specify the following preregistered protocol:

### A. Data Curation & Partitioning (Leakage-Free)
- Ingest an independent Classical Sanskrit corpus (e.g. *Hitopadesha* or *Meghaduta* excerpts, distinct from *Bhagavad Gita*).
- Partition strictly: 70% Train, 15% Validation, 15% Held-out Test.
- Guarantee 0 document ID and 0 content hash overlap with `wikipedia_sa_bhagavad_gita`.

### B. Curriculum Rebalancing Interventions
- Test 3 candidate mixture ratios during CPT:
  1. **Intervention A (Low Blend):** 10% Sanskrit, 45% English math/tech, 45% Hindi.
  2. **Intervention B (Balanced Blend):** 20% Sanskrit, 40% English math/tech, 40% Hindi.
  3. **Intervention C (Replay Buffer):** Interleaving canonical baseline replay buffers (15%) with the CPT curriculum.

### C. Gating Criteria for Candidate Promotion
- **Sanskrit Gate:** `wikipedia_sa_bhagavad_gita` test CE $\le 4.3202$ (eliminate the +0.0776 regression).
- **Geometry Non-Regression Gate:** `gutenberg_loney_coordinate_geometry` test CE $\le 4.2000$ (retain at least 80% of current geometry gains).
- **Information Theory Gate:** `wikipedia_en_information_theory` test CE $\le 4.5500$.

*Note:* In accordance with Batch 2261–2300 instructions, this experiment is proposed and preregistered only; **no model training was executed in this mission**.
