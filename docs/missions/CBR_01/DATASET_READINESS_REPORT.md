# CBR-01 — WORKSTREAM B: DATASET READINESS & PROVENANCE REPORT

---

## 1. Inventory of Available SFT Datasets

| Dataset Identifier | Path | Total Records | Unique Responses | Response Diversity | Audit Status | Primary Failure Mode |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **M3 Conversational** | `data/sft/m3_conversational/` | 600 | 125 | **20.8%** | **REJECTED** | Severe response collapse (5 responses repeated 8x across 11 categories) |
| **Exp34 Candidate v2** | `data/sft/exp34_candidate_v2/` | 1,000 | 820 | **82.0%** | **SUPERSEDED** | Partial template repetition |
| **Exp34 Candidate v3** | `data/sft/exp34_candidate_v3/` | 1,000 | 1,000 | **100.0%** | **APPROVED** | Verified diverse synthetic instruction set |
| **Exp35 DPCE** | `experiments/exp35_dpce/data/` | 1,000 | 1,000 | **100.0%** | **APPROVED** | Disjoint prompt-conditioned extraction with entity variations |

---

## 2. Quality Analysis & Diagnostic Deficits

1. **Why M3 Failed:**
   - Forensic analysis proved that M3's 20.8% response diversity forced the model to memorize fixed answers (`"Linux"`, `"Paris"`, `"nucleus"`) rather than learning generalized syntax or contextual conditioning.
2. **Why Exp35 Single-Step Extraction Failed to Generalize:**
   - Exp35 provided 100% unique entity responses, but 78.96% of supervised tokens were syntactic filler (e.g., `"The contact person for station 101 is..."`). The model lowered template loss to 1.38 while entity token loss remained at 2.44, failing to generalize to conversational prompts outside the exact template.
3. **The Solution for CBR-01 (Diverse Algorithmic & Conversational Induction):**
   - Combine diverse conversational instructions (from `Exp34 v3` and balanced multi-turn scenarios) with focused **algorithmic induction spans** (short variable lookups and prompt-entity copies) where dynamic entities receive high-weight loss.
   - Maintain strict disjoint train/validation/test partitions.
