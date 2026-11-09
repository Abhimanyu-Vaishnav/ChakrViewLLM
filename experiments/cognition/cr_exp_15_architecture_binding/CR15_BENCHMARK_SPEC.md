# CR-EXP-15 Phase 3: Benchmark Specification

## 1. Overview & Objectives
The CR-EXP-15 benchmark provides an independent, decomposed 1,200-probe test suite explicitly structured to isolate where working memory fails in ChakrMicro.

- **Total Probes:** Exactly 1,200 probes
- **Format:** JSON structure containing prompt, category, subject, expected target, earlier value (stale), and distractor values.
- **Sealed File:** `benchmark_data/benchmark_1200_sealed.json`
- **SHA-256 Hash:** `71212797975022ad4382d4d660d08125782ae473b85a5081eb2f12d7da5d28d8`
- **Frozen Status:** Immutable; locked prior to training interventions or model evaluations.

---

## 2. Category Distribution & Denominators

| Category | Identifier Prefix | Probes | Target Operation |
| :--- | :---: | :---: | :--- |
| **1. Entity Copying** | `CR15_COPY` | 150 | Unconditional subword reproduction of novel names |
| **2. Entity Selection** | `CR15_SEL` | 150 | Role-based selection between two presented entities |
| **3. Static Binding** | `CR15_STAT` | 150 | Single-turn attribute binding without state update |
| **4. Single-Entity Updates** | `CR15_SEUP` | 150 | Sequential state update for 1 entity (recency overwrite) |
| **5. Multi-Entity Tracking** | `CR15_MEST` | 150 | Tracking 2 entities where 1 is updated and other queried |
| **6. Interleaved Updates** | `CR15_ILUP` | 150 | Interleaved updates to both entities; querying target |
| **7. Delayed Retrieval** | `CR15_DRDIST` | 150 | State updates with 2 intervening conversational distractor turns |
| **8. Calibrated Abstention** | `CR15_ABST` | 150 | Unanswerable queries on unmentioned entities |
| **Total Benchmark** | — | **1,200** | — |

---

## 3. Disjoint Entity & Location Guarantee
- **Test Entities ($N=40$):** `Apurva`, `Badri`, `Bhargav`, `Chaitanya`, `Devendra`, `Digvijay`, `Eshwar`, `Fateh`, `Gautam`, `Govind`, `Harish`, `Himanshu`, `Indrajit`, `Jaideep`, `Jawahar`, `Kalyan`, `Kuldeep`, `Lalit`, `Madhav`, `Manvendra`, `Mukesh`, `Naveen`, `Niraj`, `Padma`, `Pankhuri`, `Pranav`, `Raghav`, `Rajnath`, `Ranveer`, `Sanjay`, `Sarvesh`, `Shashank`, `Siddharth`, `Somesh`, `Tarun`, `Umesh`, `Vaibhav`, `Varun`, `Vikas`, `Yashwant`.
- **Test Locations ($N=25$):** `Alwar`, `Bareilly`, `Belgaum`, `Calicut`, `Durgapur`, `Faridabad`, `Gorakhpur`, `Haridwar`, `Imphal`, `Jamnagar`, `Kakinada`, `Kollam`, `Malda`, `Muzaffarpur`, `Nellore`, `Pathankot`, `Rajahmundry`, `Rourkela`, `Saharanpur`, `Thanjavur`, `Thrissur`, `Udaipur`, `Ujjain`, `Vellore`, `Warangal`.
- **Zero Leakage:** All 40 test entities and 25 test locations are strictly barred from training sets.

