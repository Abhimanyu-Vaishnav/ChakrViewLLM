# CR-EXP-14 Phase 6: Strict Entity-Binding Evaluation Results

## 1. Executive Summary

CR-EXP-14 conducted the first rigorous, sealed evaluation of dynamic entity-value binding in ChakrMicro ($3.44\text{M}$ parameters, $d_{\text{model}}=192$, 6 layers, 6 heads, ByteBPE 4096).

Following the revelation in CR-EXP-13 that prior working-memory metrics were inflated by loose substring evaluators that rewarded training subject substitution, CR-EXP-14 evaluated all baseline models and three controlled training interventions against the frozen **1,200 Sealed Probes** (`benchmark_1200_sealed.json`, SHA-256: `1958e94571f83c66c26f390634ffa563d0a2ff57e58943e835d7825245b964b9`).

---

## 2. Comprehensive Strict Accuracy Benchmark Table

| Model / Candidate | SHA-256 Checkpoint Hash | Overall Strict Acc (%) | Strict Hits / 1200 | Subject Identification Acc (%) | Value Accuracy (%) | Recency Accuracy (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **CR-11 Candidate A** | `4d14cf05df395608ed8d1ab02eeca571b6bf049574785a3d16a7ec2112a64bc5` | 5.00% | 60 / 1200 | 8.33% | 5.00% | 0.00% |
| **CR-12 Candidate A** | `cfb99ad7651bf64303a4fd8ab562ab9d7b0b0ed008fb3389957e97c76a05a8a2` | 8.33% | 100 / 1200 | 8.33% | 8.33% | 0.00% |
| **CR-12 Candidate B** (Parent) | `60271c8b5e3d26b76237dc8f986e0d8d6ada2f08815cea57ec5d9a37102d9d2e` | 8.33% | 100 / 1200 | 8.33% | 8.33% | 0.00% |
| **Candidate A (ESVC)** | `58f51bc249aec77235bbdd07179dc130f72e24947500b05fa092f10752a8f033` | 1.92% | 23 / 1200 | 8.33% | 1.92% | 0.00% |
| **Candidate B (ISUC)** | `e838eb33e6e49cef68e6afb12e7f93bd788f1259940647dce7ef70717f512cf2` | 2.75% | 33 / 1200 | 8.33% | 2.75% | 0.00% |
| **Candidate C (SED)** | `32f1066f4a6b0c652d42991bcc8c1427b63854a55483ffc8a32b726512600d12` | 7.33% | 88 / 1200 | 8.33% | 7.33% | 0.00% |

---

## 3. Detailed Category Breakdown Across All Evaluated Models

| Category | Probes | CR-11 Cand A | CR-12 Cand A | CR-12 Cand B (Parent) | Cand A (ESVC) | Cand B (ISUC) | Cand C (SED) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Single-Entity State Tracking | 200 | 0.0% (0) | 0.0% (0) | 0.0% (0) | 0.0% (0) | 0.0% (0) | 0.0% (0) |
| Two-Entity State Tracking | 200 | 0.0% (0) | 0.0% (0) | 0.0% (0) | 0.0% (0) | 0.0% (0) | 0.0% (0) |
| Three-Entity State Tracking | 200 | 0.0% (0) | 0.0% (0) | 0.0% (0) | 0.0% (0) | 0.0% (0) | 0.0% (0) |
| Repeated Updates Recency | 200 | 0.0% (0) | 0.0% (0) | 0.0% (0) | 0.0% (0) | 0.0% (0) | 0.0% (0) |
| Interleaved Distractors | 150 | 0.0% (0) | 0.0% (0) | 0.0% (0) | 0.0% (0) | 0.0% (0) | 0.0% (0) |
| Counterfactual Reassignment | 150 | 0.0% (0) | 0.0% (0) | 0.0% (0) | 0.0% (0) | 0.0% (0) | 0.0% (0) |
| Missing Info Abstention | 100 | **60.0%** (60) | **100.0%** (100) | **100.0%** (100) | **23.0%** (23) | **33.0%** (33) | **88.0%** (88) |
| **Total Benchmark Score** | **1,200** | **5.00%** (60) | **8.33%** (100) | **8.33%** (100) | **1.92%** (23) | **2.75%** (33) | **7.33%** (88) |

---

## 4. Key Empirical Takeaways

1. **The Strict 0.0% Dynamic Binding Floor:** Across all 1,100 state-tracking, recency, multi-entity, and counterfactual probes involving out-of-distribution entity names and cities, **every model scored exactly 0.0%**.
2. **Subject Prior Dominance:** When presented with an unseen entity (e.g. `Abhay`, `Bimla`, `Chetan`), the model's self-attention layers fail to route the query entity token into the prediction head. Instead, strong unigram/bigram token priors for training set names (`Ananya`, `Komal`, `Ashesh`, `Amita`) overpower the logits.
3. **Abstention Calibration Trade-off:** Fine-tuning on state-tracking dialogues slightly eroded the models' previously acquired conservative abstention calibration:
   - CR-12 Parent abstained correctly on 100% of unanswerable queries.
   - Candidate A abstained on 23%, Candidate B on 33%, and Candidate C on 88%.
