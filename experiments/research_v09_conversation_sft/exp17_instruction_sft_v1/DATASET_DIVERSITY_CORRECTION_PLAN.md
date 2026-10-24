# ChakrView Exp17 — Dataset Diversity Correction Plan

**Document Status**: Official Architecture & Correction Proposal  
**Target Experiment**: `experiments/research_v09_conversation_sft/exp17_instruction_sft_v1`  
**Referenced Generator**: [`generate_exp17_datasets.py`](file:///D:/Project/ChakrView/experiments/research_v09_conversation_sft/exp17_instruction_sft_v1/generate_exp17_datasets.py)  
**Execution Phase**: Design Phase Only (Zero generator code modified; zero split files generated)  
**Recommendation**: **`READY FOR IMPLEMENTATION (UPON HUMAN AUTHORIZATION)`**  
**Timestamp**: 2026-10-22T23:55:00+05:30  

---

## 1. Executive Summary & Verification of Baseline Candidate

An independent, non-writing, in-memory re-audit of the candidate 400-example dataset produced by [`generate_exp17_datasets.py`](file:///D:/Project/ChakrView/experiments/research_v09_conversation_sft/exp17_instruction_sft_v1/generate_exp17_datasets.py) was conducted using the frozen project tokenizer (`data/experiments/vocab_4096`).

### 1.1 Status of Previously Reported Metrics
- **Clarification on Proposed Targets**: The 50 scientific mechanisms and 8 abstention formulations referenced in [`DATASET_DIVERSITY_REVIEW.md`](file:///D:/Project/ChakrView/experiments/research_v09_conversation_sft/exp17_instruction_sft_v1/DATASET_DIVERSITY_REVIEW.md) were **proposed design targets**, not completed implementations. The active generator still contains only 9 scientific mechanism templates and 3 generic abstention strings.
- **Confirmed Baseline Repetition in Train Split (320 records)**:
  - `short_explanation`: **5 unique responses across 40 records** (Repetition ratio: 8.0×; each response repeats 8 times).
  - `uncertainty_abstention`: **3 unique responses across 40 records** (Repetition ratio: 13.3×; templates repeat 14, 13, and 13 times).
  - `factual_qa`: **10 unique responses across 40 records** (Repetition ratio: 4.0×; each fact repeats 4 times with minor prompt prefixing like `"Could you state: ..."`).
  - `instruction_following`, `context_extraction`, `reference_resolution`, `basic_arithmetic`, `structured_formatting`: **40 unique responses across 40 records** (Repetition ratio: 1.0×; zero response template reuse).

---

## 2. Confirmed Problems vs. Suspected Risks

| Category | Confirmed Structural Problem | Suspected Learning Risk | Severity |
| :--- | :--- | :--- | :---: |
| **`short_explanation`** | Only 5 mechanisms in Train; 2 in Val; 2 in Test. Cyclic repetition across splits. | Model memorizes specific phrase sequences rather than physical reasoning patterns. | **HIGH** |
| **`uncertainty_abstention`**| Every prompt asks for `"ambient temperature"`; exactly 3 generic strings cycle across 40 examples. | Model learns a superficial keyword shortcut (`"ambient temperature" -> abstain`) without checking evidence. | **CRITICAL** |
| **`factual_qa`** | 10 facts repeated 4 times each with superficial prefixes (`"Could you state: ..."`). | Overfitting to 10 specific facts; reduced lexical and conceptual coverage. | **MEDIUM** |
| **`context_extraction`** | 0 response repetition, but contexts share fixed syntactic template (`"Mission roster update: ..."`). | Minor syntax rigidity; grounded extraction remains valid. | **LOW** |
| **`reference_resolution`**| 0 response repetition, but contexts share fixed syntactic template (`"Case file X-YY: ..."`). | Minor syntax rigidity; coreference binding remains valid. | **LOW** |
| **`basic_arithmetic`** | None. Completely unique numbers, operations, and responses. | Arithmetic reasoning remains robust. | **NONE** |
| **`instruction_following`**| None. Completely unique dynamic strings and transformations. | String manipulation remains robust. | **NONE** |
| **`structured_formatting`**| None. Completely unique JSON keys and values. | Schema compliance remains robust. | **NONE** |

---

## 3. Proposed Concrete Example-Bank Improvements

To achieve true conceptual diversity, eliminate keyword shortcuts, and ground all factual assertions without modifying the 400-example budget or schema, we propose expanding the underlying knowledge banks:

### 3.1 Proposed `factual_qa` Bank (40 Train, 5 Val, 5 Test - 50 Unique Facts)
Every record in Train, Val, and Test will feature a unique factual question, distinct answer, and verified source.

#### Representative Examples & Factual Basis:
1. **Astronomy/Planetary**:
   - *Prompt*: `"What is the largest moon of Saturn?"`
   - *Response*: `"Titan is the largest moon of Saturn."`
   - *Verification Basis*: Established astronomical ground truth (Cassini-Huygens mission data).
2. **Geography/Geology**:
   - *Prompt*: `"What is the longest river in South America?"`
   - *Response*: `"The Amazon River is the longest river in South America."`
   - *Verification Basis*: Standard physical geography consensus.
3. **Chemistry/Physics**:
   - *Prompt*: `"What is the most abundant gas in Jupiter's atmosphere?"`
   - *Response*: `"Hydrogen is the most abundant gas in Jupiter's atmosphere."`
   - *Verification Basis*: Standard planetary atmospheric spectroscopy.
4. **Biology/Physiology**:
   - *Prompt*: `"What type of blood cells carry oxygen throughout the human body?"`
   - *Response*: `"Red blood cells carry oxygen throughout the human body."`
   - *Verification Basis*: Fundamental biological fact (erythrocyte function).

---

### 3.2 Proposed `short_explanation` Bank (50 Unique Physical Mechanisms)
Replaces the 5 repeated mechanisms with 50 distinct, verified physical, chemical, biological, and astronomical mechanisms (40 Train, 5 Val, 5 Test) at a beginner-accessible level.

#### Representative Examples with Scientific Mechanisms:
1. **Rayleigh Scattering (Optics)**:
   - *Prompt*: `"Why does the daytime sky appear blue?"`
   - *Response*: `"Gases in the atmosphere scatter shorter blue wavelengths of sunlight more than longer red wavelengths."`
   - *Verification Note*: Rayleigh scattering formula ($\propto 1/\lambda^4$).
2. **Phase Change & Latent Heat (Thermodynamics)**:
   - *Prompt*: `"Why does sweating cool the human body?"`
   - *Response*: `"Evaporating sweat absorbs latent heat from the skin, transferring thermal energy away from the body."`
   - *Verification Note*: Latent heat of vaporization principle.
3. **Atmospheric Pressure & Sound (Acoustics)**:
   - *Prompt*: `"Why does sound travel faster through water than through air?"`
   - *Response*: `"Water molecules are packed much closer together than air molecules, transmitting acoustic vibrations more rapidly."`
   - *Verification Note*: Elastic bulk modulus and acoustic wave propagation.
4. **Combustion & Oxidation (Chemistry)**:
   - *Prompt*: `"Why does iron rust when exposed to moist air?"`
   - *Response*: `"Iron atoms react with oxygen and water to form hydrated iron oxide, commonly known as rust."`
   - *Verification Note*: Electrochemical corrosion equation $4\text{Fe} + 3\text{O}_2 + 6\text{H}_2\text{O} \rightarrow 4\text{Fe(OH)}_3$.
5. **Gravitational Tides (Planetary Science)**:
   - *Prompt*: `"Why do ocean tides occur on Earth?"`
   - *Response*: `"The gravitational pull of the Moon creates tidal bulges in Earth's oceans as the planet rotates."`
   - *Verification Note*: Differential gravitational field theory.

---

### 3.3 Proposed `uncertainty_abstention` Bank (Multi-Domain Evidence Calibration)
Replaces the single `"ambient temperature at station"` template with 6 distinct real-world documentation scenarios across multiple domains, pairing **unsupported questions** with **answerable control queries** to prevent blanket abstention.

#### Scenario Matrix & Control Calibration:

| Context Domain | Provided Context Summary | Unsupported Query (Requires Abstention) | Answerable Control Query (Tests Grounding) | Calibrated Abstention Response |
| :--- | :--- | :--- | :--- | :--- |
| **Server Telemetry** | Server ID, CPU load 42%, uptime 18 days. | *"What is the network bandwidth consumption?"* | *"What is the CPU load of the server?"* | *"The provided telemetry log does not state network bandwidth."* |
| **Expedition Log** | Team Alpha reached base camp at 14:00 with 4 members. | *"What was the altitude of base camp?"* | *"At what time did Team Alpha reach base camp?"* | *"Altitude is not mentioned in the expedition log."* |
| **Engine Schematic** | Model-7 features titanium valves and electric ignition. | *"What is the fuel tank capacity of Model-7?"* | *"What material is used for the valves in Model-7?"* | *"Fuel tank capacity is omitted from the specification."* |
| **Meeting Minutes** | Design review approved milestone 2; next sync Thursday. | *"What is the allocated budget for milestone 2?"* | *"Which milestone was approved in the design review?"* | *"The meeting summary does not specify the allocated budget."* |
| **Shipping Manifest**| Cargo vessel carrying 400 containers of dry grain. | *"What is the name of the vessel's captain?"* | *"What cargo is the vessel carrying?"* | *"The captain's name is not recorded in the manifest."* |
| **Weather Station** | Barometric pressure 1013 hPa; wind speed 12 knots north. | *"What was the rainfall volume recorded today?"* | *"What is the recorded barometric pressure?"* | *"Rainfall volume is not provided in the weather report."* |

---

## 4. Split Allocation & Cross-Split Leakage Safeguards

To prevent subtle conceptual or structural leakage across splits:
1. **Complete Mechanism Isolation**:
   - Mechanisms assigned to Train (e.g. Rayleigh scattering, sweating) must **never** appear in Val or Test.
   - Val mechanisms (e.g. Doppler effect, thermal conductivity of metals) are strictly reserved.
   - Test mechanisms (e.g. atmospheric scintillation, autumn leaf pigment breakdown) are completely isolated.
2. **Context Entity Segregation**:
   - Context extraction mission cycles remain indexed by split (`TRAIN-10`, `VAL-10`, `TEST-10`).
   - Coreference resolution case files remain indexed by split (`TRAIN-00`, `VAL-00`, `TEST-00`).
   - Arithmetic numbers use distinct offset domains (Train: 100–260; Val: 300–320; Test: 500–520).
3. **Exact and Semantic Disjointness**:
   - Cross-split exact prompt overlap = **0**.
   - Cross-split exact response overlap = **0** (except for generic calibrated abstentions).
   - Normal Chat Gate contamination: 0 entity hits; all Levenshtein ratios $< 0.70$.

---

## 5. Measurable Diversity Acceptance Criteria

Before the modified generator can be executed in official generation mode (`--write-splits`), it must satisfy the following automated and human-reviewed acceptance criteria:

```
[CRITERION-01] Split Quotas: Exactly 320 Train, 40 Val, 40 Test records. (Automated)
[CRITERION-02] Category Balance: Exactly 40 Train, 5 Val, 5 Test records per category. (Automated)
[CRITERION-03] Scientific Explanation Repetition Ratio = 1.0x (40 unique mechanisms in Train). (Automated)
[CRITERION-04] Factual QA Repetition Ratio = 1.0x (40 unique factual questions in Train). (Automated)
[CRITERION-05] Abstention Context Diversity: >= 6 distinct contextual domains. (Automated)
[CRITERION-06] Abstention Max Response Frequency <= 6 in Train (down from 14). (Automated)
[CRITERION-07] Response Length Hard Bounds: 100% within [3, 45] tokens. (Automated)
[CRITERION-08] Response Length Provisional Target: >= 90.0% within [5, 35] tokens. (Automated)
[CRITERION-09] Exact Cross-Split Prompt Overlap = 0. (Automated)
[CRITERION-10] Gate v0.1 Contamination: 0 entity hits; all Levenshtein ratios < 0.70. (Automated)
[CRITERION-11] Factual Correctness Verification: 100% of facts verified against ground truth. (Human Review)
[CRITERION-12] Scientific Validity: All 50 mechanisms physically sound at beginner level. (Human Review)
```

---

## 6. Expected Effects on Model Learning & Known Limitations

### 6.1 Expected Benefits
- **Elimination of Keyword Shortcuts**: By removing the `"ambient temperature"` trigger, the 13.5M parameter ChakrMicro model is forced to evaluate context tokens before emitting an abstention token.
- **Improved Relational Generalization**: Training on 40 distinct scientific mechanisms and 40 factual QA pairs teaches generalized instruction adherence rather than sequence memorization.
- **Preserved Format Stability**: Standardized system prompt serializations (`Instruction: ...\nResponse:`) and JSON formatting remain cleanly conditioned.

### 6.2 Known Architectural Limitations
- **Model Capacity Constraint**: At 13.5 million parameters and a 4,096-token vocabulary, the model cannot function as an encyclopedic knowledge base. Fine-tuning on 50 scientific facts serves to calibrate instruction-following style and concise factual formatting, not to endow broad world knowledge.
- **Surface Filter Boundary**: Automated sequence matching guarantees lexical disjointness, but cannot prove complete conceptual decoupling without human oversight.

---

## 7. Protected System Integrity & Verification

All canonical ChakrView assets and frozen artifacts were verified via SHA-256 before and after compiling this plan:

| Asset | Path | Expected SHA-256 | Current SHA-256 | Verification |
| :--- | :--- | :--- | :--- | :---: |
| **Canonical Checkpoint** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **MATCH** |
| **Tokenizer Vocab** | `data/experiments/vocab_4096/vocab.json` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | **MATCH** |
| **Tokenizer Merges** | `data/experiments/vocab_4096/merges.json` | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | **MATCH** |
| **Runtime Session** | `chakrview/runtime/session.py` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **MATCH** |
| **Brain Model** | `chakrview/brain/model.py` | `1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361` | `1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361` | **MATCH** |

- **Official Filesystem**: Confirmed `experiments/research_v09_conversation_sft/exp17_instruction_sft_v1/data/` does not exist (0 official files written).
- **Generator Source**: [`generate_exp17_datasets.py`](file:///D:/Project/ChakrView/experiments/research_v09_conversation_sft/exp17_instruction_sft_v1/generate_exp17_datasets.py) was **not modified**.
- **Working Tree**: Preserved; no Git commit, checkout, stash, or reset was performed.

---

## 8. Final Recommendation & Stop Condition

**Recommendation**: **`READY FOR IMPLEMENTATION (UPON HUMAN AUTHORIZATION)`**

The design plan resolves the template repetition and keyword shortcut risks while strictly maintaining the 400-example budget, schema constraints, and safety boundaries. 

Execution is halted. We await explicit human authorization before updating the knowledge banks in [`generate_exp17_datasets.py`](file:///D:/Project/ChakrView/experiments/research_v09_conversation_sft/exp17_instruction_sft_v1/generate_exp17_datasets.py).
