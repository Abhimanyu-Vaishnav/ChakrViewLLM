# ChakrView Exp17 — Dataset Diversity Review and Correction Plan

**Document Status**: Official Review & Correction Architecture Plan  
**Target Experiment**: `experiments/research_v09_conversation_sft/exp17_instruction_sft_v1`  
**Referenced Generator**: [`generate_exp17_datasets.py`](file:///D:/Project/ChakrView/experiments/research_v09_conversation_sft/exp17_instruction_sft_v1/generate_exp17_datasets.py)  
**Execution Context**: Pre-Generation Quality Review (Zero official split files generated)  
**Timestamp**: 2026-10-22T23:50:00+05:30  

---

## 1. Executive Summary

This report delivers a thorough diversity review of the candidate 400-example dataset synthesized in-memory by [`generate_exp17_datasets.py`](file:///D:/Project/ChakrView/experiments/research_v09_conversation_sft/exp17_instruction_sft_v1/generate_exp17_datasets.py). 

While the candidate dataset successfully achieves 100% schema compliance, exact serialization parity with the inference engine, 0% cross-split prompt leakage, and strict adherence to token-length bounds, an audit of template repetition revealed significant repetition in specific categories. In `short_explanation`, only 5 scientific mechanisms exist in Train (repeating 8 times each), and in `uncertainty_abstention`, only 3 response templates exist in Train (repeating 13–14 times each).

This review:
1. Measures template repetition across all 8 categories and splits.
2. Evaluates the tension between learning conversational consistency versus shortcut memorization.
3. Proposes an expanded, fully grounded knowledge bank of 40 distinct scientific mechanisms and 40 structured abstention scenarios.
4. Analyzes cross-split conceptual and semantic leakage.
5. Defines explicit, measurable diversity acceptance criteria required before official dataset generation can be authorized.

---

## 2. Repeated Response Templates: Measurement & Frequencies

The candidate 400-record dataset was analyzed for response template reuse across all 8 functional categories:

| Category | Split | Total Records | Unique Responses | Repetition Ratio | Most Repeated Template Count |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`factual_qa`** | Train | 40 | 10 | 4.0× | 4× (all 10 facts repeat 4 times) |
|  | Val | 5 | 5 | 1.0× | 1× (disjoint) |
|  | Test | 5 | 5 | 1.0× | 1× (disjoint) |
| **`instruction_following`**| Train | 40 | 40 | 1.0× | 1× (unique dynamic inputs) |
|  | Val | 5 | 5 | 1.0× | 1× (unique dynamic inputs) |
|  | Test | 5 | 5 | 1.0× | 1× (unique dynamic inputs) |
| **`context_extraction`** | Train | 40 | 40 | 1.0× | 1× (unique dynamic entities) |
|  | Val | 5 | 5 | 1.0× | 1× (unique dynamic entities) |
|  | Test | 5 | 5 | 1.0× | 1× (unique dynamic entities) |
| **`reference_resolution`**| Train | 40 | 40 | 1.0× | 1× (unique indexed cases) |
|  | Val | 5 | 5 | 1.0× | 1× (unique indexed cases) |
|  | Test | 5 | 5 | 1.0× | 1× (unique indexed cases) |
| **`short_explanation`** | **Train** | **40** | **5** | **8.0×** | **8× (all 5 mechanisms repeat 8 times)** |
|  | Val | 5 | 2 | 2.5× | 3× (Mechanism A repeats 3×, B repeats 2×) |
|  | Test | 5 | 2 | 2.5× | 3× (Mechanism C repeats 3×, D repeats 2×) |
| **`basic_arithmetic`** | Train | 40 | 40 | 1.0× | 1× (unique numerical pairs) |
|  | Val | 5 | 5 | 1.0× | 1× (unique numerical pairs) |
|  | Test | 5 | 5 | 1.0× | 1× (unique numerical pairs) |
| **`uncertainty_abstention`**| **Train** | **40** | **3** | **13.3×** | **14× (Template 1 repeats 14×)** |
|  | Val | 5 | 3 | 1.7× | 2× (Template 1 & 2 repeat 2×) |
|  | Test | 5 | 3 | 1.7× | 2× (Template 1 & 2 repeat 2×) |
| **`structured_formatting`**| Train | 40 | 40 | 1.0× | 1× (unique JSON key-values) |
|  | Val | 5 | 5 | 1.0× | 1× (unique JSON key-values) |
|  | Test | 5 | 5 | 1.0× | 1× (unique JSON key-values) |

---

## 3. Detailed Audit of Repeated Templates

### 3.1 The Five Repeated Scientific Mechanisms in `short_explanation` (Train)
The current Train split cycles through exactly five scientific mechanisms:
1. **Ocean Blue Light Absorption** (Repeats 8×: `EXP17_TRAIN_SHORT_EXPLANATION_000`, `005`, `010`, `015`, `020`, `025`, `030`, `035`)
   - *Prompt*: `"Why does the ocean appear blue?"` (or `"Briefly explain (train-N): Why does the ocean appear blue?"`)
   - *Response*: `"Water molecules absorb red light more strongly than blue light, leaving blue light reflected."`
2. **Gravitational Attraction** (Repeats 8×: `EXP17_TRAIN_SHORT_EXPLANATION_001`, `006`, `011`, `016`, `021`, `026`, `031`, `036`)
   - *Prompt*: `"Why do objects fall when dropped on Earth?"`
   - *Response*: `"Earth's gravitational force pulls objects downward toward its center of mass."`
3. **Axial Tilt & Seasons** (Repeats 8×: `EXP17_TRAIN_SHORT_EXPLANATION_002`, `007`, `012`, `017`, `022`, `027`, `032`, `037`)
   - *Prompt*: `"Why do we experience seasons on Earth?"`
   - *Response*: `"Earth's tilted rotational axis causes different hemispheres to receive varying sunlight throughout the year."`
4. **Ice Density Anomaly** (Repeats 8×: `EXP17_TRAIN_SHORT_EXPLANATION_003`, `008`, `013`, `018`, `023`, `028`, `033`, `038`)
   - *Prompt*: `"Why does ice float on liquid water?"`
   - *Response*: `"Water expands as it freezes, resulting in ice having lower density than liquid water."`
5. **Seasonal Bird Migration** (Repeats 8×: `EXP17_TRAIN_SHORT_EXPLANATION_004`, `009`, `014`, `019`, `024`, `029`, `034`, `039`)
   - *Prompt*: `"Why do birds migrate before winter?"`
   - *Response*: `"Birds migrate to regions with warmer temperatures and abundant food sources."`

### 3.2 The Three Repeated Abstention Templates in `uncertainty_abstention`
Across all 40 Train examples, 5 Val examples, and 5 Test examples, the context consistently describes telemetry at an indexed station (`"System log: Subsystem telemetry nominal at station {split}_{idx}. Battery status is green."`) and asks for `"What was the ambient temperature at station {split}_{idx}?"`. 

The response cycles deterministically through only three fixed strings:
1. **Template 1** (14× in Train; 2× in Val; 2× in Test):
   - *Sample IDs*: `EXP17_TRAIN_UNCERTAINTY_ABSTENTION_000`, `003`, `006`...
   - *Response*: `"The provided context does not contain this information."`
2. **Template 2** (13× in Train; 2× in Val; 2× in Test):
   - *Sample IDs*: `EXP17_TRAIN_UNCERTAINTY_ABSTENTION_001`, `004`, `007`...
   - *Response*: `"This information cannot be determined from the available text."`
3. **Template 3** (13× in Train; 1× in Val; 1× in Test):
   - *Sample IDs*: `EXP17_TRAIN_UNCERTAINTY_ABSTENTION_002`, `005`, `008`...
   - *Response*: `"Information not provided in the context."`

---

## 4. Assessment: Consistency Training vs. Shortcut Learning Risks

### 4.1 Positive Value of Controlled Repetition
In small-scale instruction fine-tuning (SFT on 400 examples with a 13.5M parameter model):
- Repeating structured output formulations (e.g. standard abstention phrases) provides a clean inductive bias for the model to emit a confident stop signal (`<eos>`) rather than hallucinating when relevant context is absent.
- Controlled repetition prevents token dispersion during initial format adaptation.

### 4.2 Acute Shortcut-Learning Risks
However, high density repetition across a small bank of 5 mechanisms presents significant failure modes:
1. **Weight Memorization Over Generalization**: With only 5 explanation targets repeated 8 times each, gradient updates heavily penalize deviations from those exact 5 strings. The model learns to recall specific n-gram sequences rather than learning the generalized instruction-following behavior of "explain a physical mechanism concisely."
2. **Context-Free Abstention Triggering**: Because the 40 abstention examples share nearly identical prompt syntax (*"What was the ambient temperature at station..."*), the model risks associating the token `"ambient temperature"` directly with the abstention reflex, rather than checking whether the requested entity is actually missing from the context.
3. **Loss of Lexical Diversity**: Training on identical target responses causes probability mass to collapse onto fixed introductory clauses (*"Water molecules absorb..."*, *"The provided context does not..."*).

---

## 5. Proposed Expanded Knowledge & Abstention Architecture

To mitigate shortcut learning without violating the 400-example budget or schema constraints, we propose expanding both banks so that **every single example in Train (40), Val (5), and Test (5) uses a unique, factually grounded concept**:

### 5.1 Expanded Scientific Explanation Bank (50 Unique Mechanisms)
A diverse library of physical, biological, chemical, and astronomical phenomena:
- **Train (40 unique facts)**:
  - Optics/Atmosphere: Sky blue scattering (Rayleigh), rainbows (refraction & internal reflection), mirages (temperature inversion refraction), sunset red wavelengths.
  - Thermodynamics/Heat: Evaporative cooling (latent heat of vaporization), thermal expansion of metals, greenhouse effect (infrared absorption), why steam burns more than boiling water.
  - Classical Physics: Action-reaction in rocket propulsion, sound travel speed in solids vs gases, centrifugal inertia in clothes spinners, why boats float (Archimedes principle).
  - Chemistry: Rust formation (iron oxidation), soap cleaning mechanism (amphiphilic micelle formation), yeast rising in bread ($CO_2$ production), acid-base neutralization.
  - Planetary Science: Lunar tides (gravitational differential), auroras (solar wind particle collision with ionosphere), craters on the Moon (lack of weathering atmosphere).
  - Biology: Sweating for thermoregulation, pupil dilation in dim light, deciduous tree leaf drop (water conservation), shivering generating muscle heat.
- **Val (5 unique facts)**: Clouds producing rain, thermal conductivity of metal spoons, Doppler effect of approaching sirens, echo formation off canyon walls, static electricity on dry days.
- **Test (5 unique facts)**: Scintillation of starlight in turbulent atmosphere, autumn carotenoid unmasking, why lightning precedes thunder, cavitation bubbles in propellers, why salt melts road ice.

### 5.2 Expanded Abstention Bank: Evidence Calibration & Multi-Domain Scenarios
Instead of asking only for "temperature" at a "station", abstention scenarios should vary context types, query attributes, and reasons for uncertainty:

| Scenario Type | Context Domain | Query Target | Missing Reason | Sample Phrasing |
| :--- | :--- | :--- | :--- | :--- |
| **System Logs** | Server telemetry | Memory bandwidth | Metric omitted from log | *"The server log does not state memory bandwidth."* |
| **Personnel Records** | Expedition manifests | Pilot's age | Personal details unrecorded | *"Age is not mentioned in the expedition manifest."* |
| **Historical Briefs** | Treaty documentation | Signing location | Geographical detail missing | *"The document does not specify where the treaty was signed."* |
| **Technical Specs** | Engine blueprint | Fuel capacity | Specification absent | *"Fuel capacity is omitted from the provided blueprint."* |
| **Meeting Notes** | Project timeline | Project budget | Financial figure not discussed | *"The meeting summary contains no information about project budget."* |
| **Clinical Trial** | Dosage log | Patient recovery date | Outcome metric unrecorded | *"The record does not indicate the patient's recovery date."* |

---

## 6. Cross-Split Conceptual and Semantic Leakage Analysis

Automated lexical checks confirmed 0% exact prompt overlap. However, conceptual auditing reveals potential semantic overlap:

### 6.1 Conceptual Leakage Vectors Identified
1. **Parallel Factual Queries**:
   - Train asks: `"What is the capital of Japan?"` (`Tokyo`)
   - Val asks: `"What is the capital of Canada?"` (`Ottawa`)
   - Test asks: `"What is the capital of Australia?"` (`Canberra`)
   *Analysis*: While prompts and answers are strictly disjoint, the *underlying relation* (`country -> capital`) is identical. This is acceptable for evaluating in-domain relational transfer, provided specific country-capital pairs never cross splits.
2. **Elementary Astronomy Queries**:
   - Train asks: `"Which planet is closest to the Sun?"` (`Mercury`)
   - Test asks: `"Which planet is famous for its prominent ring system?"` (`Saturn`)
   *Analysis*: Acceptable domain sharing; questions evaluate distinct physical characteristics.
3. **Abstention Identical Targets**:
   - In the candidate dataset, Train, Val, and Test all ask for `"ambient temperature"` against different station IDs.
   *Risk*: This constitutes severe conceptual leakage, as the model could simply memorize that "ambient temperature" triggers an abstention response. The proposed expansion eliminates this completely.

---

## 7. Measurable Diversity Acceptance Criteria

Before modifying [`generate_exp17_datasets.py`](file:///D:/Project/ChakrView/experiments/research_v09_conversation_sft/exp17_instruction_sft_v1/generate_exp17_datasets.py) or writing official dataset files, the new generation logic must satisfy the following measurable criteria:

1. **Zero Repetition in Scientific Explanations**:
   - Train `short_explanation`: Exactly 40 unique responses across 40 records (Repetition Ratio = 1.0×).
   - Val `short_explanation`: Exactly 5 unique responses across 5 records.
   - Test `short_explanation`: Exactly 5 unique responses across 5 records.
2. **Zero Prompt/Context Reuse in Abstention**:
   - Train `uncertainty_abstention`: 40 distinct missing-information queries spanning at least 6 distinct semantic domains (logs, history, biology, finance, manufacturing, transport).
   - Contexts must never reuse the `"ambient temperature at station"` template.
3. **Controlled Abstention Phrasing Diversity**:
   - Target responses must utilize at least 8 distinct, well-calibrated abstention formulations rather than 3.
   - Maximum reuse of any single abstention sentence in Train $\le 6$ (down from 14).
4. **Token Distribution Stability**:
   - Hard bounds $[3, 45]$ tokens strictly enforced (100% compliance).
   - Provisional target $[5, 35]$ tokens maintained at $\ge 90.0\%$.
   - Global response token mean remains within $12.0 - 24.0$ tokens.
5. **Exact & Surface Disjointness**:
   - Cross-split exact prompt overlap = 0.
   - Cross-split exact response overlap $\le 5$ (generic abstention phrases only).
   - Normal Chat Gate contamination: 0 entity hits; all Levenshtein ratios $< 0.70$.

---

## 8. Separation of Automated Checks vs. Human Review Items

To maintain audit integrity, responsibilities are cleanly bifurcated:

| Audit Aspect | Automated Enforceable Check | Requires Explicit Human Review |
| :--- | :--- | :--- |
| **Quotas & Splits** | Schema, counts (320/40/40), ID uniqueness | Allocation balance across capability areas |
| **Correctness** | Arithmetic evaluation, JSON parsing, regex | Scientific mechanism validity, factual accuracy |
| **Leakage** | Exact string match, Levenshtein ratio $< 0.70$ | Conceptual independence across train/val/test |
| **Token Length** | Tokenizer count programmatically bounded $[3, 45]$ | Acceptability of concise 4-token outputs |
| **Dataset Generation**| Deterministic seed matching, dry-run in-memory | Formal authorization of `--write-splits` flag |

---

## 9. Protected Asset Integrity

Prior to and following the compilation of this review, all protected workspace assets were verified via SHA-256:

| Asset | Path | Established SHA-256 | Current SHA-256 | Verification |
| :--- | :--- | :--- | :--- | :---: |
| **Canonical Checkpoint** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **MATCH** |
| **Tokenizer Vocab** | `data/experiments/vocab_4096/vocab.json` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | **MATCH** |
| **Tokenizer Merges** | `data/experiments/vocab_4096/merges.json` | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | **MATCH** |
| **Runtime Session** | `chakrview/runtime/session.py` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **MATCH** |
| **Brain Model** | `chakrview/brain/model.py` | `1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361` | `1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361` | **MATCH** |

- **Official Splits**: Verified that `experiments/research_v09_conversation_sft/exp17_instruction_sft_v1/data/` does not exist (0 files written).
- **Working Tree**: Preserved without destructive checkout, clean, reset, or stash actions.

---

## 10. Next Steps & Stop Condition

No generator source modifications have been applied, and no official files have been written. 

**Execution is halted** to allow the human reviewer to inspect the proposed diversity expansion architecture and provide explicit authorization before proceeding with generator implementation.
