# Root-Cause Hypotheses & Evidence Matrix

**Target Experiment**: `exp19_debiased_sft_cpu_600_20261024_125700`  
**Diagnosis Directory**: `experiments/overnight_runs/exp19_output_diagnosis_20261024_131331`  

---

## 1. Root-Cause Hypothesis Table

| Hypothesis | Supporting Evidence | Contradicting Evidence | Confidence | Lowest-Cost Discriminating Test | Testable Without Training? |
| :--- | :--- | :--- | :---: | :--- | :---: |
| **H1: Task Interference Across 8 Heterogeneous Categories** | Model mixes concepts (e.g. producing `{"style=` in `instruction_following`, or `"The difference"` in `factual_qa`). Training set contains only 150 examples per category, diluting task-specific signal across 8 radically different domains. | All 8 categories were formatted with the identical `Instruction: ...\nResponse:` template. | **HIGH** | Train a single-task ablation pilot (e.g. 500 steps exclusively on `factual_qa` or `basic_arithmetic`). | No (requires 1 isolated training pilot) |
| **H2: Pre-Training Code/Attribute Basin Domination in Structured Tasks** | In `structured_formatting`, 100% of generations (20/20) emit CSS/dict fragments (`{"style={"style="}`, `{"stack_paths_200"}`). In `instruction_following`, 55% emit `{"style=`. | `import doctest` specifically was 0/160; the model learned to open with `{"`, but pulls from web/CSS/dict token sequences rather than valid JSON fields. | **HIGH** | Evaluate raw canonical base checkpoint (Step 10,000) on bare `{"` to map logit probabilities across next tokens. | **Yes** (pure read-only inference) |
| **H3: Capacity Saturation at 3.44M Parameters** | Model successfully learns surface syntax and termination (EOS reached 54.37%), but fails fact retrieval, arithmetic logic, and JSON schema tracking simultaneously. Validation loss plateaus at 2.8786. | Smaller models have been shown to perform single tasks (e.g. pure arithmetic or pure extraction) when capacity is not fragmented across 8 disparate tasks. | **MEDIUM** | Compare performance of 3.44M single-task model vs 3.44M multi-task model. | No (requires comparison training run) |
| **H4: Training Data Volume / Diversity Deficiency** | 150 examples per category is very small for generalized instruction following without pre-existing instruction-tuned priors. | Training loss dropped to 1.3414, showing the model memorized the training set easily. | **MEDIUM** | Test train-split reconstruction accuracy on seen training examples. | **Yes** (pure read-only inference) |
| **H5: Evaluator Defect / False Rejections** | N/A | Manual audit of all 140 scorable records showed 0 valid answers rejected. Evaluator logic is hardened and verified. | **REJECTED (CONFIRMED ZERO IMPACT)** | Run regression suite and manual inspection. | **Yes** (Already verified) |
| **H6: Causal Loss Masking / Shifting Implementation Bug** | N/A | Code inspection and simulation proved causal alignment is mathematically correct: input token $t$ predicts target token $t$, prompt is masked with -100, EOS is supervised. | **REJECTED (CONFIRMED ZERO IMPACT)** | Step-by-step tensor indexing simulation. | **Yes** (Already verified) |
| **H7: Prompt Delimiter / Tokenization Mismatch** | N/A | Tokenizer encodes `format_conditioned_prompt` identically in both training dataset loader and evaluation pipeline. | **REJECTED (CONFIRMED ZERO IMPACT)** | Direct token ID equality check. | **Yes** (Already verified) |

---

## 2. In-Depth Diagnosis of the Code-Prior & CSS Attractor

### The Misconception Regarding `import doctest`
In earlier SFT V3 pilot experiments, when prompted with bare `\nResponse:`, the candidate model frequently collapsed into Python unit tests and doctests (`{"__main__":\n import doctest`).
In Exp19, however:
- **`import doctest` occurrences in generated outputs: 0 / 160 (0.0%)**.
- The model did **not** generate Python docstrings or test runners.
- Instead, when encountering an instruction demanding JSON (or uppercase transformation), the model produced:
  - `{"style={"style="}` (18 records)
  - `{"stack_paths_200"}` (8 records)
  - `{"session_pathsv_200"}` / `{"ses` (5 records)
- **Interpretation**: The model has learned that after `Instruction: Format ... as valid JSON\nResponse:`, the output must start with `{"`. However, within its pre-trained embedding space, `{"` is heavily associated with CSS inline style dictionaries (e.g. `{"style="...`) and server route dictionaries (`{"stack_paths...`). The 150 JSON examples in the training set were sufficient to enforce the opening `{"` and clean EOS termination, but insufficient to steer internal attention heads toward extracting the prompt's key-value pair.

---

## 3. In-Depth Diagnosis of Factual and Semantic Failures

- **Repetition Traps**: In `factual_qa`, `context_extraction`, and `uncertainty_abstention`, the model falls into cyclical phrases such as `"of the city of the city of the city"` (64/160 records).
- **Surface Mimicry Without Grounding**: Diversification successfully replaced `"The difference"` with `"The firm"`, `"'It' is"`, and `"References"`. However, the model simply uses these as interchangeable decorative prefixes followed by repetitive n-grams.
- **Root Cause**: The model has memorized the *syntactic frame* of answers (e.g. `"The X of Y is Z"`) without learning to bind entity tokens from the prompt into the response slot.
