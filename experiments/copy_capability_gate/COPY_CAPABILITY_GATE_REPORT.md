# Copy Capability Gate — Final Diagnostic Report

## 1. Executive Summary

This diagnostic investigation evaluated whether switching from the standard `Context / Instruction / Response` prompt framing to an explicit dialog format (`User: ... Assistant:`) enables native context-grounded copying in ChakrMicro, independently of JSON syntax and external decoding constraints.

The evaluation was executed entirely in inference mode on CPU across **20 verified validation records** from the SFT V3 curriculum where the target value is explicitly present in the source passage. Both the canonical Step-10,000 base checkpoint and the SFT V3 100-step pilot candidate were evaluated under identical greedy decoding settings (`max_new_tokens=40`, temperature=0.0).

### Key Empirical Findings:
1. **Attractor Suppression**: Switching to the dialog format drastically suppresses the pretraining Python doctest attractor on the Candidate checkpoint, dropping the attractor rate from **18/20 (90.0%)** under `Context / Instruction / Response` down to **3/20 (15.0%)** under `User: ... Assistant:`.
2. **Copying Remains Zero**: Despite the complete elimination of the doctest attractor across the majority of examples, the exact value match rate and normalized value match rate remained **0/20 (0.0%)** across both Base and Candidate checkpoints in all prompt formats.
3. **Behavior Shift**: Under the dialog format, instead of emitting `{"__main__":\n import doctest`, the Candidate model generated Python exception handlers (e.g., `\n raise ValueError(...)`) or repetitive token sequences (e.g., `"index", "index", "code"`). It never routed attention back to the source entity in the prompt context.

---

## 2. Experimental Setup & Cryptographic Verification

All artifacts were verified bit-identical prior to execution:
- **Base Checkpoint (Step 10,000)**: `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
  - SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
- **Candidate Checkpoint (Step 100 Pilot)**: `experiments/sft_v3_pilot_readiness/pilot_run_v3/sft_best_checkpoint.pt`
  - SHA-256: `bfde007db1a90dde5ccc75d14a9ffe59f57b2393b87ac00b19422b8d088d1879`
- **Tokenizer Vocab**: `data/experiments/vocab_4096/vocab.json`
  - SHA-256: `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a`
- **Tokenizer Merges**: `data/experiments/vocab_4096/merges.json`
  - SHA-256: `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7`
- **Protected File Status**:
  - `chakrview/runtime/session.py` remains completely untouched (`git diff` clean).

---

## 3. Dataset & Ground-Truth Verification

20 validation records were selected from the four non-abstention structured categories:
- `anti_code_attractor_guard` (5 records, target field: `status`)
- `grounded_inventory_item` (5 records, target field: `name`)
- `grounded_extraction_server_status` (5 records, target field: `status`)
- `grounded_telemetry_metric` (5 records, target field: `metric`)

### Ground-Truth Passage Containment Audit
Every single expected answer was audited against its corresponding context string. In **20/20 (100%)** records, the ground-truth string was explicitly contained in the passage text:
- `SFT3_VAL_ATC_002`: expected `'dispatched'` -> context contains `'...with status dispatched and code 202.'`
- `SFT3_VAL_INV_005`: expected `'ribbon_cable_harness'` -> context contains `'...110 units of ribbon_cable_harness on hand.'`
- `SFT3_VAL_MET_004`: expected `'acoustic_noise_db'` -> context contains `'...registers acoustic_noise_db at 42.0 decibels.'`
- `SFT3_VAL_SRV_001`: expected `'queued'` -> context contains `'...reports val_host_01 is queued with acceptance code 202.'`

No synthetic labels were inferred or created.

---

## 4. Prompt Framing Formats Tested

Three prompt framing conditions were compared using identical greedy generation parameters:

1. **Format 1: Existing CIR (Context / Instruction / Response)**:
   ```text
   Context: Metrics telemetry_sender dispatched batch with status dispatched and code 202.
   Instruction: Copy the exact value of status from context:
   Response:
   ```
   - Boundary tokens: `[115, 279, 323, 61]` (`"ponse:"`)

2. **Format 2: Explicit Dialog (User: ... Assistant:)**:
   ```text
   User: In the following context: "Metrics telemetry_sender dispatched batch with status dispatched and code 202."
   Copy the exact value of status from context:
   Assistant:
   ```
   - Boundary tokens: `[118, 1855, 1807, 61]` (`"ssistant:"`)

3. **Format 3: Plain CQA (Context / Question / Answer)**:
   ```text
   Context: Metrics telemetry_sender dispatched batch with status dispatched and code 202.
   Question: Copy the exact value of status from context:
   Answer:
   ```
   - Boundary tokens: `[118, 122, 283, 61]` (`"swer:"`)

---

## 5. Comprehensive Metric Comparison

### Normalization Rule
- Strip leading and trailing whitespace.
- Convert to lowercase.
- Strip outer quotes (`"`, `'`, `` ` ``) and trailing punctuation (`,`, `.`, `;`, `:`, `!`, `?`).

### Empirical Results Summary (N = 20 Records)

| Prompt Framing Format | Model | Exact Value Match | Normalized Match | Doctest Attractor Rate | Empty / Premature EOS | Average Generated Tokens |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Context / Instruction / Response** | **Base** | 0/20 (0.0%) | 0/20 (0.0%) | 1/20 (5.0%) | 0/20 (0.0%) | 40.0 |
| **Context / Instruction / Response** | **Candidate** | 0/20 (0.0%) | 0/20 (0.0%) | **18/20 (90.0%)** | 0/20 (0.0%) | 38.95 |
| **Dialog (User: ... Assistant:)** | **Base** | 0/20 (0.0%) | 0/20 (0.0%) | 0/20 (0.0%) | 1/20 (5.0%) | 40.0 |
| **Dialog (User: ... Assistant:)** | **Candidate** | 0/20 (0.0%) | 0/20 (0.0%) | **3/20 (15.0%)** | 0/20 (0.0%) | 40.0 |
| **Plain (Context / Question / Answer)** | **Base** | 0/20 (0.0%) | 0/20 (0.0%) | 2/20 (10.0%) | 2/20 (10.0%) | 40.0 |
| **Plain (Context / Question / Answer)** | **Candidate** | 0/20 (0.0%) | 0/20 (0.0%) | **3/20 (15.0%)** | 0/20 (0.0%) | 36.7 |

---

## 6. Qualitative Analysis of Generation Outputs

### Representative Example 1: `SFT3_VAL_ATC_002`
- **Context**: `Metrics telemetry_sender dispatched batch with status dispatched and code 202.`
- **Target Value**: `dispatched`
- **Format 1 (`\nResponse:`) Outputs**:
  - Base: `\n\n\nNovember 2005 century of the first cannot the first of the word of the word of the wor`
  - Candidate: `{"__main__":\n import doctest\n\n # import doctest\n # import doctest\n\n # import doctest` *(Attractor)*
- **Format 2 (`Assistant:`) Outputs**:
  - Base: `\n raise ValueError(")\n # If the first number of the number of the number of the number of the `
  - Candidate: `\n raise ValueError(", "index", "index", "index", "index", "in` *(No attractor, but ValueError continuation)*
- **Format 3 (`Answer:`) Outputs**:
  - Base: `\n\n\nThe first of the first of the word of the word of the word of the word of the show. It is the fi`
  - Candidate: `\n raise ValueError("index must be a non-negative", "index", "code", "in` *(No attractor, but ValueError continuation)*

### Representative Example 2: `SFT3_VAL_INV_005`
- **Context**: `Wiring depot records 110 units of ribbon_cable_harness on hand.`
- **Target Value**: `ribbon_cable_harness`
- **Format 1 (`\nResponse:`) Outputs**:
  - Base: `\n\n\n # If the first number of the number of the number of the number of the number of the number of the number of the `
  - Candidate: `{"strict", "code", "index", "index", "index", "index", "code"`
- **Format 2 (`Assistant:`) Outputs**:
  - Base: `\n\n # If the first number of the number of the number of the number of the number of the number of the number of the numb`
  - Candidate: `\n raise ValueError("index must be a resulting the number of the number of the number of the number of `

### Interpretation
In Format 1, the candidate's SFT tuning caused `\nResponse:` to strongly cue the opening brace of JSON `{`, which immediately chained into the pretraining code-corpus pattern `{"__main__":\n import doctest`.
In Format 2, `Assistant:` prevents the immediate `{` opening, which suppresses the doctest trigger. However, because the model possesses no learned cross-attention routing from prompt context to response generation, it defaults to other dominant pretraining continuation attractors (in this case, Python function definitions with `raise ValueError(...)`).

---

## 7. Evidence-Based Root Cause Diagnosis

1. **Prompt boundary tokens govern the attractor, not internal reasoning**:
   The transition from `18/20 (90%)` to `3/20 (15%)` proves that the doctest attractor is not a damaged latent state or a ruined checkpoint, but a prompt boundary artifact.
2. **Context-to-output copying is completely unlearned**:
   Neither 10,000 steps of pretraining nor 100 steps of SFT V3 established induction heads for copying arbitrary entity tokens from context.
3. **Prompt formatting alone cannot elicit capabilities that do not exist**:
   Changing prompt framing alters surface continuation styles (from doctest code to exception handling), but achieves **0.0%** value copying.

---

## 8. Final Verdict

`PROMPT IMPROVES FORMAT ONLY`

### Justification:
The dialog-style prompt format drastically reduces the code-attractor rate (from 90.0% to 15.0%) and eliminates doctest degeneration across 85% of examples. However, context-grounded copying accuracy remains strictly **0/20 (0.0%)**. The prompt framing change improves output format only and does not establish native copying capability.

---

## 9. Recommended Next Experiment

**Controlled Single-Field Copying Curriculum Pilot (250–500 steps, CPU)**:
- **Design**: Train exclusively on single-field copy examples using the dialog template (`User: In the following context: ... Assistant: <value>`).
- **Objective**: Determine whether ChakrMicro's 4.5M parameters can develop induction heads and cross-attention pointer circuits for a single entity before introducing JSON syntax, multi-field schemas, or boolean reasoning.
- **Why this must precede any other training**: Until the model can copy a single explicitly stated token from context (Task A/B), training it on multi-field structured JSON will only result in memorized syntax and continuation attractors.
