# REPLAY DATA AUDIT & MIXTURE ANALYSIS (SFT V3)
**Scope:** Replay Continuation Provenance, Loss Masking, Document Isolation & Curriculum Mixtures  
**Status:** Complete Audit (No Training Run)  
**Execution Context:** `experiments/sft_v3_pilot_readiness/`

---

## 1. Executive Summary

Replay continuation data serves as a catastrophic forgetting guardrail in small neural architectures. ChakrMicro possesses approximately 3.44 million parameters; fine-tuning on purely structured JSON without replay data carries an acute risk of **immediate-EOS collapse** (where the model terminates after 1–2 tokens) or the degeneration of general language comprehension.

In SFT V3, the training split contains:
- **140 structured grounded extraction examples** (70.0%)
- **60 unconditioned natural language replay examples** (30.0%)
- **Validation split:** 30 structured (75.0%) and 10 replay (25.0%).

This audit examines the exact serialization, document isolation, causal loss masking, and trade-offs of this replay composition.

---

## 2. Replay Data Provenance & Document Isolation

To avoid data leakage and benchmark contamination, replay examples are drawn strictly from reviewed pretraining raw corpora with guaranteed document-level partitioning:

| Split | Source Documents / Corpora | Record Count | Average Sequence Length | Disjointness Verification |
| :--- | :--- | :--- | :--- | :--- |
| **Train Replay** | `data/raw/english/english_computer_science.txt`<br>`data/raw/english/stage_b_systems.txt` | 60 | 142.4 tokens | **100% Disjoint** from Validation |
| **Val Replay** | `data/raw/english/english_systems.txt` | 10 | 138.6 tokens | **100% Disjoint** from Train |
| **Held-out Test** | `data/raw/code/code_foundation.txt`<br>`data/raw/mathematics/math_foundation.txt` | *(Held out)* | *(Frozen)* | **Untouched & Unaccessed** |

### Verification Findings:
1. **Zero Document Overlap**: Train replay and validation replay originate from completely separate files.
2. **Zero Prompt / Continuation Overlap**: Word-level prefix and suffix checks confirm 0 prompt string matches and 0 target string collisions between train and validation replay pools.

---

## 3. Serialization and Causal Token Alignment

Replay continuation differs fundamentally from instruction-response pairs:
- **No Response Delimiter**: Replay prompts do NOT end with `\nResponse:`. They consist of natural sentence prefixes (e.g. *"ChakrView is an indigenous artificial intelligence research "*).
- **No Forced EOS Boundary (`append_eos = False`)**:
  - In structured instructions (`append_eos = True`):
    - `input_seq = [BOS] + prompt_tokens + response_tokens`
    - `target_seq = [-100] * len(prompt_tokens) + response_tokens + [EOS]`
  - In replay continuation (`append_eos = False`):
    - `input_seq = [BOS] + prompt_tokens + response_tokens[:-1]`
    - `target_seq = [-100] * len(prompt_tokens) + response_tokens`
- **Causal Shift Parity**:
  At position `i = len(prompt_tokens)` (the final prompt token), the model predicts `target_seq[i] == response_tokens[0]`. This ensures standard autoregressive next-token prediction across the continuation boundary without artificially forcing the model to predict an `<|endoftext|>` token prematurely.

---

## 4. Evaluation of Replay Mixture Policies

How much replay data should be combined with structured extraction in an SFT pilot? We evaluate three distinct candidate policies:

### Policy A: Structured-Only Pilot (0% Replay, 140 Train Examples)
- **Concept**: Train exclusively on the 140 grounded structured extraction records.
- **Hypothesis**: Maximizes gradient signal directed toward JSON formatting and context extraction.
- **Known Failure Mode / Confounder**: In previous ChakrView pilots (Batch 42–62), fine-tuning on structured data without replay caused immediate-EOS regression on standard prompts. The model learned that any prompt requires ending within 10–20 tokens, destroying paragraph continuation capability.
- **Selection Criterion**: Would only be selected if model capacity is so constrained that replay actively interferes with JSON syntax acquisition.

### Policy B: Balanced 70/30 Hybrid Mix (Current SFT V3 Design — Recommended)
- **Concept**: 140 structured examples (70%) combined with 60 continuation replay examples (30%).
- **Hypothesis**: Balances dense cross-entropy signal for structured extraction while maintaining continuous gradient regularization on natural language flow.
- **Evidence from Prior Pilots**: In Batch 3071–3110 and Batch 3111–3150, the 70/30 mixture maintained 4/4 passing on canonical immediate-EOS regression probes, with continuation length remaining stable at ~40 tokens.
- **Confounder**: 30% of optimization steps are spent on general text, slightly slowing down loss reduction on schema keys.
- **Selection Criterion**: Recommended reference policy for the upcoming pilot.

### Policy C: Weighted/Phased Loss Mix (Dynamic Loss Weighting)
- **Concept**: Replay examples receive a scaled loss multiplier (e.g. $w_{\text{replay}} = 0.5$, $w_{\text{struct}} = 1.0$), or a phased warm-up schedule is applied.
- **Evaluation**: While theoretically appealing, introducing custom loss weights into the cross-entropy objective modifies the canonical training engine and requires hyperparameter tuning that cannot be grounded on CPU without multiple exploratory training runs.
- **Verdict**: Rejected for SFT V3 pilot to maintain strict fidelity to the canonical `sft_engine.py` contract.

---

## 5. Audit Conclusion

The SFT V3 replay dataset is technically sound, verified for document-level isolation, correctly uses `append_eos=False` without delimiter corruption, and appropriately balances structured formatting pressure against catastrophic forgetting risk. Policy B (70% structured / 30% replay) remains the justified, safe baseline for a future authorized pilot.
