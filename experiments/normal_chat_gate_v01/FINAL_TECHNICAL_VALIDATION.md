# ChakrView Normal Chat Gate v0.1 -- Final Technical Validation Report

**Date**: 2026-10-22  
**Target Evaluation**: `experiments/normal_chat_gate_v01/`  
**Execution Mode**: Non-training, non-destructive audit and technical verification  

---

## 1. Verified Observations

From direct file inspection, bytecode verification, and checksum validation:
1. **Model and Evaluation Rigidity**:
   - The canonical checkpoint `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` is 100% bit-for-bit intact against its canonical SHA-256 (`96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`).
   - The protected runtime file `chakrview/runtime/session.py` is identical to its expected hash (`9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67`).
   - All six source datasets across Exp16 v3 and Exp16 v3.1 match their recorded reference hashes.
2. **Evaluation Records Completeness and Exact Alignment**:
   - `prompts.jsonl` contains exactly 20 records and 20 unique IDs.
   - `outputs.jsonl` contains exactly 20 records and 20 unique IDs.
   - `SCORES.csv` contains exactly 20 rows and 20 unique IDs.
   - The three files exhibit strict 1-to-1 bijection (`p_ids == o_ids == s_ids`).
   - Every score in `SCORES.csv` has a prompt-specific evidence rationale string citing the exact output excerpt, expected intent, and failure mode.
3. **Prompt Conditioning Verification**:
   - Prompt formatting was recomputed using `format_conditioned_prompt(raw_prompt, context)` and matches the stored `formatted_prompt` across all 20 records with zero mismatches.
4. **Token Generation and Idempotent Decoding**:
   - Every generated sequence contains exactly 48 tokens.
   - Re-decoding each `generated_tokens` array using the active BPE tokenizer reproduces the recorded `generated_text` with zero character mismatches across all 20 records.
5. **Runtime Immutability (`Delta-W = 0`)**:
   - The recorded outputs show `weight_hash_verified = True`. The adapter verifies the in-memory parameter digest before and after each sequence generation.

---

## 2. Unverified or Inconclusive Claims

1. **Model EOS Generation Capacity (INCONCLUSIVE)**:
   - In the gate evaluation, the model was run under greedy decoding (`temperature = 0.0`). In all 20 prompts, token ID `1` (`<EOS>`) was never emitted as the top argmax prediction.
   - Because nonzero temperature sampling (`temperature > 0.0`), top-k, top-p, and repetition penalties were not evaluated, claiming that the model is *incapable of generating EOS under any regime* is **INCONCLUSIVE**.
2. **Pretraining Data Composition as Proven Root Cause (SUPPORTED HYPOTHESIS)**:
   - The pervasive emission of synthetic arithmetic headers (`Step-by-Step Solution:\nThe total number of...`) strongly aligns with high-frequency templates from the synthetic pretraining curriculum. However, because the pretraining corpus was not token-for-token indexed during this audit, this remains a **Supported Hypothesis**, not a directly proven root cause.
3. **Semantic Scoring Sign-off (PROVISIONAL)**:
   - Automated evaluation has assigned 0 points across all applicable dimensions based on objective evidence (token looping, non-adherence to constraints, irrelevant math templates). However, subjective semantic dimensions (e.g., relevance and task completion) require explicit human reviewer sign-off.

---

## 3. Special-Token and Stopping-Behavior Findings

Exact code citations from the active repository:
- **Designated Special Tokens** ([chakrview/tokenizer/special_tokens.py:16-18](file:///D:/Project/ChakrView/chakrview/tokenizer/special_tokens.py#L16-L18)):
  - `<BOS>` = `0` (`BOS_ID`)
  - `<EOS>` = `1` (`EOS_ID`)
  - `<PAD>` = `2` (`PAD_ID`)
  - Total special tokens: exactly 3. Tokens `<UNK>`, `<MASK>`, `<SEP>`, `<NL>` are strictly excluded.
- **Inference Contract Stop Tokens** ([chakrview/runtime/neural_inference_contract.py:90](file:///D:/Project/ChakrView/chakrview/runtime/neural_inference_contract.py#L90)):
  - `stop_tokens: List[int] = field(default_factory=lambda: [EOS_ID])` (Token ID `1`).
- **Active Tokenizer Vocabulary Verification** ([data/experiments/vocab_4096/config.json:7-12](file:///D:/Project/ChakrView/data/experiments/vocab_4096/config.json#L7-L12)):
  - The loaded tokenizer config confirms: `"<BOS>": 0`, `"<EOS>": 1`, `"<PAD>": 2`. Token `1` is verified to be `<EOS>`.
- **Generated Token Appending vs. EOS Check** ([chakrview/runtime/neural_runtime_adapter.py:329-334](file:///D:/Project/ChakrView/chakrview/runtime/neural_runtime_adapter.py#L329-L334)):
  - In `NeuralRuntimeAdapter.generate()`, `next_token` is appended to `generated_tokens` and `curr_tokens` **before** checking EOS:
    ```python
    generated_tokens.append(next_token)
    curr_tokens.append(next_token)

    if active_eos_stop and next_token == 1:  # EOS_ID
        stop_reason = "STOP_TOKEN"
        break
    ```
  - Therefore, if the model had generated `<EOS>`, token ID `1` would appear as the final element of `generated_tokens`.
- **Effective Hyperparameters in Gate Evaluation**:
  - `stop_on_eos`: Initialized as `False` by default in [neural_runtime_adapter.py:105](file:///D:/Project/ChakrView/chakrview/runtime/neural_runtime_adapter.py#L105); passed as `True` during evaluation (`active_eos_stop = True`).
  - `max_new_tokens`: `48` (capped at 64; configured to 48 in gate run).
  - `decoding`: `"greedy"` ([neural_runtime_adapter.py:99](file:///D:/Project/ChakrView/chakrview/runtime/neural_runtime_adapter.py#L99)).
  - `temperature`: `0.0` (enforced when `active_decoding == "greedy"` at [neural_runtime_adapter.py:274](file:///D:/Project/ChakrView/chakrview/runtime/neural_runtime_adapter.py#L274)).
  - `top_k`: `0`.
  - `top_p`: `1.0`.
  - `repetition_penalty`: `1.0` (unpenalized baseline).
- **Semantics of `stop_reason: MAX_TOKENS`**:
  - `stop_reason` initializes to `"MAX_TOKENS"` at [neural_runtime_adapter.py:311](file:///D:/Project/ChakrView/chakrview/runtime/neural_runtime_adapter.py#L311).
  - It only switches to `"STOP_TOKEN"` if `next_token == 1`.
  - Thus, `MAX_TOKENS` indicates that the loop exhausted its entire 48-step budget without ever selecting token `1`.
- **Contents of Recorded `generated_tokens`**:
  - The recorded `generated_tokens` lists contain raw integer token IDs.
  - Inspection confirms **0 special tokens** (no `0`, `1`, or `2`) across all 20 records. All generated tokens are non-special text tokens.

---

## 4. Evaluation Record Integrity & Human Review Table

### Record Integrity Checks:
- Parsing: 100% clean JSON / CSV parsing.
- Count validation: Exactly 20 records across all files.
- Duplicate IDs: 0.
- Missing IDs: 0.
- Bijection: Strict 1-to-1 prompt-output-score alignment.
- Denominator reconciliation: Exactly **180 possible points** (Grounding applicable to 6 prompts; Uncertainty applicable to 4 prompts; 14 and 16 N/A exclusions respectively).
- Score status: Provisional (0/180 points earned based on empirical evidence). Original `SCORES.csv` preserved.

### Human-Review Table:
| Prompt ID | Category | Output Excerpt | Expected Intent | Scores (Rel/IF/Grd/TC/Unc/Rep) | Evidence Statement | Review Status |
| :--- | :--- | :--- | :--- | :---: | :--- | :---: |
| `PROMPT_01_FACTUAL` | Simple factual question | `'\nStep-by-Step Solution:\nThe total n'` | Identify Mars as the Red Planet. | `0/0/N/A/0/N/A/0` | Output is an ungrounded degenerate loop truncated at 48 tokens (MAX_TOKENS). Snippet: '\nStep-by-Step Solution:\nThe to'. Expected intent: Identify Mars as the Red Planet. | PROVISIONAL (Requires Human Review) |
| `PROMPT_02_INSTRUCTION` | Clear instruction following | `'\n* [http://www.gov.co.com/books?id='` | Output OLLEH exactly. | `0/0/N/A/0/N/A/0` | Output is an ungrounded degenerate loop truncated at 48 tokens (MAX_TOKENS). Snippet: '\n* [http://www.gov.co.com/book'. Expected intent: Output OLLEH exactly. | PROVISIONAL (Requires Human Review) |
| `PROMPT_03_SHORT_EXPLANATION` | Short explanation | `'\n\nThe first of the city of the city'` | Mention chlorophyll absorbing sunlight. | `0/0/N/A/0/N/A/0` | Output is an ungrounded degenerate loop truncated at 48 tokens (MAX_TOKENS). Snippet: '\n\nThe first of the city of the'. Expected intent: Mention chlorophyll absorbing sunlight. | PROVISIONAL (Requires Human Review) |
| `PROMPT_04_MULTI_STEP_REASONING` | Multi-step reasoning | `'\n\nStep-by-Step Solution:\nThe total '` | 120 miles / 60 mph = 2 hours -> 4:00 PM. | `0/0/N/A/0/N/A/0` | Output is an ungrounded degenerate loop truncated at 48 tokens (MAX_TOKENS). Snippet: '\n\nStep-by-Step Solution:\nThe t'. Expected intent: 120 miles / 60 mph = 2 hours -> 4:00 PM. | PROVISIONAL (Requires Human Review) |
| `PROMPT_05_CONTEXT_RETENTION` | Context retention across turns | `'\n\n\nSince of the fileso is the first'` | Identify penguin. | `0/0/0/0/N/A/0` | Output is an ungrounded degenerate loop truncated at 48 tokens (MAX_TOKENS). Snippet: '\n\n\nSince of the fileso is the '. Expected intent: Identify penguin. | PROVISIONAL (Requires Human Review) |
| `PROMPT_06_REFERENCE_RESOLUTION` | Reference resolution | `'\n\n* Andron is the shown of the show'` | Refer to the bicycle. | `0/0/0/0/N/A/0` | Output is an ungrounded degenerate loop truncated at 48 tokens (MAX_TOKENS). Snippet: '\n\n* Andron is the shown of the'. Expected intent: Refer to the bicycle. | PROVISIONAL (Requires Human Review) |
| `PROMPT_07_USER_CORRECTION` | Following a user correction | `'\n\n\nStepackways are the fileadd of t'` | Friday. | `0/0/0/0/N/A/0` | Output is an ungrounded degenerate loop truncated at 48 tokens (MAX_TOKENS). Snippet: '\n\n\nStepackways are the fileadd'. Expected intent: Friday. | PROVISIONAL (Requires Human Review) |
| `PROMPT_08_CONFLICTING_INSTRUCTIONS` | Handling conflicting instructions | `'\n\nThe first of the city of the city'` | Identify or resolve the conflict. | `0/0/N/A/0/0/0` | Output is an ungrounded degenerate loop truncated at 48 tokens (MAX_TOKENS). Snippet: '\n\nThe first of the city of the'. Expected intent: Identify or resolve the conflict. | PROVISIONAL (Requires Human Review) |
| `PROMPT_09_APPROPRIATE_UNCERTAINTY` | Appropriate uncertainty when information is missing | `'\nStep-by-Step Solution:\nThe total o'` | Express uncertainty. | `0/0/N/A/0/0/0` | Output is an ungrounded degenerate loop truncated at 48 tokens (MAX_TOKENS). Snippet: '\nStep-by-Step Solution:\nThe to'. Expected intent: Express uncertainty. | PROVISIONAL (Requires Human Review) |
| `PROMPT_10_ABSTENTION_FACTS` | Abstention instead of invented facts | `'\n\n* John Christiangled of the C. It'` | Abstain or state manager is unknown. | `0/0/0/0/0/0` | Output is an ungrounded degenerate loop truncated at 48 tokens (MAX_TOKENS). Snippet: '\n\n* John Christiangled of the '. Expected intent: Abstain or state manager is unknown. | PROVISIONAL (Requires Human Review) |
| `PROMPT_11_CONCISE_ANSWER` | Concise answer generation | `'\n* Andromed of the Andrography of t'` | H2O. | `0/0/N/A/0/N/A/0` | Output is an ungrounded degenerate loop truncated at 48 tokens (MAX_TOKENS). Snippet: '\n* Andromed of the Andrography'. Expected intent: H2O. | PROVISIONAL (Requires Human Review) |
| `PROMPT_12_STRUCTURED_EXPLANATION` | Longer structured explanation | `'\n\nThe first of the world of the wor'` | Bullet points listing primary colors. | `0/0/N/A/0/N/A/0` | Output is an ungrounded degenerate loop truncated at 48 tokens (MAX_TOKENS). Snippet: '\n\nThe first of the world of th'. Expected intent: Bullet points listing primary colors. | PROVISIONAL (Requires Human Review) |
| `PROMPT_13_REPETITION_RESISTANCE` | Repetition resistance | `'\n\nThe first famous of the city of t'` | List three words and stop. | `0/0/N/A/0/N/A/0` | Output is an ungrounded degenerate loop truncated at 48 tokens (MAX_TOKENS). Snippet: '\n\nThe first famous of the city'. Expected intent: List three words and stop. | PROVISIONAL (Requires Human Review) |
| `PROMPT_14_OUTPUT_FORMAT_CONSTRAINT` | Output-format constraint | `'\n\nThe first of the first cannot be '` | Valid JSON status: ok. | `0/0/N/A/0/N/A/0` | Output is an ungrounded degenerate loop truncated at 48 tokens (MAX_TOKENS). Snippet: '\n\nThe first of the first canno'. Expected intent: Valid JSON status: ok. | PROVISIONAL (Requires Human Review) |
| `PROMPT_15_BASIC_ARITHMETIC` | Basic arithmetic | `'\nStep-by-Step Solution:\nThe total n'` | 42. | `0/0/N/A/0/N/A/0` | Output is an ungrounded degenerate loop truncated at 48 tokens (MAX_TOKENS). Snippet: '\nStep-by-Step Solution:\nThe to'. Expected intent: 42. | PROVISIONAL (Requires Human Review) |
| `PROMPT_16_SIMPLE_COMPARISON` | Simple comparison | `'\nStep-by-Step Solution:\nThe total n'` | An elephant. | `0/0/N/A/0/N/A/0` | Output is an ungrounded degenerate loop truncated at 48 tokens (MAX_TOKENS). Snippet: '\nStep-by-Step Solution:\nThe to'. Expected intent: An elephant. | PROVISIONAL (Requires Human Review) |
| `PROMPT_17_SUMMARIZATION` | Summarization of supplied text | `'\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n'` | Sunlight frees electrons creating current. | `0/0/0/0/N/A/0` | Output is an ungrounded degenerate loop truncated at 48 tokens (MAX_TOKENS). Snippet: '\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n'. Expected intent: Sunlight frees electrons creating current. | PROVISIONAL (Requires Human Review) |
| `PROMPT_18_STRICT_CONTEXT_QA` | Answering strictly from supplied context | `'\n\n\nSublish, the Special Spectitrone'` | Liquid nitrogen. | `0/0/0/0/N/A/0` | Output is an ungrounded degenerate loop truncated at 48 tokens (MAX_TOKENS). Snippet: '\n\n\nSublish, the Special Specti'. Expected intent: Liquid nitrogen. | PROVISIONAL (Requires Human Review) |
| `PROMPT_19_AMBIGUOUS_REQUEST` | Handling an ambiguous request appropriately | `'\nStep-by-Step Solution:\nThe total n'` | Ask for clarification. | `0/0/N/A/0/0/0` | Output is an ungrounded degenerate loop truncated at 48 tokens (MAX_TOKENS). Snippet: '\nStep-by-Step Solution:\nThe to'. Expected intent: Ask for clarification. | PROVISIONAL (Requires Human Review) |
| `PROMPT_20_ENDING_WITHOUT_RUNAWAY` | Ending a response without runaway generation | `'\nThe total of the city of the city '` | Goodbye / Bye. | `0/0/N/A/0/N/A/0` | Output is an ungrounded degenerate loop truncated at 48 tokens (MAX_TOKENS). Snippet: '\nThe total of the city of the '. Expected intent: Goodbye / Bye. | PROVISIONAL (Requires Human Review) |

---

## 5. Recomputed Repetition Metrics

Recomputed directly from `outputs.jsonl` using strict, documented heuristics:

### Heuristic Definitions:
1. **Word-Level Trigram Duplication**:
   - Extract lowercase words using regex `\b\w+\b`.
   - Form adjacent 3-tuples: `(w_i, w_{i+1}, w_{i+2})`.
   - Condition: Count of distinct trigrams < total trigrams (i.e. at least one trigram occurs $\ge 2$ times).
2. **Token-Level Bigram Duplication**:
   - Form adjacent token pairs from `generated_tokens`: `(t_i, t_{i+1})`.
   - Condition: At least one token bigram occurs $\ge 4$ times in the 48-token sequence.
3. **Consecutive-Character Runs**:
   - Regex scan for consecutive newline characters `\n{10,}` or repeated identical characters `c{10,}`.
4. **Combined Looping / Runaway**:
   - Any sequence meeting Word-Trigram, Token-Bigram, or Character-Run criteria.

### Exact Heuristic Results:
| Repetition Dimension | Threshold / Definition | Satisfying Prompts | Percentage |
| :--- | :--- | :---: | :---: |
| **Word-Trigram Duplication** | $\ge 2$ occurrences of identical word 3-tuple | **15 / 20** | **75.0%** |
| **Token-Bigram Duplication** | $\ge 4$ occurrences of identical token pair | **19 / 20** | **95.0%** |
| **Consecutive Newline Runaway** | $\ge 10$ consecutive `\n` characters | **1 / 20** (PROMPT_17) | **5.0%** |
| **Consecutive Character Runaway** | $\ge 10$ consecutive identical non-newline chars | **1 / 20** (PROMPT_02, 12 `'A'`) | **5.0%** |
| **Combined Looping / Runaway Rate** | Word-trigram OR Token-bigram OR Char-run | **20 / 20** | **100.0%** |

*Note on PROMPT_18*: PROMPT_18 contains 3 occurrences of subword token bigram `(770, 559)` (`" the"`), which falls just below the $\ge 4$ token-bigram threshold, but exhibits cyclic degenerate syntax (`Sublish, the Special Spectitrone of the celeastern the show of the world of the was the fir`). Repetition flags provide objective linguistic signals and are distinguished from human semantic failure scoring.

---

## 6. Full Artifact Hash and Integrity Verification

| Artifact Name | Classification | Path | Expected Hash | Actual Hash | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **Canonical Checkpoint** | Checkpoint File | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **MATCH** |
| **Tokenizer Vocab** | Tokenizer File | `data/experiments/vocab_4096/vocab.json` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | **MATCH** |
| **Tokenizer Merges** | Tokenizer File | `data/experiments/vocab_4096/merges.json` | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | **MATCH** |
| **Protected Runtime** | Source Code File | `chakrview/runtime/session.py` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **MATCH** |
| **Model Architecture Source** | Source Code File | `chakrview/brain/model.py` | `1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361` | `1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361` | **MATCH** |
| **Exp16 v3.1 Train** | Source Dataset | `experiments/research_v09_conversation_sft/exp16_pointer_repair_v3_1/data/train_exp16_v3_1.jsonl` | `7b0c636df298a36aa35fb50bc3532a72d9af9bda0d969d50b2a4511a57e1fe14` | `7b0c636df298a36aa35fb50bc3532a72d9af9bda0d969d50b2a4511a57e1fe14` | **MATCH** |
| **Exp16 v3.1 Val** | Source Dataset | `experiments/research_v09_conversation_sft/exp16_pointer_repair_v3_1/data/val_exp16_v3_1.jsonl` | `e7cff8b889782e0977ce150c3e2e32c552b2f9a2af7b18e0d24349d19a06fe36` | `e7cff8b889782e0977ce150c3e2e32c552b2f9a2af7b18e0d24349d19a06fe36` | **MATCH** |
| **Exp16 v3.1 Test** | Source Dataset | `experiments/research_v09_conversation_sft/exp16_pointer_repair_v3_1/data/test_exp16_v3_1.jsonl` | `bc53766b33090743affe553f25092e83432252af6e3ff3053e33dcd183d4d8af` | `bc53766b33090743affe553f25092e83432252af6e3ff3053e33dcd183d4d8af` | **MATCH** |
| **Exp16 v3 Train** | Source Dataset | `experiments/research_v09_conversation_sft/exp16_pointer_repair_v3/data/train_exp16_v3.jsonl` | `d5d8b76871faad8e449d0a59df7834e7db1f37b3c21673743f20ee87249873dc` | `d5d8b76871faad8e449d0a59df7834e7db1f37b3c21673743f20ee87249873dc` | **MATCH** |
| **Exp16 v3 Val** | Source Dataset | `experiments/research_v09_conversation_sft/exp16_pointer_repair_v3/data/val_exp16_v3.jsonl` | `c9954fb71fd5251065076ad96cfdc78812e2f970eb3cf55dc09c94c38844e95e` | `c9954fb71fd5251065076ad96cfdc78812e2f970eb3cf55dc09c94c38844e95e` | **MATCH** |
| **Exp16 v3 Test** | Source Dataset | `experiments/research_v09_conversation_sft/exp16_pointer_repair_v3/data/test_exp16_v3.jsonl` | `096d018f357878e89560f982018446c8f09e5143767c1a5425d74bea9560ad10` | `096d018f357878e89560f982018446c8f09e5143767c1a5425d74bea9560ad10` | **MATCH** |

### Critical Clarifications:
1. **Fresh-Initialized Parameter Hash vs. File Hash**:
   - `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` is the SHA-256 parameter digest of an untrained model instantiated via `ChakrMicro(ModelConfig())` initialized with `torch.manual_seed(42)`. It is **NOT** a source-code file hash. The source code file hash of `chakrview/brain/model.py` is `1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361`.
2. **On-Disk Checkpoint Integrity vs. Runtime Weight Immutability**:
   - Matching the on-disk checkpoint SHA-256 proves the file was not altered on disk.
   - It does not alone prove in-memory immutability during execution.
   - In-memory immutability is proven separately by the runtime adapter's pre- and post-inference tensor checks (`Delta-W = 0`), which passed on 100% of the 20 prompts. Limitation: this verifies tensor immutability across the tested inference harness, but does not monitor arbitrary process memory outside the contract scope.

---

## 7. Separate Formal Verdicts

To prevent conflating record consistency with model capability, three separate verdicts are rendered:

### 1. Artifact Integrity: **PASS**
All canonical checkpoints, tokenizer configurations, protected runtime files, model source code, and source datasets match their established expected SHA-256 digests bit-for-bit.

### 2. Evaluation Record Integrity: **PASS**
`prompts.jsonl`, `outputs.jsonl`, and `SCORES.csv` parse cleanly, maintain strict 1:1 bijection across all 20 prompt IDs, contain complete telemetry, provide prompt-specific evidence rationales, and programmatically resolve the scoring denominator to exactly 180 points.

### 3. Normal Chat Capability Under Tested Configuration: **FAIL**
Under greedy unpenalized decoding (`temp=0.0`, `rep_penalty=1.0`), the raw unassisted ChakrMicro step_10000 model achieves **0 / 180 possible points (0.0%)**. It exhibits severe pretraining synthetic arithmetic bias, degenerate n-gram looping across 100% of prompts, context blindness, and 100% token budget exhaustion without generating `<EOS>`.

---

## 8. Smallest Justified Next Experiment

### Recommended Experiment: `exp17_instruction_sft_v1`
- **Objective**: Establish whether minimal conversational supervised fine-tuning (SFT) can eliminate the pretraining arithmetic bias, enforce turn-boundary termination (`<EOS>`), and enable basic instruction following.
- **Core Hypothesis**: Fine-tuning the ChakrMicro base checkpoint on a compact, balanced multi-turn instruction dataset with explicit conversational delimiters (`<BOS>`, role prefixes, `<EOS>`) will cause the model to emit `<EOS>` before the token budget and follow simple single-turn instructions without falling into degenerate loops.
- **Minimal Experimental Scope**:
  1. Train on <= 500 high-quality conversational turns covering factual QA, simple formatting constraints, and context extraction.
  2. Maintain frozen base embeddings where appropriate or train with conservative learning rate (<= 5e-5) to prevent catastrophic forgetting.
  3. Evaluate under both greedy decoding and sampling with mild repetition penalty (`1.15`).
- **Strict Success Criteria**:
  1. >= 80% of chat test prompts terminate via `STOP_TOKEN` (`<EOS>`) rather than `MAX_TOKENS`.
  2. Degenerate token-bigram repetition drops below 10%.
  3. Achieve non-zero score (>= 50% task completion) on the 20-prompt Normal Chat Gate.
*(Note: No training, fine-tuning, or experimental script execution has been initiated).*
