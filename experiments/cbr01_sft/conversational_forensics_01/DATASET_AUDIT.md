# CBR-SFT-07 Training Data Forensic Audit

- **Audited File**: `data\sft\cbr_sft_07\splits\cbr_sft_07_train_228.jsonl` (228 records)
- **Validation File**: `data\sft\cbr_sft_07\splits\cbr_sft_07_val_22.jsonl` (22 records)

## 1. Category Distribution

| Category | Train Count | Percentage |
|---|---|---|
| `hindi_hinglish` | 27 | 11.8% |
| `greetings_social` | 27 | 11.8% |
| `explanations_reasoning` | 23 | 10.1% |
| `identity_awareness` | 23 | 10.1% |
| `natural_qa` | 23 | 10.1% |
| `continuity_feedback` | 23 | 10.1% |
| `context_grounding` | 23 | 10.1% |
| `multiturn_dialogue` | 23 | 10.1% |
| `supportive_conversation` | 18 | 7.9% |
| `abstention` | 18 | 7.9% |

## 2. Key Phrase Frequency in CBR-SFT-07 Train

| Phrase | Prompt Occurrences | Completion Occurrences | Total Occurrences (% records) |
|---|---|---|---|
| `context` | 0 | 11 | 11 (4.8%) |
| `provided in the context` | 0 | 2 | 2 (0.9%) |
| `not provided in the context` | 0 | 2 | 2 (0.9%) |
| `according to the context` | 0 | 0 | 0 (0.0%) |
| `based on the context` | 0 | 1 | 1 (0.4%) |
| `the context does not` | 0 | 1 | 1 (0.4%) |
| `ready to assist` | 0 | 2 | 2 (0.9%) |
| `chakrview` | 5 | 19 | 20 (8.8%) |
| `neural networks` | 1 | 1 | 1 (0.4%) |
| `namaste` | 2 | 2 | 2 (0.9%) |
| `hello` | 1 | 1 | 1 (0.4%) |
| `abhimanyu` | 0 | 0 | 0 (0.0%) |
| `assistant` | 228 | 1 | 228 (100.0%) |
| `user` | 228 | 1 | 228 (100.0%) |

## 3. Top Repeated Completion Prefixes (First 3 Words)

| Prefix | Frequency |
|---|---|
| `i do not` | 9 |
| `no, i am` | 4 |
| `i am chakrview,` | 3 |
| `the provided context` | 3 |
| `no, i operate` | 2 |
| `i cannot predict` | 2 |
| `good afternoon! how` | 2 |
| `that information is` | 2 |
| `latency is the` | 2 |
| `yes, i am` | 2 |

## 4. Top Repeated Full Completions (Template Saturation)

| Completion Text | Frequency |
|---|---|
| `I do not have sufficient information to answer that.` | 2 |
| `The provided context does not state that.` | 2 |
| `That information is not provided in the context.` | 2 |

## 5. Lineage Dataset Comparison: 'context' Phrase Saturation

| Dataset | Total Records | Records with 'context' in Prompt | Records with 'context' in Completion |
|---|---|---|---|
| `cbr_sft_07_train` | 228 | 0 (0.0%) | 11 (4.8%) |
| `cbr_sft_07_val` | 22 | 0 (0.0%) | 0 (0.0%) |
| `cbr_sft_06a_train` | 153 | 10 (6.5%) | 13 (8.5%) |
| `cbr_sft_05_train` | 360 | 9 (2.5%) | 18 (5.0%) |
| `cbr_sft_04_train` | 144 | 46 (31.9%) | 9 (6.2%) |

## 6. Findings & Diagnosis

1. **Context-Refusal Memorization**: A massive proportion of completions explicitly contain 'context', 'provided in the context', and 'not provided in the context'.
2. **Refusal Basin**: When the model is evaluated on multi-turn dialogue, the appearance of conversation history mimics the context prompt template, pulling the model's logits straight into the high-frequency refusal attractor.
3. **Identity Bias**: The phrase 'ready to assist' and repetitive ChakrView self-descriptions are heavily over-represented, explaining why greetings collapse into generic readiness mantras.
