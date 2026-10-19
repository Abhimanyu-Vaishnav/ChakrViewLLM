# SFT Comparison Report: Batch 2861–2900
## Canonical Step-10,000 Baseline vs. Authorized 100-Step SFT Candidate

### 1. Executive Summary

Under explicit user authorization, a 100-step Supervised Fine-Tuning (SFT) pilot was executed on CPU using the curated 192-example training dataset:
- **Parent Checkpoint**: Canonical Step 10,000 (`training_checkpoint_step_10000.pt`, SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`).
- **Candidate Checkpoint**: `experiments/batch2821_2860/sft_pilot_run/sft_checkpoint_step_100.pt` (SHA-256: `f911d081c8a6fbdde0e21322c15bb56d2db6f84593c47ea5e280dbfae2b574b1`).
- **Parameter Digest**: `d084b6f66d0a867a47c3a107f577e80a886e33a5557e0c9663bd25c6d4a1cee8`.
- **Training Progression**: Validation loss decreased from `4.9390` (Perplexity 139.6) to `3.9569` (Perplexity 52.3), with a valley at step 40 (`3.9523`, Perplexity 52.06).
- **Execution Invariants**: Zero canonical artifacts were modified; `session.py` remained untouched; training was isolated strictly to `experiments/batch2821_2860/sft_pilot_run/`.

---

### 2. Quantitative Performance Comparison

#### Held-Out Test Set (24 Cases across 8 Categories)

| Task Category | Canonical Baseline (Step 10,000) | SFT Candidate (Step 100) | Observed Change / Behavior |
|---|---|---|---|
| `simple_instruction` | 0/3 exact | 0/3 exact | Baseline output Wikipedia table fragments; SFT output structured "The result of..." phrases. |
| `context_extraction` | 0/3 exact | 0/3 exact | Neither model isolated the exact numerical/name entity from context. |
| `context_grounded_qa` | 0/3 grounded | 0/3 grounded | Both models fail multi-hop comprehension on 512 context tokens. |
| `short_summarization` | 0/3 concise | 0/3 concise | Both models output generic continuation patterns. |
| `structured_output` | 0/3 valid JSON (0%) | **2/3 valid JSON (66.7%)** | **Improvement**: Candidate generated valid parseable JSON array `[0]` on 2 cases. |
| `uncertainty_insufficient_info`| 0/3 abstention (100% hallucination) | 0/3 abstention (0% honest refusal) | Candidate continues text rather than refusing; 100 steps on 24 refusal train examples was insufficient to enforce refusal attractors. |
| `conversational_acknowledgement`| 0/3 natural response | 0/3 natural response | Both models fail natural conversational dialogue. |
| `bounded_text_transformation`| 0/3 accurate transform | 0/3 accurate transform | Token manipulations (uppercase/reverse) require deeper algorithmic capability. |

---

### 3. Regression Probes on Batch 2781 Generation Benchmark (28 Cases)

| Benchmark Category | Canonical Baseline | SFT Candidate (`raw_completion`) | SFT Candidate (`plain_instruction`) |
|---|---|---|---|
| `raw_continuation` (4 cases) | Generated text on all 4 cases | Smooth continuation preserved | Smooth continuation |
| `repetition_probe` (4 cases) | Severe repetition loops without penalty | Repetition loops | Repetition loops |
| Stop on EOS Behavior | Rarely stopped on EOS | Stopped on EOS on several cases | Stopped on EOS on several cases |
| Catastrophic Forgetting Probe | Intact baseline | Model weights remain close to parent; no NaN/Inf | Code syntax (`def calculate_mean`) was partially preserved |

**Important Finding on `raw_completion` mode**:
Because SFT exclusively supervised pairs formatted as `Instruction: {query}\nResponse: {answer}\n<EOS>`, the SFT candidate learned an internal transition from terminal prompt tokens directly to `<EOS>` when presented with raw unformatted queries. When prompted without the `Instruction:` wrapper, the model emitted token 1 (`<EOS>`) immediately, resulting in 0 generated tokens for 15 out of 28 raw queries. Conversely, under `plain_instruction`, the model generated non-empty text across all 28 cases.

---

### 4. Key Takeaways & Recommendations

1. **Loss Reduction Does Not Equal General Intelligence**:
   Although training cross-entropy loss dropped from ~4.9 to ~3.7 and validation loss decreased to 3.95, a 100-step pilot on 192 examples is insufficient to establish comprehensive instruction following or context-grounded reasoning in a 2.5M parameter micro-model.
2. **Emergence of Structured Syntactic Priors**:
   The candidate began learning structured JSON formatting constraints (achieving 66.7% valid JSON parsing compared to 0% in the baseline).
3. **Runtime Policy**:
   The canonical Step-10,000 checkpoint **must remain the default model**. The SFT candidate should remain an experimental artifact accessible only through explicit checkpoint selection (`--neural-checkpoint`).
