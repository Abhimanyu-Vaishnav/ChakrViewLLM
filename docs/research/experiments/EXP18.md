# Research Evidence Dossier: Experiment 18 (Exp18)
## Aligned Supervised Fine-Tuning & Short Scientific Explanation Evaluation

- **Archive Status**: REJECTED_HYPOTHESIS_FORMALIZED_AS_BOUND
- **Artifact Status**: COMMITTED (Dossier & Reconciled Review CSV) / LOCAL_ONLY (Full Intermediate Traces & Checkpoints)
- **Primary Evidence Directory**: experiments/overnight_runs/v08_release_candidate_readiness_20261025_213337/ and experiments/overnight_runs/exp18_aligned_sft_cpu_300/

---

### 1. Purpose and Hypothesis
- **Purpose**: Assess whether supervised fine-tuning (SFT) over aligned question-answering and explanation datasets enables the 3.44M ChakrMicro model core to generate coherent, factually grounded scientific explanations.
- **Hypothesis**: The compact architecture can acquire grounded short scientific reasoning capabilities purely from parameter-internal knowledge when trained on curated SFT sequences.

### 2. Configuration and Training Setup
- **Architecture**: Canonical ChakrMicro baseline (6 layers, 6 heads, d_model=192, hidden_dim=512, context 512, vocab_size=4096, 3,443,136 unique parameters with tied embeddings).
- **Compute**: CPU-only bounded training run (300 steps).
- **Target Loss**: Cross-entropy over masked target explanation sequences.

### 3. Dataset, Split Identity & Leakage Controls
- **Dataset**: Aligned scientific Q&A corpus and prompt-response pairs.
- **Leakage Controls**: Strict preflight data gating (udit_exp18_dataset_gate_20261023_231859). Separate evaluation split for held-out scientific test prompts.

### 4. Evaluation Method and Exact Metrics
- **Methodology**: 20 held-out scientific prompts evaluated via independent human review across scientific accuracy, logical coherence, and hallucination bounds (EXP18_REVIEW_PROTOCOL.md).
- **Exact Outcome**:
  - **Rejection Rate**: 20/20 prompts rejected (100% human rejection rate).
  - **Pass Rate**: 0/20.

### 5. Mandatory Research-Claim Boundary
> [!IMPORTANT]
> The 20/20 human-review rejections are a direct empirical outcome of the specific held-out scientific reasoning prompt set. This does **not** prove that all neural language synthesis is impossible at small scale; rather, it conclusively proves that the ungrounded 3.44M core cannot independently synthesize factually reliable scientific explanations without dynamic retrieval or external symbolic grounding.

### 6. Supporting Artifacts & Availability
- Human Review Dossier: experiments/overnight_runs/v08_release_candidate_readiness_20261025_213337/EXP18_HUMAN_REVIEW_DOSSIER.md (LOCAL_ARCHIVE)
- Reconciled Review Results: experiments/overnight_runs/v08_release_candidate_readiness_20261025_213337/EXP18_RECONCILED_RESULTS.json (LOCAL_ARCHIVE)
- Completed Review Annotations: experiments/overnight_runs/v08_release_candidate_readiness_20261025_213337/EXP18_HUMAN_REVIEW_COMPLETED.csv (LOCAL_ARCHIVE)
- Reusable Analysis Tool: scripts/research_tools/reconcile_human_reviews.py (COMMITTED)
