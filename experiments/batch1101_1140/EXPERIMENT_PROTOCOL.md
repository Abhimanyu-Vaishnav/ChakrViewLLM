# Experiment Protocol — Batches 1101–1140
## Learned Semantic Retrieval, Evidence-Precise Answers & Neural Capability Audit

### 1. Protocol Objective
To scientifically establish:
1. Whether the indigenous learned bi-encoder (`ChakrBiEncoder`) improves retrieval over deterministic hash-projection and pure BM25 baselines on semantic/paraphrase queries.
2. Whether extractive concise span selection (1–2 sentences / ~30 words) provides verifiable citation grounding and compression without sacrificing target factual retention.
3. The exact causal attribution across neural, deterministic, and heuristic components in the local knowledge assistant pipeline.

### 2. Frozen Datasets & Provenance
| Dataset | Location | SHA-256 Hash | Item Count | Categories |
| :--- | :--- | :--- | :--- | :--- |
| **Development Corpus** | `experiments/batch1021_1060/data/sample_corpus` | Verified in Manifest | 5 docs (14 chunks) | Technical specifications, billing SLA, incident SOP, security policy, release notes |
| **Held-Out Generalization Suite** | `experiments/batch1061_1100/data/generalization_eval.json` | `267309e6547129dac3caa77918caa09a56c978934cebfdcdd6f0ebf547e92405` | 60 items | Paraphrase (10), Synonyms (8), Negation (8), Multi-doc (8), Follow-up (10), Abstention (10), False Premise (6) |
| **Canonical Checkpoint** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | N/A | Untouched, strictly preserved |
| **Indigenous Bi-Encoder Checkpoint** | `experiments/batch171_180_pointer_learning/checkpoints/improved_bi_encoder.pt` | `d55d8272f2d93e5066a34c8a2bda0728c89c8913ffad48ec436a5faeb30d43e2` | N/A | Indigenous 2-layer transformer bi-encoder (`d_model=128`, `d_embed=64`) |

### 3. Evaluation Methodology & Metric Fixes
Prior evaluations suffered from a lenient substring check (`expected_substring in response.answer`), which rewarded whole 800-character chunk dumps where small tokens (e.g. single digits like `"6"`) could match accidentally.

The new protocol enforces:
- **Exact Normalized Answer Match (EM)**.
- **Token-Level Precision, Recall, and F1**.
- **Explicit Numeric and Unit Matching** (prevents `"1536"` from satisfying `"6"`).
- **Decoupled Citation Validity vs. Citation Support**:
  - Citation Validity: Byte offsets $[start\_char : end\_char]$ slice the disk file exactly matching `exact_text`.
  - Citation Support: The cited span contains the factual target evidence.
- **Decoupled Correct Abstention vs. Incorrect Abstention**.
