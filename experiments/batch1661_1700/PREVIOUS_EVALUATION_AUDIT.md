# PREVIOUS EVALUATION AUDIT
## Batches 1661–1700 Audit of Batch 1621–1660 Neural Evaluation Harness & Artifacts

### 1. Scope & Objective

This audit examines the scripts, datasets, execution paths, and metrics generated in Batches 1621–1660:
- `experiments/batch1621_1660/run_neural_experiment.py`
- `experiments/batch1621_1660/results/NEURAL_EVALUATION_RESULTS.json`
- `experiments/batch1621_1660/data/DATASET_MANIFEST.json`
- `experiments/batch1621_1660/data/test_held_out.json`
- `experiments/batch1621_1660/data/train.jsonl`
- `experiments/batch1621_1660/data/val.jsonl`

The audit evaluates whether the reported results were truthful, whether deterministic tool fallbacks were invoked, whether evaluation metrics were artificially permissive or rigorous, and whether leakage occurred.

---

### 2. Systematic Findings & Verification Checklist

| Audit Criterion | Verification Status | Evidence & Audit Findings |
| :--- | :---: | :--- |
| **1. Model and Tokenizer Identity** | **VERIFIED** | Loaded byte-level BPE tokenizer (`vocab_size=4096`) and `ChakrMicro` directly via PyTorch without external wrappers. |
| **2. Standalone Neural Isolation** | **VERIFIED** | Autoregressive generation calls `model(inp)[0, -1, :]` in a pure Python loop. Zero calls to `GovernedToolRouter`, `SafeCalculator`, `SandboxedPythonTool`, or `StructuredVerifier`. |
| **3. Generation Consistency** | **VERIFIED** | Greedy decoding (`torch.argmax(logits)`) with fixed sequence lengths (`max_new_tokens=20`) and EOS token ID (`1`) was uniformly applied across all 3 checkpoints. |
| **4. Arithmetic Evaluation Exactness** | **VERIFIED** | Arithmetic evaluation used exact integer matching via regex extraction: `is_pass = (extracted == expected)`. Since models produced lexical drift instead of numbers, accuracy was genuinely 0.0% (0/30). |
| **5. Compositional Evaluation Exactness** | **VERIFIED** | Checked substring match of exact target entity phrase: `is_pass = exp_target in gen_clean.lower()`. Accurate and strict. |
| **6. Language Continuation (30/30)** | **OVERLY PERMISSIVE** | The pass condition checked only: `len(words) >= 2 and len(set(words)) >= len(words) * 0.5`. Because the models repeatedly output `"in the world, which is the world of the world."`, this repetitive phrase easily passed the vocabulary variety heuristic. The 30/30 score represents grammatical fluency and absence of crash, but NOT diverse language generation. |
| **7. Factual Grounding (15/30)** | **EXPLAINED & AUDITED** | Track 5 comprised 15 taught facts and 15 unseen negative controls. All models scored 0/15 on taught facts (failing port recall) and 15/15 on negative controls (correctly not hallucinating port numbers). Total: 15/30 (50.0%). The score is genuine and reflects rejection of hallucination rather than factual knowledge. |
| **8. Partitioning & Leakage Prevention** | **VERIFIED** | Arithmetic extrapolation split strictly enforced $N \in [1, 20]$ in train/val and $N \in [25, 54]$ in held-out test. Compositional hops were isolated into independent premises. |
| **9. Loss & Perplexity Computations** | **VERIFIED** | Standard token-level cross-entropy loss over target tokens with proper denominator averaging. |
| **10. Architecture Discrepancy Note** | **RESOLVED** | The narrative report cited 1,586,816 parameters, but runtime loaded the true 3,443,136 parameter model. |

---

### 3. Machine-Readable Audit Log of Identified Discrepancies

```json
{
  "audit_version": "batch1661_1700_v1",
  "batch_audited": "batch1621_1660",
  "discrepancies": [
    {
      "id": "DISC-001",
      "severity": "DOCUMENTATION_ONLY",
      "field": "total_parameters",
      "reported_value": "1,586,816",
      "actual_runtime_value": "3,443,136",
      "resolution": "Proved weight-tying in ModelConfig(d_model=192, L=6). Code execution was unaffected because model was initialized from ckpt['config']."
    },
    {
      "id": "DISC-002",
      "severity": "METRIC_LIMITATION",
      "field": "track1_language_continuation_acceptance_criterion",
      "reported_score": "30/30 (100%)",
      "actual_behavior": "Models generated repetitive loops ('in the world, which is the world of...'). The metric threshold required only len(set(words)) >= 0.5 * len(words), allowing repetitive loops to pass.",
      "resolution": "Batch 1661–1700 benchmark replaces subjective continuation checks with strict JSON schema and intent parse validation."
    },
    {
      "id": "DISC-003",
      "severity": "BENCHMARK_DESIGN_CONFIRMED",
      "field": "track5_factual_recall_split",
      "reported_score": "15/30 (50%)",
      "actual_behavior": "0/15 on taught facts, 15/15 on negative controls. No factual learning occurred; model simply didn't emit 4-digit numbers.",
      "resolution": "Confirmed. Model has zero factual retention in weights without retrieval augmentation."
    }
  ]
}
```

### 4. Conclusion
The previous evaluation in Batch 1621–1660 was strictly neural and free of tool pollution. Its negative results (0% on arithmetic, 0% on code, 0% on multi-hop composition) were honest and accurate. The 100% on Track 1 was due to a permissive heuristic, but did not alter the fundamental finding: ChakrMicro alone cannot perform symbolic reasoning.
