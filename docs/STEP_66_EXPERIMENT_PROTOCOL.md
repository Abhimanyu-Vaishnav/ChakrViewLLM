# Step 66 Experiment Protocol: Episodic Memory Recall Loop & Validity Verification

## 1. Experimental Conditions

The benchmark script `scripts/experiment_step66_episodic_recall.py` validates 15 empirical conditions:

| Condition | Target Property | Expected Result |
|---|---|---|
| **A** | Exact task-family recall | Successfully recalls positive record matching task family |
| **B** | Same-module recall | Successfully recalls record matching target module across tasks |
| **C** | Structural-pattern recall | Computes positive token Jaccard pattern similarity |
| **D** | Irrelevant memory rejection | Unrelated task domains return 0 positive memories |
| **E** | Stale-memory rejection | Module structural drift triggers `REJECTED_STALE` |
| **F** | Superseded-memory rejection | Inactive versions classified as `REJECTED_SUPERSEDED` |
| **G** | Negative-boundary recall | Negative failure memory recalled as `NEGATIVE_BOUNDARY` |
| **H** | Positive-vs-negative conflict | Positive candidate targeting forbidden module marked `CONFLICTED` |
| **I** | Multiple-memory deterministic ordering | Top candidates sorted deterministically by score |
| **J** | Recall budget enforcement | Rejects memories exceeding `max_recalled_memories` budget |
| **K** | Repository fingerprint mismatch | Preserves evidence provenance under fingerprint divergence |
| **L** | Repeated-run determinism | Identical queries yield bit-exact identical recalled context |
| **M** | Neural-boundary immutability | Pre- and post-run weights match bit-exact ($\Delta W = 0$) |
| **N** | Empty-memory abstention | Returns empty guidance cleanly on empty memory index |
| **O** | Cross-domain negative-transfer defense | Mismatched domain tags trigger fail-closed `ABSTAIN` |

---

## 2. Benchmark Command
```powershell
.venv\Scripts\python.exe scripts/experiment_step66_episodic_recall.py
```
Output results are written to `artifacts/step66/step66_episodic_recall_evidence.json`.
