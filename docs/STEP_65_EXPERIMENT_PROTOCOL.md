# Step 65 Experiment Protocol: Empirical Evaluation of Active Episodic Learning

## 1. Experimental Conditions

The benchmark script `scripts/experiment_step65_episodic_learning.py` validates the episodic learning loop under 10 controlled conditions:

| Condition | Target Property | Expected Result |
|---|---|---|
| **A** | Verified success episode admission | Admitted as positive `RepositorySemanticRecord` with `successful_episodes = 1` |
| **B** | Unverified neural proposal admission | `REJECTED` by admission gate; zero positive memory created |
| **C** | Hallucinated proposal admission | `REJECTED` from positive memory |
| **D** | Failed execution admission | `REJECTED` from positive memory |
| **E** | Verified negative experience recording | Admitted as `NEGATIVE_BOUNDARY` with `solution_pattern = 'DO_NOT_APPLY'` |
| **F** | Stale memory detection | Safety gate detects structural mutation and abstains (`ABSTAIN`) |
| **G** | Deterministic superseding | Newer verified record supersedes older record; older marked inactive with audit links |
| **H** | Provenance preservation | Admitted records preserve links back to source episode IDs |
| **I** | Determinism & repeatability | Identical learning inputs generate bit-exact identical index representations |
| **J** | Neural core immutability | Pre- and post-benchmark ChakrMicro parameter count and weight hash match bit-exact ($\Delta W = 0$) |

---

## 2. Execution Command
```powershell
.venv\Scripts\python.exe scripts/experiment_step65_episodic_learning.py
```
Output results are written to `artifacts/step65/step65_episodic_learning_evidence.json`.
