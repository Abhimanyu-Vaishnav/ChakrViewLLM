# ChakrView Exp19 — Bounded 600-Step SFT Training Proposal (`exp19_debiased_sft_cpu_600`)

**Proposal Status**: `READY_TO_PROPOSE` (Training will NOT start without explicit user authorization)  
**Experiment Identifier**: `exp19_debiased_sft_cpu_600`  
**Base Checkpoint**: Fresh, isolated copy of canonical `training_checkpoint_step_10000.pt`  
**Execution Environment**: Pure PyTorch CPU (`torch.device("cpu")`)  
**Measured Hardware Timing**: **0.1396 sec/step** on local host  
**Estimated Total Execution Runtime**: **~83.8 seconds (~1.4 minutes)** for 600 steps  

---

## 1. Concrete Problem Addressal Matrix

| Diagnosed Failure in Exp18 | Concrete Root Cause | Proposed Mechanism in Exp19 |
| :--- | :--- | :--- |
| **77.5% Prefix Collapse** on `"The difference..."` | 41% of training targets and 100% of arithmetic/reference targets began with identical rigid `"The <noun>"` phrasing. | **Response Diversification**: In-memory rotation across 4 verified opening templates per category, reducing dominant prefix concentration from 100% to $\le 25\%$. |
| **0.0% Structured JSON Parsing** | Pre-training code priors (`import doctest`) overpowered small SFT signal at 300 steps. | **Extended Step Budget (600 steps)**: Increases gradient exposures from 600 to 1,200 examples, achieving full 1-epoch coverage of all 150 JSON examples. |
| **Sub-Threshold Task Accuracy (0.71%)** | Under-exposure: 300 steps at batch size 2 sampled only 50% of the training distribution (600/1200 examples). | **Full Distribution Coverage**: 600 steps ensures every training example is sampled on average $\ge 1$ time. |
| **Evaluator False Positives** | Evaluator previously matched `"The"` as target entity for unit questions. | **Evaluator Hardened**: Article-stripping and word-boundary rules verified in unit tests (17/17 passed) preventing spurious scoring. |

---

## 2. Training Configuration Specification

```json
{
  "experiment_id": "exp19_debiased_sft_cpu_600",
  "base_checkpoint": "experiments/overnight_runs/exp19_debiased_sft_cpu_600/checkpoints/training_checkpoint_start.pt",
  "steps": 600,
  "batch_size": 2,
  "max_seq_len": 512,
  "learning_rate": 5e-05,
  "min_learning_rate": 5e-06,
  "lr_schedule": "cosine_decay",
  "weight_decay": 0.01,
  "grad_clip": 1.0,
  "val_interval": 25,
  "save_interval": 50,
  "seed": 42,
  "device": "cpu",
  "tokenizer_dir": "data/experiments/vocab_4096",
  "save_best_checkpoint": true,
  "selection_metric": "min_validation_loss"
}
```

---

## 3. Strict Pre-Training Safety Checklist

Prior to launching any training for Exp19:
1. Verify source canonical checkpoint hash (`96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`).
2. Copy canonical checkpoint to isolated experiment directory; canonical checkpoint path is never written to.
3. No official Exp17 or Exp18 files created or overwritten.
4. Protected file `chakrview/runtime/session.py` remains untouched.
5. Post-training evaluation runs strictly against frozen 160 test records under identical decoding settings.
6. Verify all 9 protected asset hashes before and after execution.

---

## 4. Remaining Operational Risks & Mitigation

1. **Risk**: Model over-fits to training responses between step 300 and 600.  
   *Mitigation*: The engine evaluates held-out validation loss every 25 steps and automatically saves `sft_best_checkpoint.pt` at the global validation loss minimum, ensuring evaluations use the optimal generalization checkpoint even if step 600 begins over-fitting.
2. **Risk**: CPU process timeout or resource starvation.  
   *Mitigation*: Isolated timing test confirmed 20 steps took 2.79s (0.1396 s/step). Total runtime for 600 steps is under 2 minutes.

---

## 5. Experiment Gate Conclusion

Design, synthetic verification, and timing benchmarks are fully complete.

Gate Status: **READY_FOR_TRAINING** (Awaiting explicit user authorization).
