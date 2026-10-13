# Capability and Limitations Report: Batch 147–150

**Date:** 2026-10-13  
**Milestone:** Batch 147–150 Context Binding, Grounded Comprehension & Capability Growth  
**Model:** ChakrMicro (3,443,136 parameters, 6 layers, $d_{model}=192$)  

---

## 1. Measured Capabilities

| Dimension | Baseline (Arm A) | Arm B (Direct SFT) | Arm C (Warmup SFT) | Arm D (Conservative) | Arm E (Deterministic) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Val Binding Loss** | 6.2976 | **1.3162** | 1.7435 | 2.9803 | N/A |
| **Test Binding Loss** | 6.0906 | **1.4706** | 1.8777 | 3.0710 | N/A |
| **Exact Match Rate (Held-Out)**| 0.0% (0/8) | 0.0% (0/8) | 0.0% (0/8) | 0.0% (0/8) | **100.0% (8/8)** |
| **Stage C Language Loss** | **3.3053** | 3.3558 ($+0.0505$) | 3.3296 ($+0.0243$) | 3.3135 ($+0.0082$) | N/A |
| **ChatML Format Compliance** | 0% (Wikitext tables) | **100%** | **100%** | 87.5% | N/A |
| **Clean `<|im_end|>` Termination**| 0% (0/8) | **100% (8/8)** | 62.5% (5/8) | 0% (0/8) | N/A |

### Empirical Insights:
1. **Convergence on Binding Loss:**
   Direct SFT reduced held-out test cross-entropy from $6.0906 \to 1.4706$ (PPL $441.68 \to 4.35$). The neural core has successfully converged on the conditional probability distribution of the key-value task.
2. **Termination Mastery:**
   Arm B achieved **100% clean termination** with `<|im_end|>` at the exact expected token boundary, overcoming the repetitive looping of earlier pretraining runs.
3. **The Extractive Span Gap:**
   Despite low test loss ($1.47$ nats), the model's greedy argmax path does not reliably copy the exact unseen value token from the context. Instead, it generates high-frequency tokens from the training set (e.g. `"Monday"`, `"elements"`).
4. **Deterministic Superiority for Slot Extraction:**
   The deterministic extractor (Arm E) achieved **100% exact match** across direct keys, distractor noise, and negative abstention.

---

## 2. Fundamental Architectural Limitations Identified

1. **Parameter Capacity Constraints (3.44M parameters):**
   A 3.44M parameter model lacks dedicated induction heads with sufficient capacity to perform generic arbitrary-token copying across unseen vocabularies without explicit copy-attention mechanisms (such as pointer networks or larger parameter width).
2. **Short-Context Memory Retention:**
   Fine-tuning on small structured datasets shifts greedy attractors towards the most frequent training answers unless regularization or rehearsal is introduced.
3. **No Automatic Promotion:**
   In adherence to project rules, the candidate SFT checkpoints are preserved in `experiments/batch147_150_context_binding/checkpoints/` as experimental artifacts. The canonical baseline `training_checkpoint_step_10000.pt` remains frozen and unpromoted.
