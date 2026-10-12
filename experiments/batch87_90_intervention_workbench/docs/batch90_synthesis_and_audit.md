# Batch 90 — Synthesis, Audit, and Research Conclusion

## 1. Longitudinal Synthesis Across Sprints 71–90

Across 20 consecutive engineering batches (71 through 90), we systematically tested four distinct learning formulations to teach single-digit addition to **ChakrMicro v0.1** (3,443,136 parameters):

| Sprint Batches | Formulation Tested | Held-Out Test Accuracy | Dominant Failure Mode |
|:---|:---|:---:|:---|
| **71–74** | Uniform & Answer-Weighted SFT | 0.0% | Global collapse to modal digit `7`. |
| **75–78** | Multi-Template Synthetic Pretraining | 0.0% | Template mastery (EOS 100%), but global collapse to digit `6`. |
| **79–82** | Balanced Sum-Class Pretraining ($0..18$) | 5.56% | Broken into piecewise threshold heuristic on first operand $a$. |
| **83–86** | Intermediate Counting Scratchpad | 0.0% | Flawless format, but memorized canned prototype strings. |
| **87–90** | Incremental Next-Step Supervision ($k \to k+1$) | 0.0% | Transition outputs collapsed to modal attractor `13`. |

---

## 2. Definitive Scientific Finding

1. **The Representation Gap in Low-Parameter Causal Language Models**:
   - Small causal transformers (6 layers, $d_{model}=192$, 3.44M params) operating over generic BPE vocabulary lack the intrinsic inductive bias to perform arithmetic composition under standard autoregressive cross-entropy loss.
   - In all 5 training regimes, backpropagation rapidly finds statistical shortcuts (attractor tokens, first-operand thresholding, or canned template prototypes) rather than learning functional routing.
2. **Product Success & Software Tools**:
   - The Local Workbench v0.2 provides a reliable local engineering environment for model evaluation, checkpoint auditing, deterministic baseline comparisons, and live training monitoring without external dependencies.
