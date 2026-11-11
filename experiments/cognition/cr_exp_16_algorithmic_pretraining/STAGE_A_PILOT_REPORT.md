# CR-EXP-16 Stage A Pilot Report
## Isolated Pilot Training Results and Guardrail Checkpoint

**Mission Identifier:** CR-EXP-16  
**Initial Checkpoint:** Canonical Stage B Step 10,000 (`experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`)  
**Initial Checkpoint SHA-256:** `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`  
**Pilot Checkpoint Path:** `experiments/cognition/cr_exp_16_algorithmic_pretraining/checkpoints/pilot/stage_a_pilot_best.pt`  

---

## 1. Pilot Training Parameters & Loss Progression

- **Architecture:** Canonical ChakrMicro ($d_{\text{model}}=192, 6\text{ layers}, 6\text{ heads}, d_{\text{head}}=32, 3.44\text{M params}$)
- **Training Steps:** 100 steps
- **Batch Size:** 8 (effective sequence length 128)
- **Optimizer:** AdamW ($lr=5 \times 10^{-5}, \text{weight decay}=0.01$)
- **Loss Function:** Cross-entropy with token masking on pad (`ignore_index=0`)

### Loss Progression Curve
- **Step 0 (Initial Val Loss):** $4.6302$
- **Step 20:** Train Loss = $2.9474$, Val Loss = $3.1391$
- **Step 40:** Train Loss = $2.4904$, Val Loss = $2.5154$
- **Step 60:** Train Loss = $2.0596$, Val Loss = $2.1994$
- **Step 80:** Train Loss = $1.9248$, Val Loss = $2.0100$
- **Step 100:** Train Loss = $1.8939$, Val Loss = **$1.8828$** ($\Delta = -2.7474$, 59.3% reduction)

---

## 2. Guardrail Evaluation on Benchmark Copying Probes

The pilot model was evaluated across 300 copying probes from the frozen benchmark:

| Category | Probes ($N$) | Exact Accuracy | Token Presence Rate |
|---|---|---|---|
| **Exact Sequence Copying** | 100 | **0.00%** (0/100) | **0.00%** (0/100) |
| **Single-Token Copying** | 100 | **0.00%** (0/100) | **0.00%** (0/100) |
| **Multi-Token Copying** | 100 | **0.00%** (0/100) | **0.00%** (0/100) |
| **Total Copying Pilot** | **300** | **0.00%** (0/300) | **0.00%** (0/300) |

---

## 3. Forensic Analysis of Pilot Output: Repetitive Mode Collapse

Qualitative analysis of the pilot model reveals a **critical training guardrail trigger**:
1. **Loss Reduction via Mode Collapse:** The loss dropped from $4.63 \to 1.88$ not because the model learned to attend backwards and copy prompt tokens, but because it collapsed into repeating high-frequency single tokens:
   - Prompt: `Repeat: red two steel\nOutput: ` $\to$ Output: `'stone stone stone stone stone '`
   - Prompt: `Echo sequence: banana one\nOutput: ` $\to$ Output: `'north stone stone stone ston'`
   - Prompt: `Input: apple cat\nCopy: ` $\to$ Output: `'storm storm storm storm storm '`
2. **Mechanistic Explanation:** In standard causal language modeling where loss is computed over the entire sequence, predicting static unigrams minimizes cross-entropy faster than coordinating 2-layer induction circuits.
3. **Guardrail Enforced:** In accordance with Section C of the user directive:
   > *"If the pilot shows collapse, repeated-token output, or no measurable improvement, stop and investigate instead of automatically scaling training."*
   > *"Present the pilot results and wait for review before expanding the training budget."*

Training budget expansion is **PAUSED**. The pilot results are presented below for scientific investigation and direction.

