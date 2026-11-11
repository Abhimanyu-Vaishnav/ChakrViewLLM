# CR-16D Experimental Design & Hypothesis Formulation

### 1. Research Motivation
In CR-16C, fine-tuning the sequence-copying checkpoint (CR-16A Arm D) on key-value retrieval yielded strong retrieval accuracy (59.0%), but caused total catastrophic forgetting of sequence copying ($66\% \to 0\%$). Furthermore, CR-16C exhibited near-zero generalization on unseen keys and unseen values.

CR-16D investigates:
1. Whether multi-task rehearsal/replay during retrieval training can preserve or enhance copying capabilities.
2. Whether the model can learn abstract, compositional key-value routing (evaluating on a $2 \times 2$ factorial grid of seen/unseen keys and values).

### 2. Hypotheses
- **Hypothesis 1 (Catastrophic Forgetting Mitigation):**
  A non-zero replay ratio of sequence copying ($\ge 25\%$) will prevent catastrophic forgetting and maintain exact copying accuracy $\ge 50\%$.
- **Hypothesis 2 (Compositional Combination Generalization):**
  If retrieval represents a dynamic in-context routing operation rather than static key-value associative memorization, the model should achieve $>50\%$ retrieval accuracy on completely unseen pairings of familiar keys and values (Group G1).
- **Hypothesis 3 (OOD Key Generalization):**
  If routing uses positional binding rather than lexical matching on key tokens, unseen keys paired with familiar values (Group G3) will generalize at rates comparable to familiar keys.
- **Hypothesis 4 (OOD Value Generation Bottleneck):**
  Because the model lacks a dedicated pointer copy head, generating an unseen value token (Group G2 and G4) requires projecting hidden representations through a fixed unembedding matrix $W_U$. If the token was never exposed as a retrieval target, accuracy will drop significantly compared to familiar values.

### 3. Factorial Matrix
| Group | Key Status | Value Status | Hypothesized Accuracy Range |
|---|---|---|---|
| **G1** | Seen in Training | Seen in Training (Novel Pairing) | High ($>50\%$) |
| **G2** | Seen in Training | Unseen in Training | Low ($<15\%$) |
| **G3** | Unseen in Training | Seen in Training | High ($>50\%$) |
| **G4** | Unseen in Training | Unseen in Training | Low ($<15\%$) |

### 4. Controlled Execution Protocol
- Architecture: Frozen canonical ChakrMicro ($3.44\text{M}$ params, $d=192, L=6, H=6, V=4096$).
- Parent: CR-16A Arm D generalization checkpoint (`9d609a77...`).
- Seeds: 2 independent random seeds (42 and 101) per arm to verify reproducibility.
- Total Runs: 8 training runs (Arms A, B, C, D $\times$ Seeds 42, 101) + 2 frozen baselines.

