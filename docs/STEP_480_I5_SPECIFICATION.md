# STEP 480: I5 Recursive Compositional Reasoning Specification & Benchmark

## 1. Primary Research Question
Can ChakrView perform genuine recursive multi-step compositional reasoning by repeatedly refining an internal cognitive state across multiple relational hops:
$$\text{state}_0 \xrightarrow{\text{Cycle 1}} \text{state}_1 \xrightarrow{\text{Cycle 2}} \text{state}_2 \xrightarrow{\text{Cycle 3}} \text{state}_3 \xrightarrow{\dots}$$
while preserving:
- Canonical baseline: **3,443,136 parameters**, SHA-256 `c5571c...`, $\Delta W = 0$.
- I4 relational capability ($G4 \ge 77.78\%$, $H1=100\%$, $H2=88.89\%$).
- Learned neural gateway ($74,016$ parameters).
- Modular memory, adaptive computation, CPU-first low-resource execution.

---

## 2. Core I5 Benchmark Hierarchy
The I5 benchmark establishes an explicit capability ladder across 4 hop depths:
1. **I4 (1-Hop)**: Direct relation $A \xrightarrow{r_1} B$.
2. **I5-2 (2-Hop)**: Composition $A \xrightarrow{r_1} B \xrightarrow{r_2} C$. Target: Find $C$ from $A$.
3. **I5-3 (3-Hop)**: Chain $A \xrightarrow{r_1} B \xrightarrow{r_2} C \xrightarrow{r_3} D$. Target: Find $D$ from $A$.
4. **I5-4 (4-Hop)**: Long-horizon chain $A \xrightarrow{r_1} B \xrightarrow{r_2} C \xrightarrow{r_3} D \xrightarrow{r_4} E$. Target: Find $E$ from $A$.

---

## 3. Controlled Evaluation Splits & Adversarial Stress Conditions
- **Splits**:
  - `TRAIN`: Base relational chains on standard entity set.
  - `VALIDATION`: In-distribution validation chains.
  - `HELDOUT`: Unseen relation permutations and distractor insertions.
  - `STRICT GENERALIZATION`: Completely unseen entity sets (e.g. noble gases, synthetic tokens), premise reordering, and relation inversions.
- **Controlled Invariance Tests**:
  - Relation Inversion ($r^{-1}$).
  - Distractor Entities ($D$ inserted with unrelated relation).
  - Premise Shuffling ($B \to C$ stated before $A \to B$).
  - Unknown Rejection (`UNKNOWN`).
  - Contradiction Handling (`CONFLICTING`).

---

## 4. Distinction Metric: Neural State Dependence vs Repeated Lookup
- **Repeated Symbolic Lookup**: Simply queries memory repeatedly without internal state dependence.
- **Recursive Neural State Refinement**: Cycle $k$ strictly requires $s_{k-1}$ produced by Cycle $k-1$. If $s_{k-1}$ is zeroed, shuffled, or detached, accuracy on hop $k$ must degrade significantly.
