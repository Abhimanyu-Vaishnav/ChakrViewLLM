# STEP 473: Neural Cognitive Routing Gateway Interface Specification

## Purpose
Define the exact formal tensor interface connecting ChakrView's language representations (ChakrMicro v0.1 / Language Candidate) with the relational reasoning core (`UnifiedCompositionalCore`) and cognitive episodic memory, adhering to strict modularity and zero parameter contamination of the canonical baseline.

---

## 1. Concrete System Dimensionality & Boundaries

From active code inspection:
- **Language Model (`ChakrMicro`)**:
  - Vocabulary Size: `4,096`
  - Model Dimension ($d_{lang}$): `192`
  - Layers: `6`, Heads: `6` ($d_{head} = 32$)
  - Output representation: Hidden state $H_{lang} \in \mathbb{R}^{B \times T \times 192}$
  - Status: Canonical baseline (3,443,136 parameters) remains frozen. Candidate branches isolated.

- **Relational Cognitive Core (`UnifiedCompositionalCore`)**:
  - Model Dimension ($d_{cog}$): `96`
  - State Dimension ($d_{state}$): `48`
  - Binding Dimension ($d_{bind}$): `32`
  - Heads: `4` ($d_{head} = 24$)
  - Input representation expected: $X_{cog} \in \mathbb{R}^{B \times T \times 96}$
  - Second-hop query representation: $q_2 \in \mathbb{R}^{B \times T \times 96}$
  - Relational output representation: $R_2 \in \mathbb{R}^{B \times 96}$ or hidden $H_{cog} \in \mathbb{R}^{B \times T \times 96}$
  - Status: Modular cognitive core intact (69,809 parameters).

- **Cognitive Memory Store**:
  - Decoupled key-value epistemic graph: $(Entity_1, Relation) \to (Entity_2, Status)$
  - Vector index representation: $e \in \mathbb{R}^{96}$ (matching $d_{cog}$)
  - Epistemic statuses: `KNOWN`, `SUPPORTED`, `UNCERTAIN`, `CONFLICTING`, `UNKNOWN`.

---

## 2. Neural Cognitive Routing Gateway Architecture

The gateway consists of two small differentiable projections with residual gating:
1. **Forward Gateway (Language $\to$ Cognition)**:
   - Projects language hidden states $H_{lang} \in \mathbb{R}^{B \times T \times 192}$ to cognitive inputs $X_{cog} \in \mathbb{R}^{B \times T \times 96}$.
   - Linear projection: $W_{down} \in \mathbb{R}^{192 \times 96}$ (18,432 parameters).
   - Gating projection: $W_{gate} \in \mathbb{R}^{192 \times 96}$ (18,432 parameters).
   - Pre-layer normalization: `RMSNorm(192)`.
   - Output: $X_{cog} = \sigma(H_{lang} W_{gate}) \odot (H_{lang} W_{down}) \in \mathbb{R}^{B \times T \times 96}$.

2. **Reverse Gateway (Cognition $\to$ Language Verbalization)**:
   - Projects cognitive representations $R_{cog} \in \mathbb{R}^{B \times 96}$ (or $H_{cog}$) to language space $H_{lang}^{rec} \in \mathbb{R}^{B \times 192}$.
   - Linear projection: $W_{up} \in \mathbb{R}^{96 \times 192}$ (18,432 parameters).
   - Gating projection: $W_{rev\_gate} \in \mathbb{R}^{96 \times 192}$ (18,432 parameters).
   - Pre-layer normalization: `RMSNorm(96)`.
   - Output: $H_{lang}^{rec} = \sigma(R_{cog} W_{rev\_gate}) \odot (R_{cog} W_{up}) \in \mathbb{R}^{B \times 192}$.

3. **Total Gateway Parameter Budget**:
   - $18,432 \times 4 = 73,728$ weights.
   - RMSNorm scales: $192 + 96 = 288$ weights.
   - Total Gateway Parameters: **74,016 parameters** (~2.1% of baseline, well within low-resource CPU budget).

---

## 3. Trainability & Gradient Flow Invariants
- **Frozen Canonical Baseline**: `ChakrMicro` baseline weights remain strictly frozen ($\Delta W_{base} = 0$, SHA-256 intact).
- **Candidate Isolation**: Gateway weights are saved into isolated checkpoints (`artifacts/waves_473_479_execution/`).
- **Gradient Flow**: Gradients flow end-to-end through the Gateway during training, without requiring backprop into the frozen baseline when evaluating in adapter mode.
