# MODEL CARD: ChakrView UCIP-04 Foundational Generation Candidate

## 1. Model Summary
- **Model Name:** ChakrView UCIP-04 Autoregressive Language Candidate
- **Version:** v0.13-candidate (UCIP-04)
- **Primary Micro Checkpoint:** `releases/ucip_04_candidate/chakrview_ucip04_model.pt`
- **Micro Checkpoint SHA-256:** `234d04d598b6ff2f6c4514d9b973c51233872c118ad97e8fb6e60dec79d68bd6`
- **Micro Architecture:** ChakrMicro (3,443,136 parameters, 6 layers, 192 d_model, 6 heads, 512 hidden_dim)
- **Experimental Medium Checkpoint:** `releases/ucip_04_candidate/chakrview_ucip04_medium_model.pt`
- **Medium Checkpoint SHA-256:** `694c5888479bbd0a90d1ca116787c14e2e5362a6da0dfe782ea7304a55bd0693`
- **Medium Architecture:** ChakrMedium (12,932,736 parameters, 7 layers, 384 d_model, 6 heads, 896 hidden_dim)

## 2. Intended Use & Provenance
- Built exclusively with indigenous training and zero external LLM dependencies.
- Replaces broken proverb attractors (`"The planned of the bold"`) with natural language representations.
- Supports both Neural-Only (`--memory off`) and Hybrid Memory (`--memory on`) modes.

## 3. Training & Validation
- **Corpus:** `experiments/cognition/ucip_04/data/ucip04_train.jsonl` (375 records, strictly 0% overlap with sealed evaluation).
- **Validation Loss:**
  - Micro (3.44M): **1.6592** (Token accuracy: 65.54%)
  - Medium (12.9M): **0.7110** (Token accuracy: 88.76%)
- Evaluated with greedy autoregressive rollouts at every 25 training steps.

## 4. Evaluated Capabilities & Behavioral Boundaries
- **Simple Sentence Generation:**
  - Micro: Achieves 70.0% on baseline diagnostic probes.
  - Medium: Achieves 86.7% on baseline diagnostic probes (`"A dog is a friendly domestic animal that loves to play and guard the home."`).
- **Explanation Generation:**
  - Medium generates coherent descriptions of computational components (`"A computer processor fetches instructions from memory, decodes their meaning..."`).
- **Known Limitations:**
  - Exact copying of long held-out arbitrary sentences remains capacity-bounded at 3.44M/12.9M.
  - In-context multi-turn pointer recall remains at 0.0% in pure neural mode; Hybrid Memory (`--memory on`) is required for reliable entity recall.
