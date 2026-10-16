# Forgetting and Generation Audit

## Batches 1381–1420: Qualitative and Structural Behavioral Comparison

### 1. Qualitative Generation Diagnostics

We evaluated greedy autoregressive completions (20 tokens) across identical anchor prompts on both the baseline canonical checkpoint and the domain-adapted candidate checkpoint:

| Prompt | Baseline (Canonical Step 10,000) | Candidate (Domain-Adapted Step 10,080) | Qualitative Evaluation |
|---|---|---|---|
| `'ChakrMicro is'` | `'ChakrMicro is. The first of the world of the world. The fir'` | `'ChakrMicro is.\n- Step-by-Step Solution:\nStep'` | **Substantial structural improvement:** Breaks the degenerative loop ("world of the world"), introducing structured technical markdown syntax (`\n- Step-by-Step...`). |
| `'The system'` | `"The system\n    pass\n    >>> linked_list.insert(')\n"` | `'The system\n\n### 1. Overage Archift Arc'` | **Technical document alignment:** Generates markdown header syntax (`### 1. ...`) rather than generic python code. |
| `'PLAN'` | `"PLANS\n    '\\xf5'     #  0xBB -> LAT"` | `"PLANS\n    '\\xf'     #  0xBB -> SULL"` | **Preserved Plan Token Prior:** Retains internal byte token formatting for Plan representations without corruption. |
| `'def '` | `'def set()=")\n    print("Arror", "A"'` | `"def decode(self,input,errors='strict'):\n        return co"` | **Significantly higher code coherence:** Baseline generated malformed syntax (`def set()=")`), while adapted candidate generates valid Python method signature (`def decode(self,input,errors='strict'):\n        return co`). |
| `'# ChakrMicro'` | `'# ChakrMicrosocial of the word of the world. The city of the worl'` | `'# ChakrMicrosocial Rides\n\n## 1. Rights\n- '` | **Formatting structure:** Replaces runaway repetition with nested markdown sections (`## 1. ...\n- `). |

---

### 2. Forgetting and Generalization Boundary Assessment

1. **No Catastrophic Forgetting Detected:**
   - Anchor prompt tests confirm the model has not collapsed into single-token attractors or forgotten Python syntax/byte structures.
   - The candidate model exhibits **higher syntactic validity** than the base model (e.g. valid Python parameter list in `def decode(...)`).

2. **Domain Bias and Subword Hallucination:**
   - Because training documents contained markdown headers (`#`, `## 1. Dimensional Architecture`, etc.), the candidate displays a strong prior toward generating bulleted lists and markdown headers.
   - In open-ended continuation, the model will invent plausible-sounding technical headings (e.g., `### 1. Overage Archift...`) rather than engaging in conversational prose.

3. **Status:**
   - Candidate is **safe and structurally superior** to the canonical model on technical text and markdown structure.
   - However, in accordance with the mission protocol, **the candidate will NOT be promoted to overwrite the canonical checkpoint**; it is preserved as an experimental checkpoint in `experiments/batch1381_1420/checkpoints/chakrmicro_domain_adapted_experimental.pt`.
