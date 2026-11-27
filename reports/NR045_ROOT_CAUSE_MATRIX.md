# NR-045: Root Cause Classification Matrix

## 1. Executive Summary

This matrix categorizes all identified defects contributing to the generation incoherence of **ChakrView 1.0 RC1**, ranking them by severity, confidence, affected component, and empirical evidence.

---

## 2. Root Cause Classification Matrix

| Category ID | Defect Classification | Severity | Confidence | Affected Component | Empirical Evidence & Manifestation | Suggested Corrective Action |
| :---: | :--- | :---: | :---: | :--- | :--- | :--- |
| **RC-01** | **Dataset Quality & Formatting Defect** | **CRITICAL** | **HIGH (100%)** | `data/corpus_v1/` | **82.8% of training samples** contained duplicate prompt framing (`<user>\nUser: ...\nAssistant: \n<assistant>`), causing out-of-distribution prefix mismatch at inference time. | Clean dataset by stripping inner `"User: ...\nAssistant: "` tags and standardizing to pure `<user>\n{text}\n<assistant>\n`. |
| **RC-02** | **Training Objective Defect** | **CRITICAL** | **HIGH (100%)** | `train_conversational_sft.py` | Training loss unmasked on prompt tokens; over 60% of gradient updates optimized prompt prediction rather than response generation. | Mask prompt tokens with `-100` (`labels = [-100]*len(prompt) + response`) during SFT. |
| **RC-03** | **Dataset Composition Defect** | **HIGH** | **HIGH (100%)** | `master_corpus_train.jsonl` | **19.3% (1,126 samples)** were 1-2 word token echo tasks (`COPY`); average response length was only 12.5 words with high synthetic boilerplate. | Filter out repetitive `COPY` tasks; balance corpus with multi-sentence reasoning and natural dialogues. |
| **RC-04** | **Inference Decoding Defect** | **HIGH** | **HIGH (100%)** | `interactive_chat.py` / `brain_runtime.py` | Argmax greedy decoding ($T=0.0$) without repetition penalty causes low-entropy loops (`"The sum of..."`); sampling ($T=0.7$) restores coherent generation. | Default inference runtime to $T=0.7$ with top-p=0.9 and repetition penalty 1.15. |
| **RC-05** | **Architecture Context Limit** | **MEDIUM** | **MEDIUM (80%)** | `ChakrGenesis-17M` | Static RoPE cache limited to 256 sequence length caused dimension mismatch on long contexts. | Ensure RoPE embedding dynamic length interpolation is enabled across all inference paths. |
| **RC-06** | **Tokenizer Defect** | **NONE (CLEARED)**| **HIGH (100%)** | `vocab_4096` | 100% round-trip fidelity verified across all test cases. Tokenizer is NOT a source of defect. | Retain canonical Byte-Level BPE 4096 tokenizer without alteration. |

---

## 3. Salvageability Assessment of RC1 Checkpoint
- **Verdict:** **RC1 CHECKPOINT IS PARTIALLY SALVAGEABLE WITH SAMPLING, BUT CANNOT ACHIEVE FULL CONVERSATIONAL STABILITY WITHOUT SFT RETRAINING.**
- **Evidence:** Adjusting inference temperature to $T=0.7$ with prompt-template alignment immediately recovers coherent responses for Python coding (`add(a, b)`), planet names (*"Mars is called the Red Planet"*), and simple logic. However, because 60% of training loss was spent predicting unmasked prompt tokens and 19.3% of the dataset was single-token copying, the weights will remain prone to drift until retrained with clean response-only loss masking in NR-046.
