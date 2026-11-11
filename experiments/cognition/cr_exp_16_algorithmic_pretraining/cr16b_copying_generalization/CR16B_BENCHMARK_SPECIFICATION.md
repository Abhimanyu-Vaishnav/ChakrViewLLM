# CR-EXP-16B Benchmark Specification

## 1. Objective and Architecture of the Benchmark
The CR-EXP-16B benchmark was designed to stress-test sequence copying capabilities across multiple orthogonal axes: token familiarity, permutation ordering, repetition, sequence length, subword fragmentation, distractor noise, and prompt template shift.

It comprises exactly **1,500 probes** across 14 balanced categories (CAT_A through CAT_N).

## 2. Category Taxonomy & Distribution

| Code | Category Name | Probe Count | Target Characteristics | Challenge Tested |
| :--- | :--- | :---: | :--- | :--- |
| **CAT_A** | Familiar Token, Unseen Combos | 110 | Greek letters, $L=3\dots 5$ | Compositional permutation generalization |
| **CAT_B** | Unseen Sequence Orderings | 110 | Common nouns, $L=4$ | Order-invariance under vocabulary shift |
| **CAT_C** | Repeated Token Sequences | 100 | Nouns with identical repeat tokens ($L=3\dots 4$) | Pointer attention disambiguation |
| **CAT_D** | Short Sequences | 110 | Length $1\dots 3$ tokens | Base copy fidelity |
| **CAT_E** | Medium Sequences | 110 | Length $4\dots 6$ tokens | Multi-step pointer retention |
| **CAT_F** | Long Sequences | 110 | Length $7\dots 10$ tokens | Working-memory horizon boundary |
| **CAT_G** | Single-Token Name Copying | 110 | Single-word Indian/Western names | Single-item entity extraction |
| **CAT_H** | Multi-Token Name Copying | 110 | Multi-subword names ($3\dots 5$ tokens) | Subword reconstruction against priors |
| **CAT_I** | Ordinary Word Copying | 110 | Common nouns/verbs ($L=1$) | Pure zero-shot dictionary retrieval |
| **CAT_J** | Mixed Alphanumeric Copying | 100 | Token combinations (e.g. `A1 B2 C3`) | Out-of-distribution character tokens |
| **CAT_K** | Unseen Prompt Templates | 110 | Prompts like `Mirror: {s} | Result: ` | Zero-shot instruction transfer |
| **CAT_L** | Familiar Prompt Templates | 110 | Prompts like `Echo sequence: {s}\nOutput: ` | In-distribution template control |
| **CAT_M** | Distractor-Containing Copying | 100 | Prompts with `Ignore notice: [...]` | Attention selectivity and noise filtering |
| **CAT_N** | Exact Sequence Reconstruction | 100 | Multi-turn conversational framing | Format transfer to conversational turns |

## 3. Benchmark Freeze & Integrity
- File: `experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16b_copying_generalization/data/cr16b_sealed_benchmark_1500.json`
- Total Probes: 1,500
- SHA-256 Hash: `44aba606abfee0beac5b129eb7e5199b8f178c8b96314b0f7314ae75a6d56e1c`
- Status: **SEALED AND IMMUTABLE**.

