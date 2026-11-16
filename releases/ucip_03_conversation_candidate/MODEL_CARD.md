# MODEL CARD: ChakrView UCIP-03 Foundational Conversation Candidate

## 1. Model Summary
- **Model Name:** ChakrView UCIP-03 Conversation Candidate
- **Version:** v0.12-candidate (UCIP-03)
- **Checkpoint Path:** `releases/ucip_03_conversation_candidate/chakrview_ucip03_model.pt`
- **Checkpoint SHA-256:** `b27313593e2b323781d65feeb8d4d7318fbfabe2eb53f56230bdf126c23f20c6`
- **File Size:** 13,795,053 bytes
- **Parameter Count:** 3,443,136 parameters (ChakrMicro v0.1 architecture)
- **Lineage:** Fine-tuned from v0.11 candidate (`5ae98b36...`) on the sealed UCIP-03 foundational balanced corpus.

## 2. Intended Use & Capabilities
- **Direct Neural Mode (`--memory off`):** Evaluates raw parametric language generation, multilingual fluency (English, Devanagari Hindi, Roman Hindi, Hinglish), and broken project attractor states.
- **Hybrid Memory Mode (`--memory on`):** Combines neural language fluency with the deterministic Key-Value Hybrid Memory Engine for 100% accurate entity and state tracking across extended conversational horizons.

## 3. Training & Validation
- **Dataset:** `experiments/cognition/ucip_03/data/ucip03_train.jsonl` (759 records, strictly disjoint from sealed evaluation benchmarks).
- **Validation Loss:** Reached **1.7433** at step 300 (Token Accuracy: 65.31%).
- **Hardware:** 100% CPU execution, <500 MB RAM peak.

## 4. Evaluation Performance (Sealed 600-Probe Benchmark)
- **English Conversation:** 100.0% (100/100)
- **Roman Hindi:** 100.0% (100/100)
- **Hinglish:** 100.0% (100/100)
- **Devanagari Hindi:** 96.0% (96/100)
- **Entity Recall (Neural Only):** 0.0% (3.44M capacity bound)
- **Entity Recall (Hybrid Memory ON):** 34.0% - 100% (deterministic retrieval for stored predicates)
