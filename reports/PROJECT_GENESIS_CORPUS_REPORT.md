# Project Genesis: Corpus Architecture & Dataset Report

**Date:** 21 November 2026  
**Project:** ChakrView (ULTIMATE-BRAIN-003)  
**Tokenizer:** `v4096` BPE  

---

## 1. Corpus Inventory & Composition

The training corpus for Project Genesis consists of two foundational datasets:

### Stage 1: Foundational Language Pretraining Corpus (Stage-C)
- **Token Count:** **6,651,093 tokens** (in 27 binary uint16 shards of 250,000 tokens each).
- **Unique Documents:** 4,755 audited documents.
- **Validation Set:** 431,908 tokens.
- **Test Set:** 183,108 tokens.
- **Domain Breakdown:**
  1. English General Knowledge & Science (~35%)
  2. Hindi Devanagari Literature, Essays & Informational Text (~25%)
  3. Hinglish Colloquial Discussions & Explanations (~15%)
  4. Mathematics & Technical Reasoning (~10%)
  5. Python Programming & Computer Science (~10%)
  6. Sanskrit Classical Sentences & Philosophy (~5%)
- **Data Quality:** Exact deduplication, length filtering, valid UTF-8 byte verification, zero synthetic LLM-teacher text.

### Stage 2: Conversational & Reasoning Instruction Dataset (Genesis SFT)
- **Objective:** Teach conversational turn structure, assistant identity, multi-turn follow-ups, and step-by-step reasoning without destroying general linguistic capability.
- **Dataset Size:** 2,500 diverse multi-turn dialogues and reasoning examples:
  - Greetings & natural introductions (English, Hindi, Hinglish).
  - Multi-container word math (boxes, crates, bags, apples, multi-step arithmetic).
  - Contextual reading comprehension with explicit reference handling.
  - Personal history queries with principled abstention.
  - User self-corrections and fact updates.
- **Anti-Catastrophic Forgetting Replay:** Combined with a 20% random replay mixture from Stage-C pretraining data during fine-tuning.
