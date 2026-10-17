# FEEDBACK AND CONVERSATION LEARNING DESIGN

**Batch:** 2141–2180  
**Status:** `PROTOTYPE` / `END-TO-END VERIFIED` (Foundation & synthetic demonstration)  
**Target:** Safe, auditable lifelong learning architecture without live parameter updates  

---

## 1. Executive Summary and Non-Negotiable Boundaries

A foundational design requirement of ChakrView is that **conversational interaction must never directly alter canonical model weights in real time**. Live parameter updates from user chat introduce catastrophic forgetting, reward hacking, vulnerability to prompt injection, and untracked regression.

The ChakrView learning architecture strictly separates learning into three decoupled layers:
1. **Layer 1 (In-Context Episodic & Semantic Memory):** Explicitly approved project context and user facts stored in a queryable semantic store for dynamic context injection.
2. **Layer 2 (Auditable Experience & Feedback Ledger):** Immutable transaction ledger recording original generation, user feedback/corrections, ratings, domain provenance, and explicit consent for future training.
3. **Layer 3 (Controlled Offline Training & Promotion Pipeline):** Offline dataset curation, decontamination, leak-free partitioning, separate candidate model training, held-out validation against canonical baselines, and cryptographic release gating.

```
+-----------------------------------------------------------------------------------+
| Live Conversation Session (chakrview/runtime/session.py - PROTECTED)             |
+-----------------------------------------------------------------------------------+
             | (Proposes Memory / Logs Interaction)
             v
+-----------------------------------------------------------------------------------+
| Layer 1: ConversationMemoryStore                                                  |
| - Proposes candidate memory items                                                 |
| - Enforces explicit user consent & approval                                       |
| - Provides semantic/tag-based retrieval for context injection                    |
+-----------------------------------------------------------------------------------+
             | (User feedback & corrections logged)
             v
+-----------------------------------------------------------------------------------+
| Layer 2: FeedbackExperienceLedger                                                 |
| - Captures prompt, model output, user correction, rating                          |
| - Records provenance, timestamps, consent, and eligibility status                 |
| - Rejection & exclusion tracking                                                  |
+-----------------------------------------------------------------------------------+
             | (Export request with review gate)
             v
+-----------------------------------------------------------------------------------+
| Layer 3: ControlledLearningDatasetExporter                                        |
| - Filters strictly for consent=True and training_eligible=True                    |
| - Deduplication, schema validation, domain balance                                |
| - Train / Val / Test splitting without overlap                                    |
| - Versioned JSONL dataset export for offline CPT/SFT                              |
+-----------------------------------------------------------------------------------+
```

---

## 2. Layer 1: Conversation Memory (Episodic & Semantic Store)

### 2.1 Specification
- **Record Structure:**
  - `memory_id`: UUIDv4 string.
  - `timestamp`: UTC ISO-8601 string.
  - `content`: Explicit user fact or project directive.
  - `source`: User ID or session identifier.
  - `tags`: List of category strings (e.g., `["project_config", "hardware"]`).
  - `confidence`: Float between 0.0 and 1.0.
  - `approved_by_user`: Boolean flag.
- **Approval Gate:** By default, items proposed by the agent or session runtime are `approved_by_user=False`. They **cannot** be retrieved into future prompts until explicit approval is granted.
- **Search & Retrieval:** Tag-based and keyword-relevance matching. Unapproved items are filtered out at query time.
- **Correction & Tombstoning:** Memories can be updated in-place with provenance tracking or completely deleted (tombstoned).

---

## 3. Layer 2: Feedback & Experience Ledger

### 3.1 Specification
- **Record Structure:**
  - `feedback_id`: UUIDv4 string.
  - `timestamp`: UTC ISO-8601 string.
  - `prompt`: Input prompt text.
  - `model_output`: Original generated text.
  - `user_correction`: Optional ground-truth correction or refined instruction.
  - `rating`: Optional integer rating (-1 to +1 or 1 to 5).
  - `domain`: Source domain classification.
  - `source_session`: Session identifier.
  - `consent_given`: Boolean flag indicating user consent to retain data.
  - `training_eligible`: Boolean flag indicating explicit qualification for training.
  - `exclusion_reason`: Text description if excluded (e.g., `"insufficient_length"`, `"pii_detected"`, `"no_consent"`).
- **Safety Invariant:** A record with `consent_given=False` is permanently flagged `training_eligible=False` with `exclusion_reason="NO_USER_CONSENT"`.

---

## 4. Layer 3: Controlled Model Learning & Dataset Export

### 4.1 Specification
- **Exporter Responsibilities:**
  1. **Strict Gate Enforcement:** Exclude any item lacking `consent_given=True` or `training_eligible=True`.
  2. **Deduplication:** Hash-based exact deduplication on prompt + target content.
  3. **Disjoint Partitioning:** Deterministic train/validation/test split by document hash to prevent leakage.
  4. **Corpus Formatting:** Converts approved paired records into autoregressive CPT/SFT sequences formatted for the canonical tokenizer (`vocab_4096`).
  5. **Audit Metadata:** Generates dataset manifest containing record counts, domain distributions, and SHA-256 hashes of generated splits.

---

## 5. Session Integration & Adapter Pattern

In accordance with non-negotiable project constraints, `chakrview/runtime/session.py` was **not modified**.

Instead, an independent adapter interface (`SessionFeedbackAdapter` in `experiments/batch2141_2180/feedback_learning_engine.py`) wraps any active session, consuming session events and proposing memory/feedback items without mutating session internals.

```python
from experiments.batch2141_2180.feedback_learning_engine import (
    ConversationMemoryStore,
    FeedbackExperienceLedger,
    ControlledLearningDatasetExporter
)

# Initialize isolated stores
memory_store = ConversationMemoryStore(storage_path="path/to/memories.json")
feedback_ledger = FeedbackExperienceLedger(storage_path="path/to/feedback.json")
exporter = ControlledLearningDatasetExporter(feedback_ledger=feedback_ledger)

# Propose memory (requires user consent to approve)
item = memory_store.propose_memory("User prefers 256 context window", tags=["user_pref"])
memory_store.approve_memory(item.memory_id)

# Log feedback with explicit consent
fb = feedback_ledger.record_feedback(
    prompt="Explain RoPE",
    model_output="RoPE stands for...",
    user_correction="Rotary Position Embedding encodes position...",
    rating=1,
    domain="deep_learning",
    consent_given=True,
    training_eligible=True
)

# Export offline dataset
manifest = exporter.export_training_dataset(output_dir="path/to/cpt_dataset")
```

---

## 6. Verification and Demonstration Evidence

A synthetic verification script and comprehensive unit test suite confirm:
1. `test_memory_approval_gate`: Unapproved memories cannot be retrieved.
2. `test_feedback_consent_gate`: Records without consent cannot be marked eligible for training.
3. `test_dataset_export_disjoint_splits`: Exported splits have 0 token or prompt overlap and adhere to schema integrity.
4. `test_memory_deletion`: Deleting a memory completely removes it from query results.

**Demonstrated Maturity Level:** `PROTOTYPE` / `END-TO-END VERIFIED` (Verified with synthetic fixtures and automated test suite; no live model weights mutated).
