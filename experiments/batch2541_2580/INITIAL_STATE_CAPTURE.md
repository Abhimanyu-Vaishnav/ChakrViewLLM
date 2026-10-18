# INITIAL STATE CAPTURE REPORT — BATCH 2541–2580

**Batch:** 2541–2580  
**Date:** 2026-10-18  
**Branch:** `dev/v0.6-v0.7-validation`  
**Expected Starting Commit:** `3b30f86`  
**Actual Starting Commit:** `3b30f86` (`feat(batch2501-2540): hardened independent validator, derived coverage proof, 20-case adversarial suite, and preregistration audit`)  
**Commit Alignment Status:** **MATCH (VERIFIED)**  

---

## 1. Working Tree & Git Status

`git status --short`:
```text
 M chakrview/runtime/session.py
?? docs/sprints/...
?? experiments/batch...
```
*(Historical untracked sprint reports and workbench logs remain strictly preserved and untouched).*

### Working Tree Diff for Protected File (`chakrview/runtime/session.py`)
```diff
diff --git a/chakrview/runtime/session.py b/chakrview/runtime/session.py
index 02e685c..79fbd61 100644
--- a/chakrview/runtime/session.py
+++ b/chakrview/runtime/session.py
@@ -314,17 +314,20 @@ class LocalModelSession:
         # Compact history to fit remaining budget
         active_turns = self.context_manager.compact_turns(self.turns, max(0, available_budget))
 
-        # Format turn dialogue into prompt
-        prompt_sections: List[str] = []
+        # Format turn dialogue into canonical ConversationTemplate ChatML format
+        from chakrview.runtime.conversation import ConversationTemplate, ChatMessage, Role
+
+        messages: List[ChatMessage] = []
         for turn in active_turns:
-            if turn.role == ConversationRole.SYSTEM:
-                prompt_sections.append(f"System: {turn.content}")
-            elif turn.role == ConversationRole.USER:
-                prompt_sections.append(f"User: {turn.content}")
-            elif turn.role == ConversationRole.ASSISTANT:
-                prompt_sections.append(f"Assistant: {turn.content}")
-
-        prompt_sections.append(f"User: {user_input}\nAssistant:")
-        final_prompt = "\n".join(prompt_sections)
+            role_map = {
+                ConversationRole.SYSTEM: Role.SYSTEM,
+                ConversationRole.USER: Role.USER,
+                ConversationRole.ASSISTANT: Role.ASSISTANT,
+            }
+            messages.append(ChatMessage(role=role_map[turn.role], content=turn.content))
+
+        # Append current user input
+        messages.append(ChatMessage(role=Role.USER, content=user_input))
+        final_prompt = ConversationTemplate.format_conversation(messages, add_generation_prompt=True)
 
         return final_prompt, envelope, active_turns
```

---

## 2. Environment & Tokenizer Identity

- **Python Environment:** Python 3.14.7 (Windows x86_64).
- **PyTorch Environment:** `2.14.0+cpu` (CPU execution policy strictly active).
- **Canonical Tokenizer:** BPE 4,096 tokens ([vocab.json](file:///d:/Project/ChakrView/release/chakrview-v0.5.0/data/experiments/vocab_4096/vocab.json), SHA-256: `db4d6119...`; [merges.json](file:///d:/Project/ChakrView/release/chakrview-v0.5.0/data/experiments/vocab_4096/merges.json), SHA-256: `ada4b1dc...`).

---

## 3. Cryptographic Hashes of Primary Artifacts

Recorded in [`experiments/batch2541_2580/initial_artifact_hashes.json`](file:///d:/Project/ChakrView/experiments/batch2541_2580/initial_artifact_hashes.json):

| Artifact | Path | SHA-256 Digest | Size (bytes) |
| :--- | :--- | :--- | :--- |
| **Batch 2501 Hardened Validator** | `experiments/batch2501_2540/HARDENED_EVIDENCE_VALIDATOR.py` | `050a4d5386f685935ef7c9ffcc4be9f848eb3aee5d7a6be30b533cb2447d6a5c` | 21,399 |
| **Batch 2501 Hardening Tests** | `experiments/batch2501_2540/test_validator_hardening.py` | `a1fc6c9a9d20c576579b4fcb5177dfc8907f594514a3e74289456ceb1d6446e5` | 10,724 |
| **Batch 2501 Derived Coverage** | `experiments/batch2501_2540/TRAINING_COVERAGE_DERIVED.json` | `aa2648773c2e1f40d0f507ee23c53051fe04f901cb0eaee418be9d8d673187c3` | 2,752 |
| **Batch 2501 Preregistration Audit** | `experiments/batch2501_2540/PREREGISTRATION_AUDIT.json` | `9d554a9918b95fb1102ae8e7a04870f7223789b52a1b15112fa61955b0a39a06` | 2,525 |
| **Batch 2501 Required Changes** | `experiments/batch2501_2540/PREREGISTRATION_REQUIRED_CHANGES.md` | `b6e16f393da00302ce203b2909477e682b1875c7ee0f9d95f0c2a29774a3e6be` | 3,365 |
| **Per-Document Loss Records** | `experiments/batch2381_2420/PER_DOCUMENT_LOSSES.jsonl` | `56ba565fe0a01df867664d50e6a058cbe1a627b833c4517f2987bfc58459373e` | 105,427 |
| **Split V2 Manifest** | `experiments/batch2021_2060/SPLIT_V2_MANIFEST.json` | `00748cbec81c6eb52880f0b74756f2e672d65358143822dde466c9cec4124a31` | 45,759 |
| **Cleaned Corpus Manifest** | `experiments/batch2101_2140/CORPUS_V2_CLEANED_MANIFEST.json` | `e3db033c83f356f653b10d7eb00994da31d6e0738ab4177faddb133e5b8da380` | 1,345,799 |
| **Canonical Step 10,000 Checkpoint** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | 42,212,787 |
| **CPT Seed 42 Checkpoint** | `experiments/batch2141_2180/cpt/seed_42/cpt_best_checkpoint.pt` | `d3268fc49378f8c5b9a30e71d4bab58ce7c4db6bcaaa1f32932ccfd8694b07c3` | 41,402,267 |

Initial state captured and verified against git tree.
