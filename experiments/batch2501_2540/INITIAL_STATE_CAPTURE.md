# INITIAL STATE CAPTURE REPORT — BATCH 2501–2540

**Batch:** 2501–2540  
**Date:** 2026-10-18  
**Branch:** `dev/v0.6-v0.7-validation`  
**Expected Previous Commit:** `92a34cf`  
**Actual Previous Commit:** `92a34cf` (`feat(batch2461-2500): independent evidence validator, adversarial test suite, and sanskrit curriculum preregistration`)  
**Commit Alignment Status:** **MATCH (VERIFIED)**  

---

## 1. Working Tree & Git Status

`git status --short`:
```text
 M chakrview/runtime/session.py
?? docs/sprints/...
?? experiments/batch...
```
*(All historical untracked workbench runs and sprint reports remain strictly preserved and untouched).*

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
The file is uncommitted, unstaged, and preserved without modification.

---

## 2. Environment & Execution Hardware

Recorded in [`experiments/batch2501_2540/environment_capture.json`](file:///d:/Project/ChakrView/experiments/batch2501_2540/environment_capture.json):
- **Python Version:** Python 3.14.7 (Windows-11-10.0.26100-SP0)
- **PyTorch Version:** `2.14.0+cpu`
- **CUDA Acceleration:** `torch.cuda.is_available() == False` (All operations executed deterministically on CPU).

---

## 3. Cryptographic Hashes of Primary Artifacts

Recorded in [`experiments/batch2501_2540/initial_artifact_hashes.json`](file:///d:/Project/ChakrView/experiments/batch2501_2540/initial_artifact_hashes.json):

| Artifact | Path | SHA-256 Digest | Size (bytes) |
| :--- | :--- | :--- | :--- |
| **Batch 2461 Validator** | `experiments/batch2461_2500/INDEPENDENT_EVIDENCE_VALIDATOR.py` | `d6df6c990664dcf2b63577742878d6729094fe525b66d405232a9e38d6174fd5` | 17,998 |
| **Batch 2461 Audit JSON** | `experiments/batch2461_2500/INDEPENDENT_EVIDENCE_AUDIT.json` | `aa2648773c2e1f40d0f507ee23c53051fe04f901cb0eaee418be9d8d673187c3` | 3,939 |
| **Batch 2461 Regression Recheck** | `experiments/batch2461_2500/HELDOUT_REGRESSION_FINAL_RECHECK.json` | `f3ea70440ef460fe30085db3be8c42a59a9366be926588ba54032d9dbca5e836` | 3,252 |
| **Batch 2461 Coverage Recheck** | `experiments/batch2461_2500/TRAINING_COVERAGE_FINAL_RECHECK.json` | `1ffb6e6cf1e858db14b0949cb4a23e74567ecb575727e46ea9eb451639c09939` | 2,752 |
| **Batch 2461 Preregistration** | `experiments/batch2461_2500/SANSKRIT_CURRICULUM_PREREGISTRATION.json` | `f6a4c21975ae7da07498c47466847849e759160a28f413346b0e9e7cae2fb60a` | 2,693 |
| **Raw Per-Document Losses** | `experiments/batch2381_2420/PER_DOCUMENT_LOSSES.jsonl` | `56ba565fe0a01df867664d50e6a058cbe1a627b833c4517f2987bfc58459373e` | 105,427 |
| **Split V2 Manifest** | `experiments/batch2021_2060/SPLIT_V2_MANIFEST.json` | `00748cbec81c6eb52880f0b74756f2e672d65358143822dde466c9cec4124a31` | 45,759 |
| **Cleaned Corpus Manifest** | `experiments/batch2101_2140/CORPUS_V2_CLEANED_MANIFEST.json` | `e3db033c83f356f653b10d7eb00994da31d6e0738ab4177faddb133e5b8da380` | 1,345,799 |
| **Canonical Step 10,000 Checkpoint** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | 42,212,787 |
| **CPT Seed 42 Checkpoint** | `experiments/batch2141_2180/cpt/seed_42/cpt_best_checkpoint.pt` | `d3268fc49378f8c5b9a30e71d4bab58ce7c4db6bcaaa1f32932ccfd8694b07c3` | 41,402,267 |
| **Canonical Tokenizer Vocab** | `release/chakrview-v0.5.0/data/experiments/vocab_4096/vocab.json` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | 112,627 |
| **Canonical Tokenizer Merges** | `release/chakrview-v0.5.0/data/experiments/vocab_4096/merges.json` | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | 76,299 |

No files modified. Starting state recorded cleanly.
