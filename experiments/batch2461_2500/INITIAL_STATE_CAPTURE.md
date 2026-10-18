# PHASE 0 — INITIAL STATE CAPTURE REPORT
**Batch:** 2461–2500  
**Date:** 2026-10-18  
**Branch:** `dev/v0.6-v0.7-validation`  
**HEAD Commit:** `54c09de` (`feat(batch2421-2460): evidence-gated verdict repair, dynamic gate evaluator, and sanskrit coverage verification`)  

---

## 1. Git State & Working Tree Inventory

### Git Status (`git status --short`)
```text
 M chakrview/runtime/session.py
?? docs/sprints/...
?? experiments/batch...
```
*(All prior untracked scratch workbench reports and sprint notes are preserved untouched).*

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

## 2. Cryptographic Checksums of Primary Artifacts

| Artifact Name | Path | SHA-256 Digest | File Size (bytes) |
| :--- | :--- | :--- | :--- |
| **Canonical Step-10,000 Checkpoint** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | 42,212,787 |
| **CPT Candidate Seed 42** | `experiments/batch2141_2180/cpt/seed_42/cpt_best_checkpoint.pt` | `d3268fc49378f8c5b9a30e71d4bab58ce7c4db6bcaaa1f32932ccfd8694b07c3` | 41,402,267 |
| **Cleaned Corpus Manifest** | `experiments/batch2101_2140/CORPUS_V2_CLEANED_MANIFEST.json` | `e3db033c83f356f653b10d7eb00994da31d6e0738ab4177faddb133e5b8da380` | 1,345,799 |
| **Split V2 Manifest** | `experiments/batch2021_2060/SPLIT_V2_MANIFEST.json` | `00748cbec81c6eb52880f0b74756f2e672d65358143822dde466c9cec4124a31` | 45,759 |
| **Canonical Tokenizer Vocab** | `release/chakrview-v0.5.0/data/experiments/vocab_4096/vocab.json` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | 112,627 |
| **Canonical Tokenizer Merges** | `release/chakrview-v0.5.0/data/experiments/vocab_4096/merges.json` | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | 76,299 |
| **Per-Document Losses (Raw Records)** | `experiments/batch2381_2420/PER_DOCUMENT_LOSSES.jsonl` | `56ba565fe0a01df867664d50e6a058cbe1a627b833c4517f2987bfc58459373e` | 105,427 |
| **Batch 2421 Gate Evaluator** | `experiments/batch2421_2460/evaluate_evidence_gates.py` | `356f3ee253658e8bfeb5265d864414d05bc875ab01f7c47c32f5691bd561dc77` | 19,959 |
| **Batch 2421 Final Evidence** | `experiments/batch2421_2460/FINAL_EVIDENCE.json` | `a523248734862e7b0381b742a669e841cecde7d84855db18482f1d198773f2d9` | 3,372 |

All SHA-256 hashes recorded. No files modified.
